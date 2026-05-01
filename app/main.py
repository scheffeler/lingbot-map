"""FastAPI dashboard entrypoint for polevision."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import captures, db, measurements, runs, scene, test_runner, viewer_proc

VERSION = "0.1.0"
STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="PoleVision", version=VERSION)
    db.init_schema()

    # Static assets (CSS/JS/images that live alongside the SPA).
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "version": VERSION,
            "db_path": db.db_path(),
        }

    @app.get("/api/captures")
    def list_captures() -> dict:
        return {"captures": captures.list_captures()}

    @app.post("/api/captures/upload")
    async def upload_capture(
        name: str = Form(...),
        file: UploadFile = File(...),
    ) -> dict:
        body = await file.read()
        try:
            return captures.upload_zip(name, body)
        except FileExistsError as e:
            raise HTTPException(status_code=409, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/captures/{capture_id}")
    def get_capture(capture_id: int) -> dict:
        cap = captures.get_capture(capture_id)
        if cap is None:
            raise HTTPException(status_code=404, detail="capture not found")
        return cap

    @app.post("/api/captures/{capture_id}/run/{stage}")
    async def start_run(capture_id: int, stage: str, request: Request) -> dict:
        if stage not in ("exif", "pose", "sam", "scale", "fit"):
            raise HTTPException(status_code=400, detail=f"unknown stage: {stage}")
        # Optional JSON body: {"params": {...}}. Allow no-body too.
        body = {}
        try:
            if request.headers.get("content-length") not in (None, "0"):
                body = await request.json()
        except Exception:
            body = {}
        params = body.get("params") if isinstance(body, dict) else None
        try:
            return runs.start_run(capture_id, stage, params=params)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: int) -> dict:
        try:
            return runs.cancel_run(run_id)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: int) -> dict:
        run = runs.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return run

    @app.get("/api/captures/{capture_id}/runs")
    def list_capture_runs(capture_id: int) -> dict:
        return {"runs": runs.list_runs(capture_id)}

    @app.get("/api/captures/{capture_id}/measurements")
    def get_measurements(capture_id: int) -> dict:
        cap = captures.get_capture(capture_id)
        if cap is None:
            raise HTTPException(status_code=404, detail="capture not found")
        return {"measurements": measurements.measurements_for(cap["name"])}

    @app.get("/api/captures/{capture_id}/scene")
    def get_scene(capture_id: int) -> dict:
        cap = captures.get_capture(capture_id)
        if cap is None:
            raise HTTPException(status_code=404, detail="capture not found")
        s = scene.build_scene(cap["name"], capture_id=capture_id)
        if s is None:
            raise HTTPException(
                status_code=404,
                detail=("pose stage hasn't run yet — no poses.npz for "
                        "this capture. Run the pose stage first."),
            )
        return s

    @app.get("/api/captures/{capture_id}/ply")
    def get_ply(capture_id: int):
        cap = captures.get_capture(capture_id)
        if cap is None:
            raise HTTPException(status_code=404, detail="capture not found")
        path = scene.ply_path(cap["name"])
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail="PLY not found; run the pose stage first.",
            )
        return FileResponse(
            str(path),
            media_type="application/octet-stream",
            filename=path.name,
        )

    @app.get("/api/captures/{capture_id}/frame/{idx}")
    def get_frame(capture_id: int, idx: int):
        from fastapi.responses import Response
        cap = captures.get_capture(capture_id)
        if cap is None:
            raise HTTPException(status_code=404, detail="capture not found")
        data = scene.frame_jpeg_bytes(cap["name"], idx)
        if data is None:
            raise HTTPException(
                status_code=404,
                detail=f"frame {idx} not found for {cap['name']}",
            )
        return Response(content=data, media_type="image/jpeg")

    @app.get("/api/captures/{capture_id}/mask-viewer")
    def get_mask_viewer(capture_id: int):
        from fastapi.responses import HTMLResponse
        from pathlib import Path
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from visualize_masks import generate_html

        cap = captures.get_capture(capture_id)
        if cap is None:
            raise HTTPException(status_code=404, detail="capture not found")
        ws = captures.workspace_path()
        masks_path = ws / f"{cap['name']}.masks.npz"
        if not masks_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"masks NPZ not found; run the SAM stage first "
                       f"(expected at {masks_path})",
            )
        gps_scale_path = ws / f"{cap['name']}.scale.json"
        try:
            html = generate_html(
                masks_path=str(masks_path),
                image_folder=cap["folder_path"],
                gps_scale_path=str(gps_scale_path) if gps_scale_path.exists() else None,
                quiet=True,
            )
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        return HTMLResponse(content=html)

    @app.get("/api/runs/{run_id}/stream")
    async def stream_run(run_id: int) -> StreamingResponse:
        return StreamingResponse(
            runs.stream_run_log(run_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache",
                     "X-Accel-Buffering": "no"},
        )

    @app.post("/api/test")
    def post_test() -> dict:
        # Hits the dispatcher in app/test_runner. Tests monkeypatch
        # `_run_pytest_subprocess` to return a stubbed summary.
        return test_runner.run_tests()

    @app.post("/api/captures/{capture_id}/view-3d")
    def post_view_3d(capture_id: int) -> dict:
        cap = captures.get_capture(capture_id)
        if cap is None:
            raise HTTPException(status_code=404, detail="capture not found")
        try:
            return viewer_proc.start_for_capture(cap)
        except FileNotFoundError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/viewer/status")
    def get_viewer_status() -> dict:
        return viewer_proc.status()

    @app.post("/api/viewer/stop")
    def post_viewer_stop() -> dict:
        viewer_proc.stop()
        return {"running": False}

    return app


app = create_app()
