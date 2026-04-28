from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .service import ClipJobSubmission, ClipWebService, create_default_service


class SessionTokenPayload(BaseModel):
    token: str = Field(min_length=1)


def create_app(service: ClipWebService | None = None) -> FastAPI:
    web_service = service or create_default_service()
    app = FastAPI(title="openpilot Replay Clipper", version="0.1.0")
    app.state.service = web_service
    app.mount("/outputs", StaticFiles(directory=str(web_service.shared_dir)), name="outputs")

    index_path = Path(__file__).with_name("index.html")

    @app.get("/", include_in_schema=False)
    def read_index() -> FileResponse:
        return FileResponse(index_path)

    @app.get("/api/options")
    def read_options() -> dict[str, object]:
        return web_service.options_manifest()

    @app.get("/api/auth")
    def read_auth() -> dict[str, object]:
        return web_service.auth_status()

    @app.put("/api/auth/session-token")
    def put_session_token(payload: SessionTokenPayload) -> dict[str, object]:
        web_service.set_session_token(payload.token)
        return web_service.auth_status()

    @app.delete("/api/auth/session-token", status_code=status.HTTP_204_NO_CONTENT)
    def delete_session_token() -> Response:
        web_service.clear_session_token()
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/api/jobs")
    def read_jobs() -> list[dict[str, object]]:
        return web_service.list_jobs()

    @app.get("/api/jobs/{job_id}")
    def read_job(job_id: str) -> dict[str, object]:
        job = web_service.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.")
        return job

    @app.post("/api/jobs", status_code=status.HTTP_201_CREATED)
    def create_job(payload: ClipJobSubmission) -> dict[str, object]:
        try:
            return web_service.submit_job(payload)
        except ValueError as error:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error

    @app.get("/api/outputs")
    def read_outputs() -> list[dict[str, object]]:
        return web_service.list_outputs()

    return app
