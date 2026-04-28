from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from webui.app import create_app
from webui.service import ClipJobSubmission, ClipWebService, route_filename_stem


def _fake_plan_builder(request) -> SimpleNamespace:
    return SimpleNamespace(route="e99064ba80a1dc40|00000006--88ee9aafbf")


def _fake_runner(job, repo_root: Path, on_log) -> None:
    on_log("fake renderer started")
    job.output_path.write_bytes(b"mp4")
    on_log("fake renderer finished")


def _wait_for_job(service: ClipWebService, job_id: str) -> dict[str, object]:
    for _ in range(100):
        snapshot = service.get_job(job_id)
        assert snapshot is not None
        if snapshot["state"] in {"succeeded", "failed"}:
            return snapshot
        time.sleep(0.02)
    raise AssertionError("job did not finish in time")


def test_route_filename_stem_is_windows_safe() -> None:
    assert route_filename_stem("e99064ba80a1dc40|00000006--88ee9aafbf") == "e99064ba80a1dc40_00000006--88ee9aafbf"


def test_service_processes_job_and_hides_session_token(tmp_path: Path) -> None:
    service = ClipWebService(
        repo_root=tmp_path,
        shared_dir=tmp_path / "shared",
        plan_builder=_fake_plan_builder,
        runner=_fake_runner,
    )
    service.set_session_token("secret-token")
    created = service.submit_job(ClipJobSubmission(route_input="e99064ba80a1dc40|00000006--88ee9aafbf", render_type="wide"))

    finished = _wait_for_job(service, str(created["id"]))

    assert finished["state"] == "succeeded"
    assert finished["output_name"] == "e99064ba80a1dc40_00000006--88ee9aafbf.mp4"
    assert "secret-token" not in str(finished)
    assert service.list_outputs()[0]["name"] == "e99064ba80a1dc40_00000006--88ee9aafbf.mp4"


def test_app_auth_and_job_routes(tmp_path: Path) -> None:
    service = ClipWebService(
        repo_root=tmp_path,
        shared_dir=tmp_path / "shared",
        plan_builder=_fake_plan_builder,
        runner=_fake_runner,
    )
    client = TestClient(create_app(service))

    assert client.get("/api/auth").json()["has_session_token"] is False
    assert client.put("/api/auth/session-token", json={"token": "jwt-value"}).json()["has_session_token"] is True

    response = client.post(
        "/api/jobs",
        json={
            "route_input": "e99064ba80a1dc40|00000006--88ee9aafbf",
            "render_type": "ui-alt",
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert "jwt" not in payload
    finished = _wait_for_job(service, payload["id"])
    assert finished["state"] == "succeeded"

    outputs = client.get("/api/outputs").json()
    assert outputs[0]["name"] == "e99064ba80a1dc40_00000006--88ee9aafbf.mp4"

    clear = client.delete("/api/auth/session-token")
    assert clear.status_code == 204
    assert client.get("/api/auth").json()["has_session_token"] is False
