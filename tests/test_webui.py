from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from webui.app import create_app
from webui.service import (
    ClipJobSubmission,
    ClipWebService,
    _parse_route_duration_seconds,
    _normalize_route_input,
    route_filename_stem,
)


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
        route_length_resolver=lambda route, jwt_token: 630,
        runner=_fake_runner,
    )
    service.set_session_token("secret-token")
    created = service.submit_job(ClipJobSubmission(route_input="e99064ba80a1dc40|00000006--88ee9aafbf", render_type="wide"))

    finished = _wait_for_job(service, str(created["id"]))

    assert finished["state"] == "succeeded"
    assert finished["output_name"] == "e99064ba80a1dc40_00000006--88ee9aafbf.mp4"
    assert "secret-token" not in str(finished)
    assert service.list_outputs()[0]["name"] == "e99064ba80a1dc40_00000006--88ee9aafbf.mp4"


def test_service_persists_and_reloads_session_token(tmp_path: Path) -> None:
    service = ClipWebService(
        repo_root=tmp_path,
        shared_dir=tmp_path / "shared",
        plan_builder=_fake_plan_builder,
        route_length_resolver=lambda route, jwt_token: 630,
        runner=_fake_runner,
    )

    service.set_session_token("persisted-token")

    token_file = tmp_path / ".cache/webui/comma.jwt"
    assert token_file.read_text(encoding="utf-8").strip() == "persisted-token"
    assert service.auth_status()["token_storage"] == "repo_local_ignored_cache"
    assert service.auth_status()["token_path"] == ".cache/webui/comma.jwt"

    reloaded = ClipWebService(
        repo_root=tmp_path,
        shared_dir=tmp_path / "shared",
        plan_builder=_fake_plan_builder,
        runner=_fake_runner,
    )

    assert reloaded.auth_status()["has_session_token"] is True
    reloaded.clear_session_token()
    assert token_file.exists() is False
    assert reloaded.auth_status()["has_session_token"] is False


def test_service_defaults_raw_route_to_full_length(tmp_path: Path) -> None:
    observed = {}

    def _runner(job, repo_root: Path, on_log) -> None:
        observed["start_seconds"] = job.request.start_seconds
        observed["length_seconds"] = job.request.length_seconds
        job.output_path.write_bytes(b"mp4")

    service = ClipWebService(
        repo_root=tmp_path,
        shared_dir=tmp_path / "shared",
        plan_builder=_fake_plan_builder,
        route_length_resolver=lambda route, jwt_token: 630,
        runner=_runner,
    )

    created = service.submit_job(ClipJobSubmission(route_input="e99064ba80a1dc40|00000006--88ee9aafbf", render_type="wide"))
    finished = _wait_for_job(service, str(created["id"]))

    assert finished["state"] == "succeeded"
    assert observed == {"start_seconds": 0, "length_seconds": 630}
    assert any("Using the full route length of 630 seconds by default." in line for line in finished["logs"])


def test_service_preserves_url_embedded_timing_defaults(tmp_path: Path) -> None:
    def _runner(job, repo_root: Path, on_log) -> None:
        job.output_path.write_bytes(b"mp4")

    def _fail_if_called(route: str, jwt_token: str | None) -> int:
        raise AssertionError("raw-route full-length resolver should not run for clip URLs")

    service = ClipWebService(
        repo_root=tmp_path,
        shared_dir=tmp_path / "shared",
        plan_builder=lambda request: SimpleNamespace(route="e99064ba80a1dc40|00000006--88ee9aafbf"),
        route_length_resolver=_fail_if_called,
        runner=_runner,
    )

    created = service.submit_job(
        ClipJobSubmission(
            route_input="https://connect.comma.ai/e99064ba80a1dc40/00000006--88ee9aafbf/21/90",
            render_type="wide",
        )
    )
    finished = _wait_for_job(service, str(created["id"]))

    assert finished["state"] == "succeeded"
    assert all("Using the full route length" not in line for line in finished["logs"])


def test_parse_route_duration_prefers_route_start_and_end_times() -> None:
    assert _parse_route_duration_seconds({"start_time": "2026-02-25T17:30:30", "end_time": "2026-02-25T17:45:55"}) == 925
    assert _parse_route_duration_seconds({"maxqlog": 15}) == 960


def test_normalize_route_input_accepts_slash_route_id() -> None:
    normalized = _normalize_route_input(
        ClipJobSubmission(route_input="e99064ba80a1dc40/00000006--88ee9aafbf", render_type="wide")
    )
    assert normalized.route_input == "e99064ba80a1dc40|00000006--88ee9aafbf"


def test_normalize_route_input_accepts_slash_route_with_start() -> None:
    normalized = _normalize_route_input(
        ClipJobSubmission(route_input="e99064ba80a1dc40/00000006--88ee9aafbf/11", render_type="wide")
    )
    assert normalized.route_input == "e99064ba80a1dc40|00000006--88ee9aafbf"
    assert normalized.start_seconds == 11
    assert normalized.length_seconds == 20


def test_normalize_route_input_accepts_hostless_connect_clip_url() -> None:
    normalized = _normalize_route_input(
        ClipJobSubmission(route_input="e99064ba80a1dc40/00000006--88ee9aafbf/21/90", render_type="wide")
    )
    assert normalized.route_input == "https://connect.comma.ai/e99064ba80a1dc40/00000006--88ee9aafbf/21/90"


def test_app_auth_and_job_routes(tmp_path: Path) -> None:
    service = ClipWebService(
        repo_root=tmp_path,
        shared_dir=tmp_path / "shared",
        plan_builder=_fake_plan_builder,
        route_length_resolver=lambda route, jwt_token: 630,
        runner=_fake_runner,
    )
    client = TestClient(create_app(service))

    assert client.get("/api/auth").json()["has_session_token"] is False
    auth_after_put = client.put("/api/auth/session-token", json={"token": "jwt-value"}).json()
    assert auth_after_put["has_session_token"] is True
    assert auth_after_put["token_storage"] == "repo_local_ignored_cache"
    assert auth_after_put["token_path"] == ".cache/webui/comma.jwt"

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
