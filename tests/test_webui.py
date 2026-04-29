from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from webui.app import create_app
from webui.service import (
    COMMA_JWT_MIN_LENGTH,
    ClipJobSubmission,
    ClipWebService,
    _bookmark_python_bin,
    _parse_route_duration_seconds,
    _normalize_route_input,
    has_valid_session_token,
    route_filename_stem,
)

VALID_TEST_TOKEN = "x" * COMMA_JWT_MIN_LENGTH


def _fake_plan_builder(request) -> SimpleNamespace:
    return SimpleNamespace(route="e99064ba80a1dc40|00000006--88ee9aafbf")


def _fake_runner(job, repo_root: Path, on_log) -> None:
    on_log("fake renderer started")
    job.output_path.write_bytes(b"mp4")
    on_log("fake renderer finished")


def _fake_bookmark_resolver(route: str, jwt_token: str | None, data_root: Path, openpilot_dir: Path, route_length_seconds: int) -> list[int]:
    return [30, 120]


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


def test_bookmark_python_bin_prefers_openpilot_venv(tmp_path: Path) -> None:
    venv_python = tmp_path / ".venv/bin/python"
    venv_python.parent.mkdir(parents=True, exist_ok=True)
    venv_python.write_text("", encoding="utf-8")

    assert _bookmark_python_bin(tmp_path) == str(venv_python)


def test_service_processes_job_and_hides_session_token(tmp_path: Path) -> None:
    service = ClipWebService(
        repo_root=tmp_path,
        shared_dir=tmp_path / "shared",
        plan_builder=_fake_plan_builder,
        route_length_resolver=lambda route, jwt_token: 630,
        runner=_fake_runner,
    )
    service.set_session_token(VALID_TEST_TOKEN)
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

    service.set_session_token(VALID_TEST_TOKEN)

    token_file = tmp_path / ".cache/webui/comma.jwt"
    assert token_file.read_text(encoding="utf-8").strip() == VALID_TEST_TOKEN
    assert service.auth_status()["token_storage"] == "repo_local_ignored_cache"
    assert service.auth_status()["token_path"] == ".cache/webui/comma.jwt"
    assert service.auth_status()["has_valid_session_token"] is True

    reloaded = ClipWebService(
        repo_root=tmp_path,
        shared_dir=tmp_path / "shared",
        plan_builder=_fake_plan_builder,
        runner=_fake_runner,
    )

    assert reloaded.auth_status()["has_session_token"] is True
    assert reloaded.auth_status()["has_valid_session_token"] is True
    reloaded.clear_session_token()
    assert token_file.exists() is False
    assert reloaded.auth_status()["has_session_token"] is False
    assert reloaded.auth_status()["has_valid_session_token"] is False


def test_invalid_session_token_stays_visible_for_reentry(tmp_path: Path) -> None:
    token_file = tmp_path / ".cache/webui/comma.jwt"
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text("short-token\n", encoding="utf-8")

    service = ClipWebService(
        repo_root=tmp_path,
        shared_dir=tmp_path / "shared",
        plan_builder=_fake_plan_builder,
        runner=_fake_runner,
    )

    assert service.auth_status()["has_session_token"] is True
    assert service.auth_status()["has_valid_session_token"] is False


def test_set_session_token_rejects_short_values(tmp_path: Path) -> None:
    service = ClipWebService(
        repo_root=tmp_path,
        shared_dir=tmp_path / "shared",
        plan_builder=_fake_plan_builder,
        runner=_fake_runner,
    )

    try:
        service.set_session_token("short-token")
    except ValueError as error:
        assert "looks incomplete" in str(error)
    else:
        raise AssertionError("Expected short token to be rejected")


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


def test_service_queues_multiple_bookmark_clips(tmp_path: Path) -> None:
    observed: list[tuple[int, int, str]] = []

    def _runner(job, repo_root: Path, on_log) -> None:
        observed.append((job.request.start_seconds, job.request.length_seconds, job.output_path.name))
        job.output_path.write_bytes(b"mp4")

    service = ClipWebService(
        repo_root=tmp_path,
        shared_dir=tmp_path / "shared",
        plan_builder=_fake_plan_builder,
        route_length_resolver=lambda route, jwt_token: 180,
        bookmark_resolver=_fake_bookmark_resolver,
        runner=_runner,
    )

    created = service.submit_job(
        ClipJobSubmission(
            route_input="e99064ba80a1dc40|00000006--88ee9aafbf",
            render_type="wide",
            use_bookmarks=True,
            bookmark_padding_seconds=15,
        )
    )

    assert created["job_count"] == 2
    assert created["bookmark_times_seconds"] == [30, 120]
    snapshots = created["jobs"]
    assert isinstance(snapshots, list)
    first = _wait_for_job(service, str(snapshots[0]["id"]))
    second = _wait_for_job(service, str(snapshots[1]["id"]))

    assert first["state"] == "succeeded"
    assert second["state"] == "succeeded"
    assert observed == [
        (15, 30, "e99064ba80a1dc40_00000006--88ee9aafbf_BM1.mp4"),
        (105, 30, "e99064ba80a1dc40_00000006--88ee9aafbf_BM2.mp4"),
    ]


def test_service_rejects_bookmark_mode_without_bookmarks(tmp_path: Path) -> None:
    service = ClipWebService(
        repo_root=tmp_path,
        shared_dir=tmp_path / "shared",
        plan_builder=_fake_plan_builder,
        route_length_resolver=lambda route, jwt_token: 180,
        bookmark_resolver=lambda route, jwt_token, data_root, openpilot_dir, route_length_seconds: [],
        runner=_fake_runner,
    )

    try:
        service.submit_job(
            ClipJobSubmission(
                route_input="e99064ba80a1dc40|00000006--88ee9aafbf",
                render_type="wide",
                use_bookmarks=True,
            )
        )
    except ValueError as error:
        assert "No bookmarks were found" in str(error)
    else:
        raise AssertionError("Expected missing bookmarks to be rejected")


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


def test_service_expands_start_only_route_input_to_route_end(tmp_path: Path) -> None:
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

    created = service.submit_job(
        ClipJobSubmission(route_input="e99064ba80a1dc40/00000006--88ee9aafbf/0", render_type="wide")
    )
    finished = _wait_for_job(service, str(created["id"]))

    assert finished["state"] == "succeeded"
    assert observed == {"start_seconds": 0, "length_seconds": 630}
    assert any("Using the remaining route length of 630 seconds from 0 seconds by default." in line for line in finished["logs"])


def test_parse_route_duration_prefers_route_start_and_end_times() -> None:
    assert _parse_route_duration_seconds({"start_time": "2026-02-25T17:30:30", "end_time": "2026-02-25T17:45:55"}) == 925
    assert _parse_route_duration_seconds({"maxqlog": 15}) == 960
    assert has_valid_session_token(VALID_TEST_TOKEN) is True
    assert has_valid_session_token("short-token") is False


def test_normalize_route_input_accepts_slash_route_id() -> None:
    normalized = _normalize_route_input(
        ClipJobSubmission(route_input="e99064ba80a1dc40/00000006--88ee9aafbf", render_type="wide")
    )
    assert normalized.submission.route_input == "e99064ba80a1dc40|00000006--88ee9aafbf"
    assert normalized.expand_start_only_to_route_end is False


def test_normalize_route_input_accepts_slash_route_with_start() -> None:
    normalized = _normalize_route_input(
        ClipJobSubmission(route_input="e99064ba80a1dc40/00000006--88ee9aafbf/11", render_type="wide")
    )
    assert normalized.submission.route_input == "e99064ba80a1dc40|00000006--88ee9aafbf"
    assert normalized.submission.start_seconds == 11
    assert normalized.submission.length_seconds == 20
    assert normalized.expand_start_only_to_route_end is True


def test_normalize_route_input_accepts_bare_connect_route_url() -> None:
    normalized = _normalize_route_input(
        ClipJobSubmission(
            route_input="https://connect.comma.ai/e99064ba80a1dc40/0000000a--e9c896d5fd",
            render_type="wide",
        )
    )
    assert normalized.submission.route_input == "e99064ba80a1dc40|0000000a--e9c896d5fd"
    assert normalized.expand_start_only_to_route_end is False


def test_normalize_route_input_accepts_hostless_connect_clip_url() -> None:
    normalized = _normalize_route_input(
        ClipJobSubmission(route_input="e99064ba80a1dc40/00000006--88ee9aafbf/21/90", render_type="wide")
    )
    assert normalized.submission.route_input == "https://connect.comma.ai/e99064ba80a1dc40/00000006--88ee9aafbf/21/90"
    assert normalized.expand_start_only_to_route_end is False


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
    auth_after_put = client.put("/api/auth/session-token", json={"token": VALID_TEST_TOKEN}).json()
    assert auth_after_put["has_session_token"] is True
    assert auth_after_put["has_valid_session_token"] is True
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
    assert payload["job_count"] == 1
    finished = _wait_for_job(service, payload["id"])
    assert finished["state"] == "succeeded"

    outputs = client.get("/api/outputs").json()
    assert outputs[0]["name"] == "e99064ba80a1dc40_00000006--88ee9aafbf.mp4"

    clear = client.delete("/api/auth/session-token")
    assert clear.status_code == 204
    assert client.get("/api/auth").json()["has_session_token"] is False


def test_app_bookmark_job_route_returns_multiple_jobs(tmp_path: Path) -> None:
    service = ClipWebService(
        repo_root=tmp_path,
        shared_dir=tmp_path / "shared",
        plan_builder=_fake_plan_builder,
        route_length_resolver=lambda route, jwt_token: 180,
        bookmark_resolver=_fake_bookmark_resolver,
        runner=_fake_runner,
    )
    client = TestClient(create_app(service))

    response = client.post(
        "/api/jobs",
        json={
            "route_input": "e99064ba80a1dc40|00000006--88ee9aafbf",
            "render_type": "wide",
            "use_bookmarks": True,
            "bookmark_padding_seconds": 15,
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["job_count"] == 2
    assert len(payload["jobs"]) == 2
