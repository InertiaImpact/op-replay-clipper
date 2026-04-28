from __future__ import annotations

from datetime import datetime
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Literal, cast, get_args
from urllib.parse import quote, urlparse
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator
import requests

from core.clip_orchestrator import ClipRequest, LocalAccel, OutputFormatInput, RenderType, build_clip_plan
from core.driver_face_swap import (
    DriverFaceAnonymizationMode,
    DriverFaceAnonymizationProfile,
    DriverFaceSelectionMode,
    DriverFaceSwapPreset,
    PassengerRedactionStyle,
    canonical_driver_face_profile,
    default_driver_face_donor_bank_dir,
    default_driver_face_source_image,
    default_facefusion_model,
    default_facefusion_root,
)
from core.forward_upon_wide import DEFAULT_FORWARD_UPON_WIDE_H, ForwardUponWideHInput, parse_forward_upon_wide_h
from core.openpilot_config import default_local_openpilot_root
from core.ui_layouts import UI_ALT_VARIANTS, UIAltVariant

JOB_STATES = ("queued", "preparing", "running", "succeeded", "failed")
JobState = Literal["queued", "preparing", "running", "succeeded", "failed"]
PASSENGER_REDACTION_STYLE_CHOICES: tuple[PassengerRedactionStyle, ...] = (
    "blur",
    "silhouette",
    "black_silhouette",
    "ir_tint",
)
DEFAULT_SESSION_TOKEN_PATH = Path(".cache/webui/comma.jwt")
COMMA_JWT_MIN_LENGTH = 181
DEFAULT_START_SECONDS = 50
DEFAULT_LENGTH_SECONDS = 20
LOCAL_MAXIMUM_LENGTH_SECONDS = 12 * 60 * 60


def route_filename_stem(route: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1F]+', "_", route).strip(" .")
    return sanitized or "clip"


def _timestamp(value: float | None) -> str | None:
    if value is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(value))


def has_valid_session_token(token: str | None) -> bool:
    return isinstance(token, str) and len(token.strip()) >= COMMA_JWT_MIN_LENGTH


def _normalize_route_input(submission: "ClipJobSubmission") -> "ClipJobSubmission":
    route_input = submission.route_input.strip()
    if not route_input:
        return submission
    parsed = urlparse(route_input)
    if parsed.scheme == "https" and parsed.hostname == "connect.comma.ai":
        return submission.model_copy(update={"route_input": route_input})
    if route_input.startswith("connect.comma.ai/"):
        return submission.model_copy(update={"route_input": f"https://{route_input}"})

    parts = [part for part in route_input.strip("/").split("/") if part]
    if len(parts) == 2:
        dongle_id, route_slug = parts
        return submission.model_copy(update={"route_input": f"{dongle_id}|{route_slug}"})
    if len(parts) == 3 and parts[2].isdigit():
        dongle_id, route_slug, start_seconds = parts
        return submission.model_copy(
            update={
                "route_input": f"{dongle_id}|{route_slug}",
                "start_seconds": int(start_seconds),
            }
        )
    if len(parts) == 4 and parts[2].isdigit() and parts[3].isdigit():
        return submission.model_copy(update={"route_input": f"https://connect.comma.ai/{'/'.join(parts)}"})
    return submission.model_copy(update={"route_input": route_input})


class ClipJobSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route_input: str = Field(min_length=1)
    render_type: RenderType = "ui-alt"
    start_seconds: int = DEFAULT_START_SECONDS
    length_seconds: int = DEFAULT_LENGTH_SECONDS
    smear_seconds: int = 3
    target_mb: int = 9
    file_format: OutputFormatInput = "auto"
    qcam: bool = False
    ui_alt_variant: UIAltVariant | None = None
    windowed: bool = False
    skip_openpilot_update: bool = False
    skip_openpilot_bootstrap: bool = False
    openpilot_dir: str = Field(default_factory=default_local_openpilot_root)
    data_root: str = "./shared/data_dir"
    data_dir: str | None = None
    skip_download: bool = False
    accel: LocalAccel = "auto"
    forward_upon_wide_h: ForwardUponWideHInput = DEFAULT_FORWARD_UPON_WIDE_H
    driver_face_anonymization: DriverFaceAnonymizationMode = "none"
    driver_face_profile: DriverFaceAnonymizationProfile = "driver_face_swap_passenger_face_swap"
    passenger_redaction_style: PassengerRedactionStyle = "blur"
    driver_face_source_image: str = Field(default_factory=default_driver_face_source_image)
    driver_face_selection: DriverFaceSelectionMode = "manual"
    driver_face_donor_bank_dir: str = Field(default_factory=default_driver_face_donor_bank_dir)
    driver_face_preset: DriverFaceSwapPreset = "fast"
    facefusion_root: str = Field(default_factory=default_facefusion_root)
    facefusion_model: str = Field(default_factory=default_facefusion_model)

    @field_validator("route_input", "openpilot_dir", "data_root", "driver_face_source_image", "driver_face_donor_bank_dir", "facefusion_root", "facefusion_model", mode="before")
    @classmethod
    def _strip_required_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("data_dir", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator("forward_upon_wide_h", mode="before")
    @classmethod
    def _parse_forward_upon_wide_h(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return DEFAULT_FORWARD_UPON_WIDE_H
            return parse_forward_upon_wide_h(stripped)
        return value

    def to_clip_request(self, *, output_path: str, jwt_token: str | None) -> ClipRequest:
        return ClipRequest(
            render_type=self.render_type,
            route_or_url=self.route_input,
            start_seconds=self.start_seconds,
            length_seconds=self.length_seconds,
            target_mb=self.target_mb,
            ui_alt_variant=self.ui_alt_variant,
            file_format=self.file_format,
            output_path=output_path,
            smear_seconds=self.smear_seconds,
            jwt_token=jwt_token,
            forward_upon_wide_h=self.forward_upon_wide_h,
            explicit_data_dir=self.data_dir,
            data_root=self.data_root,
            execution_context="local",
            minimum_length_seconds=1,
            maximum_length_seconds=LOCAL_MAXIMUM_LENGTH_SECONDS,
            local_acceleration=self.accel,
            openpilot_dir=self.openpilot_dir,
            qcam=self.qcam,
            headless=not self.windowed,
            skip_download=self.skip_download,
            driver_face_anonymization=self.driver_face_anonymization,
            driver_face_profile=canonical_driver_face_profile(self.driver_face_profile),
            passenger_redaction_style=self.passenger_redaction_style,
            driver_face_source_image=self.driver_face_source_image,
            driver_face_preset=self.driver_face_preset,
            facefusion_root=self.facefusion_root,
            facefusion_model=self.facefusion_model,
            driver_face_selection=self.driver_face_selection,
            driver_face_donor_bank_dir=self.driver_face_donor_bank_dir,
        )


@dataclass
class ClipJob:
    id: str
    submission: ClipJobSubmission
    request: ClipRequest
    route: str
    output_path: Path
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    state: JobState = "queued"
    stage: str = "Waiting in queue"
    error: str | None = None
    logs: list[str] = field(default_factory=list)
    jwt_token: str | None = field(default=None, repr=False)


PlanBuilder = Callable[[ClipRequest], SimpleNamespace]
JobRunner = Callable[[ClipJob, Path, Callable[[str], None]], None]
RouteLengthResolver = Callable[[str, str | None], int]


def build_clip_command(job: ClipJob) -> list[str]:
    request = job.request
    submission = job.submission
    command = [
        sys.executable,
        "clip.py",
        request.render_type,
        request.route_or_url,
        "--start-seconds",
        str(request.start_seconds),
        "--length-seconds",
        str(request.length_seconds),
        "--smear-seconds",
        str(request.smear_seconds),
        "--output",
        request.output_path,
        "--openpilot-dir",
        request.openpilot_dir,
        "--file-size-mb",
        str(request.target_mb),
        "--file-format",
        request.file_format,
        "--forward-upon-wide-h",
        str(request.forward_upon_wide_h),
        "--data-root",
        request.data_root,
        "--accel",
        request.local_acceleration,
        "--driver-face-anonymization",
        request.driver_face_anonymization,
        "--driver-face-profile",
        request.driver_face_profile,
        "--passenger-redaction-style",
        request.passenger_redaction_style,
        "--driver-face-source-image",
        request.driver_face_source_image,
        "--driver-face-selection",
        request.driver_face_selection,
        "--driver-face-donor-bank-dir",
        request.driver_face_donor_bank_dir,
        "--driver-face-preset",
        request.driver_face_preset,
        "--facefusion-root",
        request.facefusion_root,
        "--facefusion-model",
        request.facefusion_model,
    ]
    if request.ui_alt_variant is not None:
        command.extend(["--ui-alt-variant", request.ui_alt_variant])
    if request.explicit_data_dir:
        command.extend(["--data-dir", request.explicit_data_dir])
    if request.qcam:
        command.append("--qcam")
    if not request.headless:
        command.append("--windowed")
    if request.skip_download:
        command.append("--skip-download")
    if submission.skip_openpilot_update:
        command.append("--skip-openpilot-update")
    if submission.skip_openpilot_bootstrap:
        command.append("--skip-openpilot-bootstrap")
    return command


def default_job_runner(job: ClipJob, repo_root: Path, on_log: Callable[[str], None]) -> None:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    if job.jwt_token:
        env["COMMA_JWT"] = job.jwt_token
    command = build_clip_command(job)
    on_log(f"Starting {job.request.render_type} for {job.route} -> {job.output_path.name}")
    process = subprocess.Popen(
        command,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.rstrip()
        if line:
            on_log(line)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"clip.py exited with code {return_code}.")


class ClipWebService:
    def __init__(
        self,
        *,
        repo_root: Path,
        shared_dir: Path,
        session_token_path: Path | None = None,
        route_length_resolver: RouteLengthResolver | None = None,
        plan_builder: PlanBuilder | None = None,
        runner: JobRunner | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.shared_dir = shared_dir.resolve()
        self.shared_dir.mkdir(parents=True, exist_ok=True)
        self._session_token_path = (
            (self.repo_root / session_token_path).resolve()
            if session_token_path is not None
            else (self.repo_root / DEFAULT_SESSION_TOKEN_PATH).resolve()
        )
        self._route_length_resolver = route_length_resolver or resolve_full_route_length_seconds
        self._plan_builder = plan_builder or cast(PlanBuilder, build_clip_plan)
        self._runner = runner or default_job_runner
        self._jobs: dict[str, ClipJob] = {}
        self._pending: deque[str] = deque()
        self._session_token = self._load_session_token()
        self._active_job_id: str | None = None
        self._condition = threading.Condition()
        self._worker = threading.Thread(target=self._worker_loop, name="clip-web-worker", daemon=True)
        self._worker.start()

    def options_manifest(self) -> dict[str, object]:
        return {
            "render_types": list(get_args(RenderType)),
            "file_formats": list(get_args(OutputFormatInput)),
            "accelerations": list(get_args(LocalAccel)),
            "ui_alt_variants": list(UI_ALT_VARIANTS),
            "driver_face_anonymization": list(get_args(DriverFaceAnonymizationMode)),
            "driver_face_profiles": list(get_args(DriverFaceAnonymizationProfile)),
            "driver_face_selection_modes": list(get_args(DriverFaceSelectionMode)),
            "driver_face_presets": list(get_args(DriverFaceSwapPreset)),
            "passenger_redaction_styles": list(PASSENGER_REDACTION_STYLE_CHOICES),
            "defaults": {
                "render_type": "ui-alt",
                "start_seconds": DEFAULT_START_SECONDS,
                "length_seconds": DEFAULT_LENGTH_SECONDS,
                "smear_seconds": 3,
                "target_mb": 9,
                "file_format": "auto",
                "qcam": False,
                "ui_alt_variant": "",
                "windowed": False,
                "skip_openpilot_update": False,
                "skip_openpilot_bootstrap": False,
                "openpilot_dir": default_local_openpilot_root(),
                "data_root": "./shared/data_dir",
                "data_dir": "",
                "skip_download": False,
                "accel": "auto",
                "forward_upon_wide_h": str(DEFAULT_FORWARD_UPON_WIDE_H),
                "driver_face_anonymization": "none",
                "driver_face_profile": "driver_face_swap_passenger_face_swap",
                "passenger_redaction_style": "blur",
                "driver_face_source_image": default_driver_face_source_image(),
                "driver_face_selection": "manual",
                "driver_face_donor_bank_dir": default_driver_face_donor_bank_dir(),
                "driver_face_preset": "fast",
                "facefusion_root": default_facefusion_root(),
                "facefusion_model": default_facefusion_model(),
            },
            "jwt_url": "https://jwt.comma.ai",
        }

    def auth_status(self) -> dict[str, object]:
        token_path = self._session_token_path.relative_to(self.repo_root).as_posix()
        with self._condition:
            return {
                "has_session_token": self._session_token is not None,
                "has_valid_session_token": has_valid_session_token(self._session_token),
                "jwt_url": "https://jwt.comma.ai",
                "token_storage": "repo_local_ignored_cache",
                "token_path": token_path,
            }

    def set_session_token(self, token: str) -> None:
        stripped = token.strip()
        if not stripped:
            raise ValueError("JWT token cannot be empty.")
        if not has_valid_session_token(stripped):
            raise ValueError("JWT token looks incomplete. Paste the full token from https://jwt.comma.ai .")
        self._write_session_token(stripped)
        with self._condition:
            self._session_token = stripped

    def clear_session_token(self) -> None:
        if self._session_token_path.exists():
            self._session_token_path.unlink()
        with self._condition:
            self._session_token = None

    def _load_session_token(self) -> str | None:
        if not self._session_token_path.exists():
            return None
        token = self._session_token_path.read_text(encoding="utf-8").strip()
        return token or None

    def _write_session_token(self, token: str) -> None:
        self._session_token_path.parent.mkdir(parents=True, exist_ok=True)
        self._session_token_path.write_text(f"{token}\n", encoding="utf-8")
        try:
            self._session_token_path.chmod(0o600)
        except OSError:
            pass

    def submit_job(self, submission: ClipJobSubmission) -> dict[str, object]:
        with self._condition:
            jwt_token = self._session_token
        normalized_submission = _normalize_route_input(submission)
        resolved_submission = self._resolve_submission_defaults(normalized_submission, jwt_token)
        provisional_request = resolved_submission.to_clip_request(
            output_path=str(self.shared_dir / "_pending.mp4"),
            jwt_token=jwt_token,
        )
        plan = self._plan_builder(provisional_request)
        route = str(plan.route)
        with self._condition:
            output_path = self._reserve_output_path_locked(route)
            request = resolved_submission.to_clip_request(output_path=str(output_path), jwt_token=jwt_token)
            job = ClipJob(
                id=uuid4().hex,
                submission=resolved_submission,
                request=request,
                route=route,
                output_path=output_path,
                jwt_token=jwt_token,
            )
            job.logs.append(f"Queued {resolved_submission.render_type} render for {route}.")
            if resolved_submission.start_seconds == 0 and resolved_submission.route_input == route:
                job.logs.append(f"Using the full route length of {resolved_submission.length_seconds} seconds by default.")
            self._jobs[job.id] = job
            self._pending.append(job.id)
            self._condition.notify()
            return self._snapshot_locked(job.id)

    def _resolve_submission_defaults(
        self,
        submission: ClipJobSubmission,
        jwt_token: str | None,
    ) -> ClipJobSubmission:
        if not _should_expand_raw_route_defaults(submission):
            return submission
        full_length_seconds = self._route_length_resolver(submission.route_input, jwt_token)
        return submission.model_copy(update={"start_seconds": 0, "length_seconds": full_length_seconds})

    def list_jobs(self) -> list[dict[str, object]]:
        with self._condition:
            job_ids = sorted(self._jobs, key=lambda job_id: self._jobs[job_id].created_at, reverse=True)
            return [self._snapshot_locked(job_id) for job_id in job_ids]

    def get_job(self, job_id: str) -> dict[str, object] | None:
        with self._condition:
            if job_id not in self._jobs:
                return None
            return self._snapshot_locked(job_id)

    def list_outputs(self) -> list[dict[str, object]]:
        outputs: list[dict[str, object]] = []
        for path in sorted(self.shared_dir.glob("*.mp4"), key=lambda candidate: candidate.stat().st_mtime, reverse=True):
            stat = path.stat()
            outputs.append(
                {
                    "name": path.name,
                    "size_bytes": stat.st_size,
                    "modified_at": _timestamp(stat.st_mtime),
                    "url": f"/outputs/{quote(path.name)}",
                }
            )
        return outputs

    def _reserve_output_path_locked(self, route: str) -> Path:
        stem = route_filename_stem(route)
        reserved = {job.output_path.name for job in self._jobs.values()}
        candidate = self.shared_dir / f"{stem}.mp4"
        suffix = 2
        while candidate.exists() or candidate.name in reserved:
            candidate = self.shared_dir / f"{stem}-{suffix}.mp4"
            suffix += 1
        return candidate

    def _append_log(self, job_id: str, message: str) -> None:
        with self._condition:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.logs.append(message)
            if len(job.logs) > 200:
                del job.logs[:-200]
            job.updated_at = time.time()

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._pending:
                    self._condition.wait()
                job_id = self._pending.popleft()
                job = self._jobs[job_id]
                job.state = "preparing"
                job.stage = "Starting render"
                job.started_at = time.time()
                job.updated_at = job.started_at
                self._active_job_id = job_id
            try:
                self._append_log(job_id, "Starting queued render job.")
                with self._condition:
                    job = self._jobs[job_id]
                    job.state = "running"
                    job.stage = "Rendering clip"
                    job.updated_at = time.time()
                self._runner(job, self.repo_root, lambda line: self._append_log(job_id, line))
                if not job.output_path.exists():
                    raise RuntimeError(f"Render finished without creating {job.output_path.name}.")
            except Exception as error:
                with self._condition:
                    job = self._jobs[job_id]
                    job.state = "failed"
                    job.stage = "Render failed"
                    job.error = str(error)
                    job.finished_at = time.time()
                    job.updated_at = job.finished_at
                self._append_log(job_id, f"ERROR: {error}")
            else:
                with self._condition:
                    job = self._jobs[job_id]
                    job.state = "succeeded"
                    job.stage = "Render finished"
                    job.finished_at = time.time()
                    job.updated_at = job.finished_at
                self._append_log(job_id, f"Finished {job.output_path.name}.")
            finally:
                with self._condition:
                    self._active_job_id = None

    def _snapshot_locked(self, job_id: str) -> dict[str, object]:
        job = self._jobs[job_id]
        queue_positions = {pending_id: index + 1 for index, pending_id in enumerate(self._pending)}
        return {
            "id": job.id,
            "route": job.route,
            "route_input": job.submission.route_input,
            "render_type": job.submission.render_type,
            "state": job.state,
            "stage": job.stage,
            "error": job.error,
            "created_at": _timestamp(job.created_at),
            "updated_at": _timestamp(job.updated_at),
            "started_at": _timestamp(job.started_at),
            "finished_at": _timestamp(job.finished_at),
            "queue_position": queue_positions.get(job.id),
            "output_name": job.output_path.name,
            "output_url": f"/outputs/{quote(job.output_path.name)}",
            "logs": job.logs[-20:],
        }


def create_default_service() -> ClipWebService:
    repo_root = Path(__file__).resolve().parents[1]
    shared_dir = repo_root / "shared"
    return ClipWebService(repo_root=repo_root, shared_dir=shared_dir)


def _parse_route_duration_seconds(metadata: object) -> int | None:
    if not isinstance(metadata, dict):
        return None
    start_time = metadata.get("start_time")
    end_time = metadata.get("end_time")
    if isinstance(start_time, str) and isinstance(end_time, str):
        return max(1, int((datetime.fromisoformat(end_time) - datetime.fromisoformat(start_time)).total_seconds()))
    maxqlog = metadata.get("maxqlog")
    procqlog = metadata.get("procqlog")
    numeric_segment_counts = [value for value in (maxqlog, procqlog) if isinstance(value, int) and value >= 0]
    if numeric_segment_counts:
        return (max(numeric_segment_counts) + 1) * 60
    segment_numbers = metadata.get("segment_numbers")
    if isinstance(segment_numbers, list):
        numeric_segments = [value for value in segment_numbers if isinstance(value, int) and value >= 0]
        if numeric_segments:
            return (max(numeric_segments) + 1) * 60
    return None


def resolve_full_route_length_seconds(route: str, jwt_token: str | None) -> int:
    route_url = route.replace("|", "%7C")
    headers = {"Authorization": f"JWT {jwt_token}"} if jwt_token else None
    endpoints = (
        f"https://api.commadotai.com/v1/route/{route_url}/files",
        f"https://api.commadotai.com/v1/route/{route_url}",
    )
    for endpoint in endpoints:
        response = requests.get(endpoint, headers=headers, timeout=30)
        if response.status_code != 200:
            continue
        duration_seconds = _parse_route_duration_seconds(response.json())
        if duration_seconds is not None:
            return duration_seconds
    raise ValueError(f"Could not determine the full route length for {route}.")


def _should_expand_raw_route_defaults(submission: ClipJobSubmission) -> bool:
    if "connect.comma.ai" in submission.route_input:
        return False
    if "|" not in submission.route_input:
        return False
    return (
        submission.start_seconds == DEFAULT_START_SECONDS
        and submission.length_seconds == DEFAULT_LENGTH_SECONDS
    )
