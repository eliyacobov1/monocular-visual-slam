"""Multi-camera rig abstraction and calibration validation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Iterable, Mapping
import logging
import re

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraIntrinsics:
    matrix: np.ndarray

    @property
    def fx(self) -> float:
        return float(self.matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.matrix[1, 2])

    @property
    def skew(self) -> float:
        return float(self.matrix[0, 1])


@dataclass(frozen=True)
class CameraExtrinsics:
    rotation: np.ndarray
    translation: np.ndarray

    def as_matrix(self) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, :3] = self.rotation
        transform[:3, 3] = self.translation
        return transform


@dataclass(frozen=True)
class CameraModel:
    name: str
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics


@dataclass(frozen=True)
class CalibrationIssue:
    level: str
    message: str
    hint: str | None = None


@dataclass
class CalibrationReport:
    issues: list[CalibrationIssue] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(issue.level == "error" for issue in self.issues)

    def add_issue(self, level: str, message: str, hint: str | None = None) -> None:
        self.issues.append(CalibrationIssue(level=level, message=message, hint=hint))

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "issues": [
                {"level": issue.level, "message": issue.message, "hint": issue.hint}
                for issue in self.issues
            ],
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class CameraRig:
    cameras: dict[str, CameraModel]
    reference_camera: str

    @classmethod
    def from_kitti_calibration(
        cls,
        calibration: Mapping[str, np.ndarray],
        camera_names: Iterable[str] | None = None,
        reference_camera: str | None = None,
    ) -> "CameraRig":
        key_map = _extract_kitti_projection_keys(calibration)
        if camera_names is None:
            camera_names = sorted(key_map.keys())
        camera_names = list(camera_names)
        if not camera_names:
            raise ValueError("No camera projection matrices found in calibration data.")

        cameras: dict[str, CameraModel] = {}
        for name in camera_names:
            key = key_map.get(name)
            if key is None:
                raise KeyError(f"Projection matrix for {name} not found in calibration.")
            P = calibration[key]
            if P.shape != (3, 4):
                raise ValueError(f"Projection matrix {key} must be 3x4, got {P.shape}")
            K = _intrinsics_from_projection(P)
            extrinsics = _extrinsics_from_projection(P, K)
            cameras[name] = CameraModel(
                name=name,
                intrinsics=CameraIntrinsics(matrix=K),
                extrinsics=extrinsics,
            )

        reference = reference_camera or camera_names[0]
        if reference not in cameras:
            raise KeyError(f"Reference camera {reference} missing from calibration.")
        return cls(cameras=cameras, reference_camera=reference)

    def baseline_to(self, camera_name: str) -> float:
        if camera_name not in self.cameras:
            raise KeyError(f"Camera {camera_name} not in rig.")
        reference = self.cameras[self.reference_camera]
        target = self.cameras[camera_name]
        return float(np.linalg.norm(target.extrinsics.translation - reference.extrinsics.translation))

    def validate(self) -> CalibrationReport:
        start = perf_counter()
        report = CalibrationReport()
        report.metrics["num_cameras"] = float(len(self.cameras))

        for name, camera in self.cameras.items():
            _validate_intrinsics(camera.intrinsics, report, name)
            _validate_extrinsics(camera.extrinsics, report, name)

        reference = self.cameras[self.reference_camera]
        for name, camera in self.cameras.items():
            if name == self.reference_camera:
                continue
            baseline = float(
                np.linalg.norm(camera.extrinsics.translation - reference.extrinsics.translation)
            )
            report.metrics[f"baseline_m_{name}"] = baseline
            if baseline <= 0:
                report.add_issue(
                    "error",
                    f"Baseline between {self.reference_camera} and {name} is non-positive.",
                    hint="Check projection matrices for stereo baselines.",
                )
            elif baseline < 1e-3:
                report.add_issue(
                    "warning",
                    f"Baseline between {self.reference_camera} and {name} is very small.",
                    hint="Stereo depth estimates may be unstable with tiny baselines.",
                )

        report.metrics["validation_time_ms"] = (perf_counter() - start) * 1000.0
        return report


def _extract_kitti_projection_keys(calibration: Mapping[str, np.ndarray]) -> dict[str, str]:
    key_map: dict[str, str] = {}
    for key in calibration:
        match = re.match(r"P_rect_0?(\d+)$", key)
        if match is None:
            match = re.match(r"P_?(\d+)$", key)
        if match is None:
            continue
        camera_id = int(match.group(1))
        name = f"image_{camera_id}"
        key_map[name] = key
    return key_map


def _intrinsics_from_projection(P: np.ndarray) -> np.ndarray:
    K = P[:3, :3].copy()
    if K[2, 2] != 0:
        K /= K[2, 2]
    return K


def _extrinsics_from_projection(P: np.ndarray, K: np.ndarray) -> CameraExtrinsics:
    if np.linalg.det(K) == 0:
        raise ValueError("Intrinsic matrix is singular, cannot extract extrinsics.")
    translation = -np.linalg.inv(K) @ P[:, 3]
    rotation = np.eye(3)
    return CameraExtrinsics(rotation=rotation, translation=translation)


def _validate_intrinsics(
    intrinsics: CameraIntrinsics,
    report: CalibrationReport,
    camera_name: str,
) -> None:
    K = intrinsics.matrix
    if K.shape != (3, 3):
        report.add_issue(
            "error",
            f"{camera_name} intrinsics must be 3x3, got {K.shape}.",
            hint="Ensure calibration matrices are parsed correctly.",
        )
        return

    fx, fy = intrinsics.fx, intrinsics.fy
    report.metrics[f"{camera_name}_fx"] = fx
    report.metrics[f"{camera_name}_fy"] = fy
    report.metrics[f"{camera_name}_cx"] = intrinsics.cx
    report.metrics[f"{camera_name}_cy"] = intrinsics.cy
    report.metrics[f"{camera_name}_skew"] = intrinsics.skew

    if fx <= 0 or fy <= 0:
        report.add_issue(
            "error",
            f"{camera_name} intrinsics have non-positive focal length.",
            hint="Check calibration file or dataset preprocessing.",
        )
    if abs(intrinsics.skew) > 1e-3:
        report.add_issue(
            "warning",
            f"{camera_name} intrinsics have notable skew ({intrinsics.skew:.4f}).",
            hint="Verify rectification or calibration process.",
        )
    if not np.isclose(K[2, 2], 1.0, atol=1e-3):
        report.add_issue(
            "warning",
            f"{camera_name} intrinsics K[2,2] is {K[2,2]:.4f}, expected 1.0.",
            hint="Ensure normalization is applied to projection matrices.",
        )
    if intrinsics.cx < 0 or intrinsics.cy < 0:
        report.add_issue(
            "warning",
            f"{camera_name} intrinsics principal point is negative.",
            hint="Check image coordinate conventions in calibration.",
        )
    condition_number = float(np.linalg.cond(K))
    report.metrics[f"{camera_name}_condition_number"] = condition_number
    if condition_number > 1e6:
        report.add_issue(
            "warning",
            f"{camera_name} intrinsics are ill-conditioned (cond={condition_number:.2e}).",
            hint="Calibration may be unstable; verify inputs.",
        )


def _validate_extrinsics(
    extrinsics: CameraExtrinsics,
    report: CalibrationReport,
    camera_name: str,
) -> None:
    R = extrinsics.rotation
    t = extrinsics.translation
    if R.shape != (3, 3) or t.shape != (3,):
        report.add_issue(
            "error",
            f"{camera_name} extrinsics must be R(3x3) and t(3,), got {R.shape} and {t.shape}.",
            hint="Ensure calibration extraction produces SE(3) transforms.",
        )
        return
    ortho_error = float(np.linalg.norm(R.T @ R - np.eye(3)))
    det = float(np.linalg.det(R))
    report.metrics[f"{camera_name}_rotation_orthonormal_error"] = ortho_error
    report.metrics[f"{camera_name}_rotation_det"] = det
    report.metrics[f"{camera_name}_translation_norm"] = float(np.linalg.norm(t))
    if ortho_error > 1e-2 or not np.isclose(det, 1.0, atol=1e-2):
        report.add_issue(
            "error",
            f"{camera_name} rotation matrix is not a valid SO(3) rotation.",
            hint="Verify calibration matrices or rectify before parsing.",
        )
    if np.linalg.norm(t) > 10.0:
        report.add_issue(
            "warning",
            f"{camera_name} translation norm is large ({np.linalg.norm(t):.2f}m).",
            hint="Check units (meters expected for KITTI).",
        )

