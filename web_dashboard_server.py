"""WebSocket + HTTP server for the SLAM live dashboard."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import websockets

from cam_intrinsics_estimation import load_K_from_file, make_K
from demo_utils import DEFAULT_VIDEO_PATH, ensure_sample_video

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebDashboardConfig:
    """Configuration for the dashboard servers."""

    ws_host: str = "0.0.0.0"
    ws_port: int = 8000
    http_host: str = "0.0.0.0"
    http_port: int = 8001
    video_path: Path = DEFAULT_VIDEO_PATH
    intrinsics_file: Path | None = None
    max_frames: int | None = None
    target_fps: float = 24.0
    seed: int = 11


@dataclass(frozen=True)
class FrameStatus:
    """Frame-level status payload for dashboard streaming."""

    frame_id: int
    timestamp: float
    total_frames: int
    features: int
    matches: int
    inliers: int
    inlier_ratio: float
    position: tuple[float, float, float]
    yaw_pitch_roll: tuple[float, float, float]
    status: str
    status_level: str
    progress: float
    fps: float
    log: str

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "total_frames": self.total_frames,
            "features": self.features,
            "matches": self.matches,
            "inliers": self.inliers,
            "inlier_ratio": self.inlier_ratio,
            "position": self.position,
            "yaw_pitch_roll": self.yaw_pitch_roll,
            "status": self.status,
            "status_level": self.status_level,
            "progress": self.progress,
            "fps": self.fps,
            "log": self.log,
        }


class DashboardStream:
    """Generate SLAM telemetry updates for the web dashboard."""

    def __init__(self, config: WebDashboardConfig) -> None:
        self.config = config
        np.random.seed(config.seed)
        cv2.setRNGSeed(config.seed)
        video_path = config.video_path
        if not video_path.exists():
            if video_path == DEFAULT_VIDEO_PATH:
                video_path = ensure_sample_video(video_path)
            else:
                raise FileNotFoundError(f"Video not found: {video_path}")
        self.video_path = video_path

    def _load_intrinsics(self, width: int, height: int) -> np.ndarray:
        if self.config.intrinsics_file is not None:
            return load_K_from_file(self.config.intrinsics_file)
        return make_K(width, height)

    def _status_for(self, matches: int, inlier_ratio: float) -> tuple[str, str]:
        if matches < 40:
            return "Low match density", "warning"
        if inlier_ratio < 0.2:
            return "Tracking lost", "error"
        if inlier_ratio < 0.35:
            return "Unstable pose", "warning"
        return "Tracking stable", "ok"

    def stream(self) -> Iterator[FrameStatus]:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {self.video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, first_frame = cap.read()
        if not ok:
            raise RuntimeError("Video is empty")
        height, width = first_frame.shape[:2]
        K = self._load_intrinsics(width, height)
        orb = cv2.ORB_create(1000)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        prev_kp, prev_desc = orb.detectAndCompute(prev_gray, None)
        pose = np.eye(4)
        frame_id = 0
        log_index = 0
        start_time = time.perf_counter()

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp, desc = orb.detectAndCompute(gray, None)
            matches = []
            if prev_desc is not None and desc is not None and len(prev_desc) > 0 and len(desc) > 0:
                matches = matcher.match(prev_desc, desc)
                matches = sorted(matches, key=lambda m: m.distance)
            pts_prev = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts_curr = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            R = np.eye(3)
            t = np.zeros((3, 1))
            inlier_mask = None
            inlier_ratio = 0.0
            if len(matches) >= 8:
                try:
                    E, inlier_mask = cv2.findEssentialMat(
                        pts_prev, pts_curr, K, method=cv2.RANSAC, threshold=1.0
                    )
                    if E is not None:
                        _, R, t, inlier_mask = cv2.recoverPose(
                            E, pts_prev, pts_curr, K, mask=inlier_mask
                        )
                        inlier_mask = inlier_mask.ravel().astype(bool)
                        inlier_ratio = float(inlier_mask.sum()) / len(matches)
                    else:
                        inlier_mask = np.zeros(len(matches), dtype=bool)
                except cv2.error as exc:
                    LOGGER.warning(
                        "Essential matrix estimation failed; keeping identity pose.",
                        exc_info=exc,
                    )
                    inlier_mask = np.zeros(len(matches), dtype=bool)
            else:
                inlier_mask = np.zeros(len(matches), dtype=bool)

            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.ravel()
            pose = pose @ T
            pos = pose[:3, 3]
            yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
            pitch = np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)))
            roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))

            status, status_level = self._status_for(len(matches), inlier_ratio)
            progress = frame_id / total_frames if total_frames else 0.0
            elapsed = time.perf_counter() - start_time
            fps = frame_id / elapsed if elapsed > 0.0 else 0.0
            log_index += 1
            log = f"Frame {frame_id}: {status} ({len(matches)} matches)"

            yield FrameStatus(
                frame_id=frame_id,
                timestamp=time.time(),
                total_frames=total_frames,
                features=len(kp),
                matches=len(matches),
                inliers=int(inlier_mask.sum()),
                inlier_ratio=inlier_ratio,
                position=(float(pos[0]), float(pos[1]), float(pos[2])),
                yaw_pitch_roll=(float(yaw), float(pitch), float(roll)),
                status=status,
                status_level=status_level,
                progress=progress,
                fps=fps,
                log=log,
            )

            prev_gray, prev_kp, prev_desc = gray, kp, desc

            if self.config.max_frames is not None and frame_id >= self.config.max_frames:
                break
            if self.config.target_fps > 0:
                time.sleep(max(0.0, (1.0 / self.config.target_fps)))

        cap.release()


class DashboardServer:
    """Run the HTTP server and WebSocket broadcaster."""

    def __init__(self, config: WebDashboardConfig) -> None:
        self.config = config
        self._stop_event = threading.Event()

    def _start_http_server(self) -> threading.Thread:
        from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

        directory = Path(__file__).resolve().parent / "web_dashboard"
        handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(  # noqa: E731
            *args, directory=str(directory), **kwargs
        )
        httpd = ThreadingHTTPServer((self.config.http_host, self.config.http_port), handler)

        def serve() -> None:
            LOGGER.info(
                "HTTP dashboard server started",
                extra={"host": self.config.http_host, "port": self.config.http_port},
            )
            with httpd:
                httpd.serve_forever(poll_interval=0.5)

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        return thread

    async def _stream(self, websocket: websockets.WebSocketServerProtocol) -> None:
        streamer = DashboardStream(self.config)
        logs: list[str] = []
        await websocket.send(
            json.dumps(
                {
                    "type": "hello",
                    "payload": {
                        "video": str(streamer.video_path),
                        "total_frames": None,
                    },
                }
            )
        )
        for status in streamer.stream():
            logs.append(status.log)
            logs = logs[-6:]
            payload = status.to_dict()
            payload["logs"] = logs
            await websocket.send(json.dumps({"type": "frame", "payload": payload}))

    async def _ws_handler(self, websocket: websockets.WebSocketServerProtocol) -> None:
        try:
            await self._stream(websocket)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("WebSocket session failed", exc_info=exc)

    async def run(self) -> None:
        self._start_http_server()
        async with websockets.serve(self._ws_handler, self.config.ws_host, self.config.ws_port):
            LOGGER.info(
                "WebSocket server started",
                extra={"host": self.config.ws_host, "port": self.config.ws_port},
            )
            while not self._stop_event.is_set():
                await asyncio.sleep(0.5)

    def stop(self) -> None:
        self._stop_event.set()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SLAM web dashboard server")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--intrinsics_file", type=Path)
    parser.add_argument("--ws-port", type=int, default=8000)
    parser.add_argument("--http-port", type=int, default=8001)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--target-fps", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=11)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    config = WebDashboardConfig(
        ws_port=args.ws_port,
        http_port=args.http_port,
        video_path=args.video,
        intrinsics_file=args.intrinsics_file,
        max_frames=args.max_frames,
        target_fps=args.target_fps,
        seed=args.seed,
    )
    server = DashboardServer(config)
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
