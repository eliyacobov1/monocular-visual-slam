#!/usr/bin/env python3
"""Unified entry point for SLAM visual interfaces."""

from __future__ import annotations

import argparse
import importlib
import logging
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from demo_utils import DEFAULT_VIDEO_PATH

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class InterfaceConfig:
    """Configuration for launching SLAM interfaces."""

    interface: str
    video: Path
    intrinsics_file: Path | None
    ws_port: int
    http_port: int


def _port_available(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _require_module(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        raise SystemExit(
            f"Missing dependency '{name}'. Run: python -m pip install {name}"
        )


def _validate_environment(config: InterfaceConfig) -> None:
    if config.interface == "gui":
        _require_module("matplotlib")
        _require_module("cv2")
    if config.interface == "web":
        _require_module("websockets")
        _require_module("cv2")
        if not _port_available(config.ws_port):
            raise SystemExit(f"WebSocket port {config.ws_port} is already in use")
        if not _port_available(config.http_port):
            raise SystemExit(f"HTTP port {config.http_port} is already in use")


def _launch_gui(config: InterfaceConfig) -> None:
    LOGGER.info("Launching local GUI viewer")
    args = [sys.executable, "slam_viewer.py", "--video", str(config.video)]
    if config.intrinsics_file:
        args.extend(["--intrinsics_file", str(config.intrinsics_file)])
    subprocess.run(args, check=True)


def _launch_web(config: InterfaceConfig) -> None:
    LOGGER.info("Launching web dashboard")
    args = [
        sys.executable,
        "web_dashboard_server.py",
        "--video",
        str(config.video),
        "--ws-port",
        str(config.ws_port),
        "--http-port",
        str(config.http_port),
    ]
    if config.intrinsics_file:
        args.extend(["--intrinsics_file", str(config.intrinsics_file)])
    subprocess.run(args, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the SLAM demo with a user-facing interface"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gui", action="store_true", help="Launch Matplotlib GUI")
    group.add_argument("--web", action="store_true", help="Launch web dashboard")
    parser.add_argument(
        "--interface",
        choices=("gui", "web"),
        default="gui",
        help="Interface to launch when no explicit flag is provided",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help="Path to input video (defaults to sample clip)",
    )
    parser.add_argument("--intrinsics_file", type=Path)
    parser.add_argument("--ws-port", type=int, default=8000)
    parser.add_argument("--http-port", type=int, default=8001)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    interface = "web" if args.web else "gui" if args.gui else args.interface
    config = InterfaceConfig(
        interface=interface,
        video=args.video,
        intrinsics_file=args.intrinsics_file,
        ws_port=args.ws_port,
        http_port=args.http_port,
    )
    _validate_environment(config)
    if config.interface == "gui":
        _launch_gui(config)
    else:
        _launch_web(config)


if __name__ == "__main__":
    main()
