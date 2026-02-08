"""Deterministic ordering tests for pose-graph inputs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from pose_graph import _order_edges


def test_pose_graph_edge_ordering_is_deterministic() -> None:
    edges = [
        {"i": 2, "j": 3, "R": [[1.0]], "t": [0.1], "weight": 1.0},
        {"i": 1, "j": 2, "R": [[1.0]], "t": [0.2], "weight": 0.5},
        {"i": 1, "j": 2, "R": [[1.0]], "t": [0.2], "weight": 0.5},
    ]

    ordered = _order_edges(edges)

    assert [edge["i"] for edge in ordered] == [1, 1, 2]
    assert [edge["j"] for edge in ordered] == [2, 2, 3]
