"""Deterministic ordering tests for loop-closure ranking."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from loop_closure import BoWDatabase


def test_loop_closure_rank_candidates_tiebreaks_by_frame_id() -> None:
    db = BoWDatabase(vocab_size=1, batch_size=1)
    db.vocab_trained = True
    db.vocab = np.zeros((1, 32), dtype=np.float32)
    db.hists = [np.array([1.0], dtype=np.float32), np.array([1.0], dtype=np.float32)]
    db.frame_ids = [2, 1]
    desc = np.zeros((4, 32), dtype=np.uint8)

    candidates = db.rank_candidates(desc)

    assert [frame_id for frame_id, _ in candidates] == [1, 2]
