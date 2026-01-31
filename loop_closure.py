import logging
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances_argmin_min

logger = logging.getLogger(__name__)

class BoWDatabase:
    """Simple BoW database for loop closure detection with ORB descriptors."""

    def __init__(self, vocab_size: int = 500, batch_size: int = 1000):
        self.vocab_size = vocab_size
        self.kmeans = MiniBatchKMeans(n_clusters=vocab_size, batch_size=batch_size, random_state=0)
        self.vocab_trained = False
        self.hists: list[np.ndarray] = []
        self.frame_ids: list[int] = []
        self.descriptors: list[np.ndarray] = []
        self.vocab: np.ndarray | None = None

    def add_frame(self, frame_id: int, desc: np.ndarray) -> None:
        if desc is None or len(desc) == 0:
            return
        self.descriptors.append(desc)
        if not self.vocab_trained and len(self.descriptors) * len(desc) >= self.vocab_size * 10:
            stacked = np.vstack(self.descriptors)
            self.kmeans.fit(stacked)
            self.vocab_trained = True
            self.vocab = self.kmeans.cluster_centers_.astype(np.float32)
            self.descriptors = []
            logger.info("BoW vocabulary trained on %d descriptors", len(stacked))
        if self.vocab_trained:
            hist = self._compute_hist(desc)
            self.hists.append(hist)
            self.frame_ids.append(frame_id)
            logger.debug("Added frame %d to BoW database", frame_id)

    def _compute_hist(self, desc: np.ndarray) -> np.ndarray:
        if self.vocab is None:
            words = self.kmeans.predict(desc)
        else:
            words, _ = pairwise_distances_argmin_min(
                desc.astype(np.float32, copy=False),
                self.vocab,
            )
        hist, _ = np.histogram(words, bins=np.arange(self.vocab_size + 1))
        hist = hist.astype(np.float32)
        if hist.sum() > 0:
            hist /= hist.sum()
        return hist

    def export_vocabulary(self) -> np.ndarray:
        if not self.vocab_trained or self.vocab is None:
            raise RuntimeError("BoW vocabulary has not been trained")
        return self.vocab.copy()

    def detect_loop(self, desc: np.ndarray, threshold: float = 0.75) -> int | None:
        if not self.vocab_trained or len(self.hists) == 0 or desc is None or len(desc) == 0:
            return None
        hist = self._compute_hist(desc)
        sims = cosine_similarity([hist], self.hists)[0]
        best_idx = int(np.argmax(sims))
        if sims[best_idx] > threshold:
            loop_id = self.frame_ids[best_idx]
            logger.info("Detected loop with frame %d (score=%.2f)", loop_id, sims[best_idx])
            return loop_id
        logger.debug("No loop detected: best score %.2f", sims[best_idx])
        return None
