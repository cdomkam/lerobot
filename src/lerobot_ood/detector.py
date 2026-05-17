"""Mahalanobis-distance OOD detector.

Fits a Gaussian density model on a stack of in-distribution embeddings.
Scores new embeddings by their Mahalanobis distance from the fitted
mean/covariance; flags as OOD when the score exceeds a threshold.

The detector takes already-encoded vectors as input: it does not own
the encoder. Keeping these decoupled lets the same detector work with
DINOv2, the policy's own backbone, or anything else that produces a
fixed-dimension vector per frame.

Embeddings should be L2-normalised before being passed in. The encoder
in ``lerobot_ood.encoder`` does this; if you wire up a different
encoder, normalise its output before calling ``fit`` / ``score``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class OODResult:
    score: float
    is_ood: bool
    threshold: float
    embedding: np.ndarray = field(repr=False)


class OODDetector:
    """Mahalanobis OOD detector with optional PCA pre-projection."""

    def __init__(
        self,
        threshold_percentile: float = 95.0,
        pca_components: int | None = 32,
    ):
        self.threshold_percentile = threshold_percentile
        self.pca_components = pca_components
        self._mean: np.ndarray | None = None
        self._cov_inv: np.ndarray | None = None
        self._threshold: float | None = None
        self._pca = None
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def threshold(self) -> float:
        self._require_fitted()
        return float(self._threshold)

    def fit(self, embeddings: np.ndarray) -> "OODDetector":
        """Fit on a stack of in-distribution embeddings.

        ``embeddings``: ``(N, D)`` array of pre-L2-normalised vectors.
        """
        z = np.asarray(embeddings, dtype=np.float32)
        if z.ndim != 2:
            raise ValueError(f"expected (N, D) embeddings, got shape {z.shape}")
        if z.shape[0] < 2:
            raise ValueError(f"need >= 2 samples to fit, got {z.shape[0]}")

        if self.pca_components is not None:
            from sklearn.decomposition import PCA

            n = min(self.pca_components, z.shape[1], z.shape[0])
            self._pca = PCA(n_components=n, whiten=True)
            z = self._pca.fit_transform(z)

        self._mean = z.mean(axis=0)
        cov = np.cov(z, rowvar=False)
        cov = cov + np.eye(cov.shape[0], dtype=np.float64) * 1e-5
        self._cov_inv = np.linalg.inv(cov).astype(np.float32)

        scores = self._score_batch(z)
        self._threshold = float(np.percentile(scores, self.threshold_percentile))
        self._fitted = True
        return self

    def score(self, embedding: np.ndarray) -> OODResult:
        """Score a single embedding."""
        self._require_fitted()
        z = np.asarray(embedding, dtype=np.float32).ravel()
        if self._pca is not None:
            z = self._pca.transform(z[None])[0]
        s = self._mahalanobis(z)
        return OODResult(
            score=float(s),
            is_ood=bool(s >= self._threshold),
            threshold=float(self._threshold),
            embedding=z,
        )

    def save(self, path: str | Path) -> None:
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, np.ndarray] = {
            "version": np.array(1),
            "threshold_percentile": np.array(self.threshold_percentile),
            "threshold": np.array(self._threshold),
            "mean": self._mean,
            "cov_inv": self._cov_inv,
        }
        if self._pca is not None:
            data["pca_components"] = np.array(self.pca_components)
            data["pca_mean"] = self._pca.mean_
            data["pca_components_matrix"] = self._pca.components_
            data["pca_explained_variance"] = self._pca.explained_variance_
            data["pca_whiten"] = np.array(self._pca.whiten)
        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, path: str | Path) -> "OODDetector":
        d = np.load(path, allow_pickle=False)
        pca_components = int(d["pca_components"]) if "pca_components" in d.files else None
        det = cls(
            threshold_percentile=float(d["threshold_percentile"]),
            pca_components=pca_components,
        )
        det._mean = d["mean"]
        det._cov_inv = d["cov_inv"]
        det._threshold = float(d["threshold"])
        if pca_components is not None:
            from sklearn.decomposition import PCA

            whiten = bool(d["pca_whiten"]) if "pca_whiten" in d.files else False
            pca = PCA(n_components=pca_components, whiten=whiten)
            pca.mean_ = d["pca_mean"]
            pca.components_ = d["pca_components_matrix"]
            pca.explained_variance_ = d["pca_explained_variance"]
            pca.n_components_ = pca_components
            pca.n_features_in_ = len(d["pca_mean"])
            pca.whiten = whiten
            det._pca = pca
        det._fitted = True
        return det

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Detector not fitted. Call fit() or load() first.")

    def _mahalanobis(self, z: np.ndarray) -> float:
        diff = z - self._mean
        return float(diff @ self._cov_inv @ diff)

    def _score_batch(self, z: np.ndarray) -> np.ndarray:
        diff = z - self._mean
        return np.einsum("ni,ij,nj->n", diff, self._cov_inv, diff)
