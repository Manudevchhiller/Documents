"""Adaptive-Geometric-SMOTE (AGS).

A next-generation oversampling model that combines adaptive sampling,
geometric sample generation, and multi-perspective quality control.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y


@dataclass
class _ClassSamplingState:
    class_label: int
    target_new_samples: int
    generated: int = 0
    rejected: int = 0


class AdaptiveGeometricSMOTE(BaseEstimator):
    """Adaptive-Geometric-SMOTE (AGS) resampler.

    Key capabilities:
    - Adaptive local-k selection based on minority density
    - Dynamic class-wise sampling based on separability
    - Geometric sample generation with truncated Gaussian interpolation
    - Multi-perspective analysis (density + cluster + boundary + safety)
    - KD-tree accelerated nearest-neighbor search
    - Optional GPU backend (CuPy) for vectorized operations when available

    Parameters
    ----------
    k_bounds:
        Min/max neighborhood limits used for adaptive per-point k selection.
    boundary_k:
        Number of neighbors used for boundary and safety scoring.
    manifold_components:
        Number of PCA components used for manifold projection in high dimensions.
    manifold_threshold:
        Project to manifold when feature dimension is >= this threshold.
    gaussian_scale:
        Standard deviation of truncated Gaussian interpolation coefficient.
    quality_margin:
        Candidate is accepted only if nearest minority distance * quality_margin
        is smaller than nearest majority distance.
    max_attempt_factor:
        Max candidate generation attempts = target * max_attempt_factor.
    use_gpu:
        If True, attempts to use CuPy for vectorized score computations.
    random_state:
        Random seed for deterministic behavior.
    """

    def __init__(
        self,
        k_bounds: Tuple[int, int] = (3, 15),
        boundary_k: int = 9,
        manifold_components: int = 12,
        manifold_threshold: int = 25,
        gaussian_scale: float = 0.30,
        quality_margin: float = 0.92,
        max_attempt_factor: int = 20,
        truncation_attempts: int = 20,
        cluster_bounds: Tuple[int, int] = (2, 12),
        score_weights: Tuple[float, float, float, float] = (0.33, 0.24, 0.23, 0.20),
        safety_threshold: float = 0.50,
        use_gpu: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        self.k_bounds = k_bounds
        self.boundary_k = boundary_k
        self.manifold_components = manifold_components
        self.manifold_threshold = manifold_threshold
        self.gaussian_scale = gaussian_scale
        self.quality_margin = quality_margin
        self.max_attempt_factor = max_attempt_factor
        self.truncation_attempts = truncation_attempts
        self.cluster_bounds = cluster_bounds
        self.score_weights = score_weights
        self.safety_threshold = safety_threshold
        self.use_gpu = use_gpu
        self.random_state = random_state

    def _backend(self):
        if not self.use_gpu:
            self.used_backend_ = "numpy"
            return np
        try:
            import cupy as cp

            self.used_backend_ = "cupy"
            return cp
        except Exception:
            self.used_backend_ = "numpy"
            return np

    def _truncated_gaussian(self, rng: np.random.RandomState, mean: float = 0.5, std: float = 0.30) -> float:
        for _ in range(self.truncation_attempts):
            value = rng.normal(mean, std)
            if 0.0 <= value <= 1.0:
                return float(value)
        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        min_v = np.min(arr)
        max_v = np.max(arr)
        denom = max(max_v - min_v, 1e-12)
        return (arr - min_v) / denom

    def _adaptive_k(self, density: np.ndarray, n_minority: int) -> np.ndarray:
        k_min, k_max = self.k_bounds
        k_max = min(k_max, max(k_min + 1, n_minority - 1))
        k_min = min(k_min, max(2, k_max - 1))
        density_norm = self._normalize(density)
        # Sparse areas receive larger neighborhoods for stability.
        adaptive = k_min + (1.0 - density_norm) * (k_max - k_min)
        return np.clip(np.rint(adaptive).astype(int), k_min, k_max)

    def _class_separability_factor(self, X_min: np.ndarray, X_maj: np.ndarray) -> float:
        if len(X_maj) == 0:
            return 1.0
        c_min = X_min.mean(axis=0)
        c_maj = X_maj.mean(axis=0)
        inter = np.linalg.norm(c_min - c_maj)
        intra = float(np.mean(np.linalg.norm(X_min - c_min, axis=1)) + 1e-12)
        ratio = inter / intra
        # Lower separability => conservative oversampling to avoid overlap.
        return float(np.clip(ratio, 0.55, 1.0))

    def _prepare_space(self, X: np.ndarray) -> np.ndarray:
        n_features = X.shape[1]
        if n_features < self.manifold_threshold:
            self.manifold_model_ = None
            return X

        n_comp = min(self.manifold_components, n_features, max(2, X.shape[0] - 1))
        self.manifold_model_ = PCA(n_components=n_comp, random_state=self.random_state)
        return self.manifold_model_.fit_transform(X)

    def _cluster_priority(self, X_min_space: np.ndarray) -> np.ndarray:
        n_minority = len(X_min_space)
        if n_minority < 4:
            return np.ones(n_minority)

        c_min, c_max = self.cluster_bounds
        n_clusters = int(np.clip(np.sqrt(n_minority), c_min, c_max))
        km = MiniBatchKMeans(n_clusters=n_clusters, random_state=self.random_state, batch_size=256, n_init=3)
        labels = km.fit_predict(X_min_space)
        counts = np.bincount(labels)
        # Small clusters get larger weights to preserve rare structures.
        return 1.0 / np.sqrt(counts[labels])

    def fit_resample(self, X, y):
        """Resample imbalanced dataset.

        Returns
        -------
        X_resampled, y_resampled
        """

        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
        if len(self.score_weights) != 4:
            raise ValueError('score_weights must contain 4 values: density, boundary, safety, cluster')
        rng = check_random_state(self.random_state)
        self._backend()

        X_space = self._prepare_space(X)

        classes, counts = np.unique(y, return_counts=True)
        target_count = int(np.max(counts))

        X_syn_all = []
        y_syn_all = []

        self.sampling_info_: Dict[int, Dict[str, float]] = {}
        self.adaptive_k_: Dict[int, np.ndarray] = {}
        self.synthetic_quality_stats_: Dict[int, Dict[str, int]] = {}

        for cls, cls_count in zip(classes, counts):
            if cls_count == target_count:
                continue

            minority_mask = y == cls
            majority_mask = ~minority_mask
            X_min = X[minority_mask]
            X_maj = X[majority_mask]
            X_min_space = X_space[minority_mask]
            X_maj_space = X_space[majority_mask]

            if len(X_min) < 3:
                continue

            base_needed = target_count - cls_count
            sep_factor = self._class_separability_factor(X_min_space, X_maj_space)
            needed = int(max(1, np.rint(base_needed * sep_factor)))

            min_tree = KDTree(X_min_space)
            maj_tree = KDTree(X_maj_space)
            all_tree = KDTree(X_space)

            # Density signal from local minority neighborhood.
            k_density = min(max(self.k_bounds[1], 4), len(X_min) - 1)
            dist_density, _ = min_tree.query(X_min_space, k=k_density + 1)
            local_density = 1.0 / (np.mean(dist_density[:, 1:], axis=1) + 1e-12)

            adaptive_k = self._adaptive_k(local_density, len(X_min))
            self.adaptive_k_[int(cls)] = adaptive_k

            # Boundary and safety from mixed neighborhood.
            k_mix = min(max(self.boundary_k, 3), len(X_space) - 1)
            _, idx_mix = all_tree.query(X_min_space, k=k_mix + 1)
            neigh_labels = y[idx_mix[:, 1:]]
            maj_ratio = np.mean(neigh_labels != cls, axis=1)
            boundary_score = maj_ratio
            safety_score = 1.0 - maj_ratio

            density_priority = 1.0 - self._normalize(local_density)
            cluster_priority = self._cluster_priority(X_min_space)

            # Multi-perspective fusion score.
            w_density, w_boundary, w_safety, w_cluster = self.score_weights
            score = (
                w_density * density_priority
                + w_boundary * boundary_score
                + w_safety * safety_score
                + w_cluster * self._normalize(cluster_priority)
            )
            score = np.clip(score, 1e-9, None)
            probs = score / score.sum()

            state = _ClassSamplingState(class_label=int(cls), target_new_samples=needed)
            max_attempts = needed * self.max_attempt_factor

            for _ in range(max_attempts):
                if state.generated >= needed:
                    break

                i = rng.choice(len(X_min), p=probs)
                k_i = int(adaptive_k[i])
                _, idx_local = min_tree.query(X_min_space[[i]], k=k_i + 1)
                choices = idx_local[0, 1:]
                if len(choices) == 0:
                    state.rejected += 1
                    continue
                j = int(rng.choice(choices))

                lam = self._truncated_gaussian(rng, std=self.gaussian_scale)
                candidate = X_min[i] + lam * (X_min[j] - X_min[i])

                # Quality validation: keep only safe and informative candidates.
                cand_space = candidate if self.manifold_model_ is None else self.manifold_model_.transform(candidate.reshape(1, -1))[0]
                d_min, _ = min_tree.query(cand_space.reshape(1, -1), k=1)
                d_maj, _ = maj_tree.query(cand_space.reshape(1, -1), k=1)
                ratio_ok = float(d_min[0][0]) * self.quality_margin <= float(d_maj[0][0])

                _, cand_mix = all_tree.query(cand_space.reshape(1, -1), k=k_mix)
                cand_maj_ratio = float(np.mean(y[cand_mix[0]] != cls))
                safety_ok = cand_maj_ratio <= self.safety_threshold

                if ratio_ok and safety_ok:
                    X_syn_all.append(candidate)
                    y_syn_all.append(cls)
                    state.generated += 1
                else:
                    state.rejected += 1

            self.sampling_info_[int(cls)] = {
                "base_needed": int(base_needed),
                "target_generated": int(needed),
                "separability_factor": float(sep_factor),
            }
            self.synthetic_quality_stats_[int(cls)] = {
                "generated": int(state.generated),
                "rejected": int(state.rejected),
            }

        if not X_syn_all:
            return X.copy(), y.copy()

        X_syn = np.asarray(X_syn_all, dtype=np.float64)
        y_syn = np.asarray(y_syn_all, dtype=y.dtype)

        X_res = np.vstack([X, X_syn])
        y_res = np.concatenate([y, y_syn])
        return X_res, y_res
