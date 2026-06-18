import unittest

import numpy as np
from sklearn.datasets import make_classification

from ags_smote import AdaptiveGeometricSMOTE


class TestAdaptiveGeometricSMOTE(unittest.TestCase):
    def test_binary_resampling_increases_minority(self):
        X, y = make_classification(
            n_samples=600,
            n_features=20,
            n_informative=8,
            n_redundant=2,
            weights=[0.90, 0.10],
            class_sep=1.2,
            random_state=42,
        )
        model = AdaptiveGeometricSMOTE(random_state=42)
        X_res, y_res = model.fit_resample(X, y)

        before = np.bincount(y)
        after = np.bincount(y_res)

        self.assertGreater(after[1], before[1])
        self.assertGreaterEqual(after[1], int(before[0] * 0.5))
        self.assertEqual(X_res.shape[1], X.shape[1])

    def test_adaptive_k_changes_by_local_density(self):
        rng = np.random.RandomState(0)
        dense = rng.normal(0, 0.2, size=(80, 6))
        sparse = rng.normal(2, 1.2, size=(30, 6))
        majority = rng.normal(5, 1.0, size=(220, 6))

        X = np.vstack([dense, sparse, majority])
        y = np.array([1] * (len(dense) + len(sparse)) + [0] * len(majority))

        model = AdaptiveGeometricSMOTE(random_state=1, k_bounds=(3, 11))
        model.fit_resample(X, y)
        k_values = model.adaptive_k_[1]

        self.assertTrue(np.min(k_values) >= 3)
        self.assertTrue(np.max(k_values) <= 11)
        self.assertGreater(len(np.unique(k_values)), 1)

    def test_multiclass_support(self):
        X, y = make_classification(
            n_samples=900,
            n_classes=3,
            n_clusters_per_class=1,
            n_features=16,
            n_informative=10,
            weights=[0.70, 0.20, 0.10],
            random_state=21,
        )

        model = AdaptiveGeometricSMOTE(random_state=21)
        X_res, y_res = model.fit_resample(X, y)

        c_before = np.bincount(y)
        c_after = np.bincount(y_res)

        self.assertEqual(X_res.shape[1], X.shape[1])
        self.assertGreater(c_after[1], c_before[1])
        self.assertGreater(c_after[2], c_before[2])

    def test_quality_filter_rejects_some_noisy_candidates(self):
        X, y = make_classification(
            n_samples=500,
            n_features=12,
            n_informative=5,
            n_redundant=2,
            class_sep=0.6,
            flip_y=0.08,
            weights=[0.88, 0.12],
            random_state=17,
        )
        model = AdaptiveGeometricSMOTE(random_state=17)
        model.fit_resample(X, y)

        stats = model.synthetic_quality_stats_[1]
        self.assertGreater(stats["generated"], 0)
        self.assertGreater(stats["rejected"], 0)


if __name__ == "__main__":
    unittest.main()
