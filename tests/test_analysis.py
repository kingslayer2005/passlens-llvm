#!/usr/bin/env python3
"""
test_analysis.py — Unit tests for the statistics and model helpers used by
Phases 4 to 7.  Expected values are worked out by hand or checked against an
independent implementation (scikit-learn, the shap package).
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.analysis_common import (
    FULL, SMOKE, candidate_configs, fold_is_usable, holm_correction, importance_share,
    outer_splits, percentile_ci, pr_auc, program_bootstrap_indices, ranks_from_scores,
)


class TestPrAuc(unittest.TestCase):
    def test_hand_example(self):
        # Scores sorted: 0.9(+) 0.8(-) 0.7(+) 0.1(-); 2 positives.
        #   after 1st: precision 1/1, recall 1/2  -> adds 0.5 * 1
        #   after 3rd: precision 2/3, recall 2/2  -> adds 0.5 * 2/3
        # total = 0.5 + 0.3333 = 0.8333
        self.assertAlmostEqual(pr_auc(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.7, 0.1])),
                               0.5 + 1 / 3)

    def test_constant_score_equals_positive_rate(self):
        # A model that says the same thing for everyone scores the positive rate
        y = np.array([1, 0, 0, 1, 0, 0, 0, 0])
        self.assertAlmostEqual(pr_auc(y, np.full(8, 0.3)), 2 / 8)

    def test_matches_scikit_learn(self):
        from sklearn.metrics import average_precision_score
        rng = np.random.RandomState(0)
        for _ in range(20):
            y = rng.randint(0, 2, size=200)
            if y.min() == y.max():
                continue
            scores = np.round(rng.rand(200), 2)        # rounding creates ties
            self.assertAlmostEqual(pr_auc(y, scores), average_precision_score(y, scores), places=10)


class TestHolm(unittest.TestCase):
    def test_hand_example(self):
        # p = 0.01, 0.04, 0.03 ; m = 3
        # sorted: 0.01*3 = 0.03 ; 0.03*2 = 0.06 ; 0.04*1 = 0.04 -> raised to 0.06
        self.assertEqual([round(v, 4) for v in holm_correction([0.01, 0.04, 0.03])],
                         [0.03, 0.06, 0.06])

    def test_nan_is_not_a_test(self):
        adjusted = holm_correction([0.01, float("nan"), 0.04])
        self.assertAlmostEqual(adjusted[0], 0.02)     # m = 2, not 3
        self.assertTrue(np.isnan(adjusted[1]))
        self.assertAlmostEqual(adjusted[2], 0.04)

    def test_capped_at_one(self):
        self.assertEqual(holm_correction([0.6, 0.9]), [1.0, 1.0])


class TestRankingHelpers(unittest.TestCase):
    def test_ranks(self):
        self.assertEqual(ranks_from_scores(np.array([0.1, 0.5, 0.3])).tolist(), [3, 1, 2])

    def test_importance_share(self):
        # mean |value| per column: 2 and 6 -> shares 0.25 and 0.75
        values = np.array([[1.0, -6.0], [-3.0, 6.0]])
        self.assertEqual(importance_share(values).tolist(), [0.25, 0.75])

    def test_percentile_ci_ignores_nan(self):
        low, high = percentile_ci(np.array([np.nan] + list(range(101))), alpha=0.10)
        self.assertAlmostEqual(low, 5.0)
        self.assertAlmostEqual(high, 95.0)


class TestCrossValidation(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(1)
        self.groups = np.repeat([f"program_{i}" for i in range(12)], 20)
        self.y = rng.randint(0, 2, size=len(self.groups))

    def test_groups_never_on_both_sides(self):
        for repeat in range(FULL.n_repeats):
            splits = outer_splits(self.y, self.groups, repeat, FULL)
            self.assertEqual(len(splits), FULL.n_outer_folds)
            for train, test in splits:
                self.assertEqual(set(self.groups[train]) & set(self.groups[test]), set())
            # every sample is in exactly one test fold
            self.assertEqual(sorted(np.concatenate([t for _, t in splits]).tolist()),
                             list(range(len(self.y))))

    def test_same_seed_same_folds(self):
        a = outer_splits(self.y, self.groups, 0, FULL)
        b = outer_splits(self.y, self.groups, 0, FULL)
        for (train_a, test_a), (train_b, test_b) in zip(a, b):
            self.assertTrue(np.array_equal(test_a, test_b))

    def test_repeats_differ(self):
        a = outer_splits(self.y, self.groups, 0, FULL)
        b = outer_splits(self.y, self.groups, 1, FULL)
        self.assertFalse(all(np.array_equal(x[1], y[1]) for x, y in zip(a, b)))

    def test_few_programs(self):
        # with 3 programs there can be at most 3 folds
        groups = np.repeat(["a", "b", "c"], 10)
        y = np.tile([0, 1], 15)
        self.assertEqual(len(outer_splits(y, groups, 0, FULL)), 3)

    def test_fold_is_usable(self):
        self.assertTrue(fold_is_usable(np.array([0, 1]), np.array([1, 0]))[0])
        self.assertFalse(fold_is_usable(np.array([1, 1]), np.array([1, 0]))[0])
        self.assertFalse(fold_is_usable(np.array([0, 1]), np.array([0, 0]))[0])

    def test_bootstrap_resamples_whole_programs(self):
        for idx in program_bootstrap_indices(self.groups, n_bootstrap=5, seed=0):
            self.assertEqual(len(idx), len(self.groups))    # 12 programs x 20 rows each
            counts = {g: int((self.groups[idx] == g).sum()) for g in set(self.groups[idx])}
            self.assertTrue(all(c % 20 == 0 for c in counts.values()))


class TestConfigs(unittest.TestCase):
    def test_deterministic_and_distinct(self):
        a = candidate_configs("gvn", "fired", FULL)
        b = candidate_configs("gvn", "fired", FULL)
        self.assertEqual(a, b)
        self.assertEqual(len(a), FULL.n_tuning_configs)
        self.assertEqual(len({tuple(sorted(c.items())) for c in a}), len(a))

    def test_settings(self):
        self.assertEqual((FULL.n_repeats, FULL.n_outer_folds), (3, 5))
        self.assertEqual((FULL.keep_low, FULL.keep_high), (0.05, 0.95))
        self.assertEqual(SMOKE.name, "smoke")


class TestHeuristicScores(unittest.TestCase):
    def test_agreement_scores(self):
        from scripts.phase5_shap import agreement_scores
        # expected clusters ranked 1, 4 and 10 among 21 clusters
        scores = agreement_scores(np.array([1, 4, 10]), n_clusters=21)
        self.assertAlmostEqual(scores["hit_at_5"], 2 / 3)
        self.assertAlmostEqual(scores["mrr"], (1 + 1 / 4 + 1 / 10) / 3)
        self.assertAlmostEqual(scores["mean_percentile"], (0 + 3 + 9) / 3 / 20)

    def test_null_ranks(self):
        from scripts.phase5_shap import null_ranks
        ranks = null_ranks(n_clusters=10, n_expected=3, n_permutations=2000,
                           rng=np.random.RandomState(0))
        self.assertEqual(ranks.shape, (2000, 3))
        self.assertTrue(((ranks >= 1) & (ranks <= 10)).all())
        # ranks in one permutation are distinct
        self.assertTrue(all(len(set(row)) == 3 for row in ranks[:200]))
        # a random rank is uniform on 1..10, so its mean is about 5.5
        self.assertAlmostEqual(ranks.mean(), 5.5, delta=0.15)


class TestTreeShap(unittest.TestCase):
    """XGBoost's own TreeSHAP must agree with the shap package."""

    def test_additivity_and_agreement_with_shap(self):
        try:
            import xgboost
        except ImportError:
            self.skipTest("xgboost is not installed")
        from scripts.analysis_common import make_xgb, tree_shap
        rng = np.random.RandomState(0)
        X = rng.rand(300, 6).astype(np.float32)
        y = (X[:, 0] + 0.5 * X[:, 1] > 0.8).astype(int)
        model = make_xgb({"n_estimators": 30, "max_depth": 3, "learning_rate": 0.2,
                          "subsample": 1.0, "colsample_bytree": 1.0}, y, seed=0)
        model.fit(X, y)
        values = tree_shap(model, X)
        self.assertEqual(values.shape, (300, 6))

        # SHAP values plus the bias equal the model's raw (log-odds) output
        contributions = model.get_booster().predict(xgboost.DMatrix(X), pred_contribs=True)
        margin = model.predict(X, output_margin=True)
        self.assertTrue(np.allclose(contributions.sum(axis=1), margin, atol=1e-3))

        # The two features that define the label must rank first and second
        self.assertEqual(set(np.argsort(-np.abs(values).mean(axis=0))[:2].tolist()), {0, 1})

        try:
            import shap
            reference = np.asarray(shap.TreeExplainer(model).shap_values(X))
        except Exception as error:      # shap missing or not compatible with this xgboost
            self.skipTest(f"shap package not usable here: {type(error).__name__}")
        self.assertTrue(np.allclose(values, reference, atol=1e-3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
