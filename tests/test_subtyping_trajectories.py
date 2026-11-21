"""Pure-synthetic unit tests for the PD subtyping dual-trajectory pipeline (no private data required)."""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from sklearn.metrics import adjusted_rand_score

_HERE = Path(__file__).resolve().parent
_CODE_ROOT = _HERE.parent if _HERE.name == "tests" else _HERE
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import run_subtyping_trajectories as runner
import subtyping_registry as reg


class TestSubtypingTrajectories(unittest.TestCase):
    def test_resolve_data_dir_priority(self) -> None:
        marker = reg.extract_filename(reg.REFERENCE)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli = root / "cli"
            env = root / "env" / "curated"
            cli.mkdir(parents=True)
            env.mkdir(parents=True)
            (cli / marker).write_text("x", encoding="utf-8")
            (env / marker).write_text("x", encoding="utf-8")
            with mock.patch.dict(os.environ, {"PPMI_DATA_DIR": str(root / "env")}):
                self.assertEqual(reg.resolve_data_dir(cli), cli)
                self.assertEqual(reg.resolve_data_dir(None), env)

    def test_stratified_pd_reserve_deterministic_and_30pct(self) -> None:
        versions = ("20230612", "20240129", "20260810")
        pd_by_cut = {
            "20230612": {f"P{i}": {} for i in range(100)},
            "20240129": {f"P{i}": {} for i in range(200)},
            "20260810": {f"P{i}": {} for i in range(250)},
        }
        r1, m1 = reg.stratified_pd_reserve(pd_by_cut, versions=versions)
        r2, m2 = reg.stratified_pd_reserve(pd_by_cut, versions=versions)
        self.assertEqual(r1, r2)
        self.assertEqual(m1["eligible_pool"], 200)
        self.assertEqual(m1["total_reserved"], 60)
        self.assertTrue(all(int(pid[1:]) < 200 for pid in r1))

    def test_training_only_preprocessing_no_eval_leakage(self) -> None:
        train_x = np.array([[1.0, 10.0], [3.0, np.nan], [5.0, 30.0]])
        eval_x = np.array([[np.nan, 1000.0], [2.0, np.nan]])
        z_tr, z_ev, mean, scale = runner.preprocess_train_eval(train_x, eval_x)
        # Eval extreme outlier (1000.0) must NOT change training mean/scale or medians
        np.testing.assert_allclose(mean, [3.0, 20.0])
        # eval_x[0,0] imputed with train median 3.0 -> z = 0.0; eval_x[1,1] imputed with train median 20.0 -> z = 0.0
        self.assertAlmostEqual(float(z_ev[0, 0]), 0.0, places=10)
        self.assertAlmostEqual(float(z_ev[1, 1]), 0.0, places=10)

    def test_all_three_algorithms_and_leak_free_centroid_alignment(self) -> None:
        rng = np.random.default_rng(42)
        # 3 well-separated synthetic Gaussian clusters in 7D (col 2 is higher=better like MoCA)
        centers = np.array(
            [
                [-3.0, -3.0, +3.0, -3.0, -3.0, -3.0, -3.0],  # Mild
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],          # Intermediate
                [+3.0, +3.0, -3.0, +3.0, +3.0, +3.0, +3.0],  # Severe
            ]
        )
        train_x = np.vstack([rng.normal(c, 0.6, size=(60, 7)) for c in centers])
        eval_x = np.vstack([rng.normal(c, 0.6, size=(25, 7)) for c in centers])
        true_eval = np.repeat([0, 1, 2], 25)
        signs = np.array([1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0])

        z_tr, z_ev, m_ref, s_ref = runner.preprocess_train_eval(train_x, eval_x)
        for alg in reg.ALGORITHMS:
            _, ev_raw, c_raw, ok = runner.fit_and_predict_clusters(alg, 3, z_tr, z_ev, fit_seed=7, is_reference_or_release=True)
            self.assertTrue(ok)
            remap = runner.canonical_order_by_severity(c_raw, signs)
            ev_ord = remap[ev_raw]
            self.assertGreater(adjusted_rand_score(true_eval, ev_ord), 0.95)
            # Mildest cluster must be 0, most severe must be 2
            self.assertEqual(int(np.bincount(ev_ord[:25]).argmax()), 0)
            self.assertEqual(int(np.bincount(ev_ord[50:]).argmax()), 2)

            # Aligning a permuted version of the same centroids recovers exact zero flip rate
            ref_c = np.empty_like(c_raw)
            for oi, ni in enumerate(remap):
                ref_c[ni] = c_raw[oi]
            perm = np.array([2, 0, 1])
            al_map, l2 = runner.align_by_training_centroids(ref_c[perm], m_ref, s_ref, ref_c, m_ref, s_ref)
            self.assertLess(l2, 1e-12)
            np.testing.assert_array_equal(al_map, perm)

    def test_tie_permutation_preserves_snapshot_order(self) -> None:
        cut_rank = np.array([0] * 30 + [1] * 20 + [2] * 10, dtype=int)
        o0 = runner.tie_permutation(cut_rank, "P1__kmeans__k3", 0)
        o1 = runner.tie_permutation(cut_rank, "P1__kmeans__k3", 1)
        np.testing.assert_array_equal(cut_rank[o0], cut_rank)
        self.assertEqual(set(o0[:30]), set(range(30)))
        self.assertFalse(np.array_equal(o0, o1))

    def test_external_validity_metrics_exact(self) -> None:
        labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        anchors = {name: np.full(9, np.nan) for name, _, _ in reg.EXTERNAL_ANCHORS}
        anchors["2y_moca"] = np.array([28.0, 28.0, 28.0, 25.0, 25.0, 25.0, 20.0, 20.0, 20.0])
        anchors["2y_pm_any"] = np.array([0.0, 0.0, 0.0, 0.0, 1.0, np.nan, 1.0, 1.0, 1.0])
        res = runner.external_validity_metrics(labels, 3, anchors)
        self.assertAlmostEqual(res["2y_moca"][0], 1.0, places=12)
        self.assertAlmostEqual(res["2y_moca"][1], -8.0, places=12)
        self.assertAlmostEqual(res["2y_pm_any"][1], 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
