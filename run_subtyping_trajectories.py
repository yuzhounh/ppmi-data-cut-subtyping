"""End-to-end dual-trajectory runner for PPMI unsupervised PD subtyping stability & external prognostic validity."""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import hashlib
import json
import math
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import sklearn
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

import subtyping_registry as reg

SEED = 20261121
ROUNDS = 1000
PILOT_ROUNDS = 10

_IN_CODE = Path(__file__).resolve().parent.name == "code"
DEFAULT_RESULTS = (
    reg.PROJECT / "results" / "unified_subtyping_trajectories"
    if _IN_CODE
    else reg.PROJECT / "outputs" / "results"
)
DEFAULT_CACHE = reg.PROJECT / "data" / "derived" / "unified_subtyping_trajectories"


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_arrays(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def halving(total: int) -> tuple[int, ...]:
    end = min(range(total.bit_length()), key=lambda k: abs(math.log(total / (30 * 2**k))))
    return tuple(math.floor(total / 2**k + 0.5) for k in range(end + 1))


def random_stream(tag: str, repeat: int, purpose: int = 2) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([SEED, int.from_bytes(tag.encode("ascii")), repeat, purpose])
    )


def records_match_exact(obs: tuple[int, tuple] | None, ref: tuple[int, tuple]) -> bool:
    """Exact match on baseline month, all panel features, and missingness pattern (no age in clustering panels)."""
    if obs is None:
        return False
    return obs[0] == ref[0] and obs[1] == ref[1]


def preprocess_train_eval(train_x: np.ndarray, eval_x: np.ndarray):
    """Training-only median imputation and Z-score standardization applied to train_x and eval_x."""
    miss_tr = np.isnan(train_x)
    miss_ev = np.isnan(eval_x)
    medians = np.zeros(train_x.shape[1], dtype=float)
    for col in range(train_x.shape[1]):
        present = train_x[~miss_tr[:, col], col]
        if present.size > 0:
            medians[col] = float(np.median(present))
    tr_imp = np.where(miss_tr, medians, train_x)
    ev_imp = np.where(miss_ev, medians, eval_x)
    mean = tr_imp.mean(axis=0)
    scale = tr_imp.std(axis=0)
    scale[scale == 0.0] = 1.0
    return (tr_imp - mean) / scale, (ev_imp - mean) / scale, mean, scale


def fit_and_predict_clusters(
    algorithm: str,
    k: int,
    z_train: np.ndarray,
    z_eval: np.ndarray,
    fit_seed: int,
    is_reference_or_release: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Fit clustering on z_train only and predict labels on z_eval.

    Returns (train_labels, eval_labels, centroids_in_z_train, converged_and_full_k).
    """
    if algorithm == "kmeans":
        n_init = 50 if is_reference_or_release else 20
        model = KMeans(n_clusters=k, n_init=n_init, max_iter=300, tol=1e-5, random_state=fit_seed)
        train_labels = model.fit_predict(z_train)
        eval_labels = model.predict(z_eval)
        centroids = model.cluster_centers_
        ok = len(np.unique(train_labels)) == k
        return train_labels, eval_labels, centroids, ok

    if algorithm == "ward":
        z_link = linkage(z_train, method="ward")
        train_labels = fcluster(z_link, t=k, criterion="maxclust") - 1
        ok = len(np.unique(train_labels)) == k
        if not ok:
            return train_labels, np.zeros(len(z_eval), dtype=int), np.zeros((k, z_train.shape[1])), False
        centroids = np.vstack([z_train[train_labels == c].mean(axis=0) for c in range(k)])
        eval_labels = np.argmin(cdist(z_eval, centroids, metric="euclidean"), axis=1)
        return train_labels, eval_labels, centroids, True

    if algorithm == "gmm":
        n_init = 50 if is_reference_or_release else 20
        model = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            reg_covar=1e-4,
            tol=1e-4,
            n_init=n_init,
            max_iter=300,
            random_state=fit_seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(z_train)
        train_labels = model.predict(z_train)
        eval_labels = model.predict(z_eval)
        centroids = model.means_
        ok = bool(model.converged_) and (len(np.unique(train_labels)) == k)
        return train_labels, eval_labels, centroids, ok

    raise ValueError(f"Unknown algorithm: {algorithm}")


def canonical_order_by_severity(centroids_ref: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Return permutation mapping old cluster index -> canonical severity rank (0 = mildest, K-1 = most severe)."""
    severity = (centroids_ref * signs).mean(axis=1)
    order = np.argsort(severity, kind="mergesort")
    remap = np.empty(len(order), dtype=int)
    for new_rank, old_idx in enumerate(order):
        remap[old_idx] = new_rank
    return remap


def align_by_training_centroids(
    centroids_z: np.ndarray,
    mean_tr: np.ndarray,
    scale_tr: np.ndarray,
    centroids_ref_z: np.ndarray,
    mean_ref: np.ndarray,
    scale_ref: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Leak-free Hungarian alignment matching training centroids projected into reference standardized space."""
    raw_c = centroids_z * scale_tr + mean_tr
    c_in_ref = (raw_c - mean_ref) / scale_ref
    cost = cdist(c_in_ref, centroids_ref_z, metric="sqeuclidean")
    row_ind, col_ind = linear_sum_assignment(cost)
    remap = np.empty(len(row_ind), dtype=int)
    for r, c in zip(row_ind, col_ind):
        remap[r] = c
    mean_l2 = float(np.sqrt(cost[row_ind, col_ind]).mean())
    return remap, mean_l2


def external_validity_metrics(eval_labels_aligned: np.ndarray, k: int, anchor_arrays: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    """Compute (eta2, severe_minus_mild) for each external 2-year clinical anchor on fixed evaluation set."""
    out = {}
    for name, _, _ in reg.EXTERNAL_ANCHORS:
        y = anchor_arrays[name]
        mask = np.isfinite(y)
        yv = y[mask]
        lv = eval_labels_aligned[mask]
        if yv.size < 2:
            out[name] = (math.nan, math.nan)
            continue
        grand_mean = float(yv.mean())
        sst = float(((yv - grand_mean) ** 2).sum())
        ssb = 0.0
        for c in range(k):
            cm = lv == c
            if cm.any():
                ssb += float(cm.sum()) * (float(yv[cm].mean()) - grand_mean) ** 2
        eta2 = ssb / sst if sst > 0 else math.nan
        mild = yv[lv == 0]
        sev = yv[lv == (k - 1)]
        contrast = float(sev.mean() - mild.mean()) if (mild.size > 0 and sev.size > 0) else math.nan
        out[name] = (eta2, contrast)
    return out


def run_identity(data_dir: Path) -> dict:
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(reg.__file__)):
        digest.update(path.read_bytes())
    snapshots = {}
    for v in reg.VERSIONS:
        info = (data_dir / reg.extract_filename(v)).stat()
        snapshots[v] = [info.st_size, info.st_mtime_ns]
    return {
        "code_digest": digest.hexdigest(),
        "snapshots": snapshots,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "reserve_seed": reg.RESERVE_SEED,
        "reserve_fraction": reg.RESERVE_FRACTION,
        "min_observed_fraction": reg.MIN_OBSERVED_FRACTION,
        "tie_seed": SEED,
        "tie_rounds": ROUNDS,
    }


def prepare_panels(data_dir: Path, results_dir: Path, cache_dir: Path, identity: dict) -> dict:
    inputs_dir = cache_dir / "inputs"
    marker = inputs_dir / "manifest.json"
    if marker.exists() and (results_dir / "panel_inventory.json").exists() and read_json(marker)["identity"] == identity:
        return read_json(results_dir / "panel_inventory.json")

    print("Loading all 11 official curated extracts for PD subtyping...", flush=True)
    pd_by_cut = {}
    for v in reg.VERSIONS:
        _, people, _ = reg.load_pd_release(data_dir, v)
        pd_by_cut[v] = people
        print(f"  {v}: {len(people):,} baseline PD participants", flush=True)

    reserved, reserve_meta = reg.stratified_pd_reserve(pd_by_cut)
    common_all_11 = set.intersection(*(set(pd_by_cut[v]) for v in reg.VERSIONS))
    fixed_eval_11 = sorted(common_all_11 & reserved)

    inputs_dir.mkdir(parents=True, exist_ok=True)
    panel_meta = {}

    for panel in reg.PANELS:
        by_v_elig = {v: reg.extract_panel_eligible(pd_by_cut[v], panel) for v in reg.VERSIONS}
        # Fixed evaluation set for this panel: reserved 11-cut PD meeting panel >=80% eligibility in 20260810
        ref_elig = by_v_elig[reg.REFERENCE]
        eval_ids = sorted(p for p in fixed_eval_11 if p in ref_elig)
        eval_x = np.asarray([ref_elig[p][1] for p in eval_ids], dtype=float)
        anchors = reg.extract_external_anchors(pd_by_cut[reg.REFERENCE], eval_ids)

        release_arrays = {}
        release_counts = {}
        for v in reg.VERSIONS:
            dev_ids = sorted(p for p in by_v_elig[v] if p not in reserved)
            assert set(dev_ids).isdisjoint(eval_ids)
            release_arrays[f"x_{v}"] = np.asarray([by_v_elig[v][p][1] for p in dev_ids], dtype=float)
            release_counts[v] = {
                "dev_n": len(dev_ids),
                "subgroups": reg.subgroup_summary(pd_by_cut[v], dev_ids),
            }

        # Chronological backward verification in 20260810 dev pool
        ref_dev_ids = sorted(p for p in ref_elig if p not in reserved)
        membership = np.zeros((len(reg.VERSIONS), len(ref_dev_ids)), dtype=bool)
        for j, v in enumerate(reg.VERSIONS):
            obs_v = by_v_elig[v]
            for i, pid in enumerate(ref_dev_ids):
                if records_match_exact(obs_v.get(pid), ref_elig[pid]):
                    membership[j, i] = True
        assert membership[-1].all()
        stable = np.logical_and.accumulate(membership[::-1], axis=0)
        first_idx = len(reg.VERSIONS) - stable.sum(axis=0)
        order = np.lexsort((np.arange(len(ref_dev_ids)), first_idx))
        chrono_x = np.asarray([ref_elig[p][1] for p in ref_dev_ids], dtype=float)[order]

        anchor_npz = {f"anchor_{name}": np.asarray(vals, dtype=float) for name, vals in anchors.items()}
        save_arrays(
            inputs_dir / f"{panel.id}.npz",
            **release_arrays,
            eval_x=eval_x,
            eval_ids=np.asarray(eval_ids, dtype=str),
            chrono_x=chrono_x,
            chrono_ids=np.asarray(ref_dev_ids, dtype=str)[order],
            cutoff_idx=first_idx[order],
            membership=membership[:, order],
            severity_signs=np.asarray(reg.severity_signs(panel), dtype=float),
            **anchor_npz,
        )

        anchor_n = {
            name: {
                "n": int(np.isfinite(vals).sum()),
                "events": int(np.nansum(vals)) if kind == "binary" else None,
            }
            for (name, _, kind), vals in zip(reg.EXTERNAL_ANCHORS, anchors.values())
        }
        panel_meta[panel.id] = {
            "label": panel.label,
            "features": list(panel.features),
            "eval_n": len(eval_ids),
            "eval_subgroups": reg.subgroup_summary(pd_by_cut[reg.REFERENCE], eval_ids),
            "dev_202608_n": len(ref_dev_ids),
            "dev_202306_verified_n": int((first_idx == 0).sum()),
            "release_counts": release_counts,
            "external_anchors": anchor_n,
        }

    inv = {
        "versions": list(reg.VERSIONS),
        "reference": reg.REFERENCE,
        "reserve": reserve_meta,
        "fixed_eval_11_pd": len(fixed_eval_11),
        "panels": panel_meta,
    }
    save_json(results_dir / "panel_inventory.json", inv)
    save_json(marker, {"identity": identity, "panels": list(panel_meta)})
    print(f"Prepared {len(panel_meta)} PD clustering panels cleanly.", flush=True)
    return inv


def checked_stages(block, versions: tuple[str, ...]) -> list[dict]:
    x = block["chrono_x"]
    ids, ev_ids = block["chrono_ids"], block["eval_ids"]
    cut_rank = block["cutoff_idx"]
    membership = block["membership"]
    full = len(x)
    assert len(np.unique(ids)) == full
    assert not (set(ids) & set(ev_ids))
    assert np.all(np.diff(cut_rank) >= 0) and membership[-1].all()
    former_n, last_cut = 0, -1
    stages = []
    for n in reversed(halving(full)):
        cutoff = int(cut_rank[n - 1])
        assert cutoff >= last_cut and n > former_n and membership[cutoff, :n].all()
        earlier_n = int(np.count_nonzero(cut_rank < cutoff))
        tied_total = int(np.count_nonzero(cut_rank == cutoff))
        tied_used = n - earlier_n
        assert 1 <= tied_used <= tied_total
        stages.append(
            {
                "training_n": n,
                "verified_cut": versions[cutoff],
                "cutoff_idx": cutoff,
                "observation_through": reg.calendar_month(reg.snapshot_month(versions[cutoff])),
                "earlier_fixed_n": earlier_n,
                "boundary_group_n": tied_total,
                "used_from_boundary_group": tied_used,
                "boundary_group_split": tied_used < tied_total,
            }
        )
        former_n, last_cut = n, cutoff
    return stages


def tie_permutation(cut_rank: np.ndarray, strategy_id: str, repeat: int) -> np.ndarray:
    rng = random_stream(strategy_id, repeat, purpose=2)
    order = np.arange(len(cut_rank))
    starts = np.flatnonzero(np.concatenate(([True], np.diff(cut_rank) > 0)))
    ends = np.concatenate((starts[1:], [len(cut_rank)]))
    for s, e in zip(starts, ends):
        if e - s > 1:
            order[s:e] = rng.permutation(order[s:e])
    return order


def evaluate_strategy(panel_id: str, algorithm: str, k: int, cache_dir: Path, rounds: int, identity: dict, pilot: bool = False) -> dict:
    strategy_id = f"{panel_id}__{algorithm}__k{k}"
    versions = reg.VERSIONS
    anchor_names = [name for name, _, _ in reg.EXTERNAL_ANCHORS]

    with np.load(cache_dir / "inputs" / f"{panel_id}.npz") as block:
        eval_x = block["eval_x"]
        eval_n = len(eval_x)
        signs = block["severity_signs"]
        anchors = {name: block[f"anchor_{name}"] for name in anchor_names}

        # 1. Fit 20260810 full-pool reference solution & canonically order clusters by baseline severity
        ref_x = block[f"x_{reg.REFERENCE}"]
        z_ref_tr, z_ref_ev, mean_ref, scale_ref = preprocess_train_eval(ref_x, eval_x)
        ref_tr_raw, ref_ev_raw, ref_c_raw, ref_ok = fit_and_predict_clusters(
            algorithm, k, z_ref_tr, z_ref_ev, fit_seed=SEED, is_reference_or_release=True
        )
        assert ref_ok, f"Reference fit failed for {strategy_id}"

        # In pilot mode, cross-check Ward against sklearn AgglomerativeClustering
        if pilot and algorithm == "ward":
            sk_ward = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(z_ref_tr)
            assert adjusted_rand_score(ref_tr_raw, sk_ward) > 1.0 - 1e-12, f"Ward mismatch for {strategy_id}"

        ref_remap = canonical_order_by_severity(ref_c_raw, signs)
        ref_ev_labels = ref_remap[ref_ev_raw]
        ref_centroids = np.empty_like(ref_c_raw)
        for old_i, new_i in enumerate(ref_remap):
            ref_centroids[new_i] = ref_c_raw[old_i]
        ref_ext = external_validity_metrics(ref_ev_labels, k, anchors)

        # 2. Trajectory A: 11 official curated releases
        started = time.perf_counter()
        release_rows = []
        release_ext_rows = []
        prev_ev_labels = None
        for v in versions:
            x_v = block[f"x_{v}"]
            z_tr, z_ev, mean_tr, scale_tr = preprocess_train_eval(x_v, eval_x)
            _, ev_raw, c_raw, ok = fit_and_predict_clusters(
                algorithm, k, z_tr, z_ev, fit_seed=SEED, is_reference_or_release=True
            )
            assert ok, f"Official release fit failed for {strategy_id} @ {v}"
            remap, c_shift = align_by_training_centroids(c_raw, mean_tr, scale_tr, ref_centroids, mean_ref, scale_ref)
            ev_aligned = remap[ev_raw]
            ari_ref = float(adjusted_rand_score(ref_ev_labels, ev_aligned))
            nmi_ref = float(normalized_mutual_info_score(ref_ev_labels, ev_aligned))
            flip_ref = float(np.mean(ev_aligned != ref_ev_labels))
            step_ari = float(adjusted_rand_score(prev_ev_labels, ev_aligned)) if prev_ev_labels is not None else math.nan
            sil = float(silhouette_score(z_ev, ev_aligned)) if len(np.unique(ev_aligned)) >= 2 else math.nan
            shares = np.bincount(ev_aligned, minlength=k) / eval_n
            if v == reg.REFERENCE:
                assert abs(ari_ref - 1.0) < 1e-12 and abs(flip_ref) < 1e-12 and c_shift < 1e-11

            release_rows.append(
                {
                    "strategy": strategy_id,
                    "panel": panel_id,
                    "algorithm": algorithm,
                    "k": k,
                    "version": v,
                    "training_n": int(len(x_v)),
                    "evaluation_n": eval_n,
                    "ari_to_ref": ari_ref,
                    "nmi_to_ref": nmi_ref,
                    "flip_rate_to_ref": flip_ref,
                    "step_ari": step_ari,
                    "centroid_shift_l2": c_shift,
                    "silhouette_eval": sil,
                    "min_cluster_share": float(shares.min()),
                    "max_cluster_share": float(shares.max()),
                    "cluster_shares": json.dumps([round(float(s), 4) for s in shares]),
                }
            )
            ext_v = external_validity_metrics(ev_aligned, k, anchors)
            for aname in anchor_names:
                eta2, contrast = ext_v[aname]
                release_ext_rows.append(
                    {
                        "strategy": strategy_id,
                        "panel": panel_id,
                        "algorithm": algorithm,
                        "k": k,
                        "version": v,
                        "training_n": int(len(x_v)),
                        "anchor": aname,
                        "eta2": eta2,
                        "severe_minus_mild": contrast,
                    }
                )
            prev_ev_labels = ev_aligned

        # 3. Trajectory B: Chronological doubling within 20260810 (1,000 nested within-snapshot permutations)
        stages = checked_stages(block, versions)
        chrono_x = block["chrono_x"]
        chrono_ids = block["chrono_ids"]
        cut_rank = block["cutoff_idx"]
        n_stages = len(stages)
        assert stages[-1]["training_n"] == len(ref_x)

        ari_mat = np.full((rounds, n_stages), np.nan)
        nmi_mat = np.full((rounds, n_stages), np.nan)
        flip_mat = np.full((rounds, n_stages), np.nan)
        shift_mat = np.full((rounds, n_stages), np.nan)
        sil_mat = np.full((rounds, n_stages), np.nan)
        min_sh_mat = np.full((rounds, n_stages), np.nan)
        max_sh_mat = np.full((rounds, n_stages), np.nan)
        step_ari_mat = np.full((rounds, n_stages - 1), np.nan)
        ext_eta2 = np.full((rounds, n_stages, len(anchor_names)), np.nan)
        ext_diff = np.full((rounds, n_stages, len(anchor_names)), np.nan)

        # Precompute unsplit stages once (includes the full-size reference anchor in canonical PATNO order!)
        unsplit_cache = {}
        for col, st in enumerate(stages):
            if not st["boundary_group_split"]:
                n = st["training_n"]
                idx = np.arange(n)
                canon_idx = idx[np.argsort(chrono_ids[idx], kind="mergesort")]
                z_tr, z_ev, mean_tr, scale_tr = preprocess_train_eval(chrono_x[canon_idx], eval_x)
                _, ev_raw, c_raw, ok = fit_and_predict_clusters(
                    algorithm, k, z_tr, z_ev, fit_seed=SEED, is_reference_or_release=True
                )
                assert ok
                remap, c_shift = align_by_training_centroids(c_raw, mean_tr, scale_tr, ref_centroids, mean_ref, scale_ref)
                ev_aligned = remap[ev_raw]
                ari_v = float(adjusted_rand_score(ref_ev_labels, ev_aligned))
                nmi_v = float(normalized_mutual_info_score(ref_ev_labels, ev_aligned))
                flip_v = float(np.mean(ev_aligned != ref_ev_labels))
                sil_v = float(silhouette_score(z_ev, ev_aligned)) if len(np.unique(ev_aligned)) >= 2 else math.nan
                shares = np.bincount(ev_aligned, minlength=k) / eval_n
                ext_v = external_validity_metrics(ev_aligned, k, anchors)
                if col == n_stages - 1:
                    # Full-size anchor must exactly match 20260810 official release (zero drift)
                    assert abs(ari_v - 1.0) < 1e-12 and abs(flip_v) < 1e-12 and c_shift < 1e-11
                unsplit_cache[col] = (ev_aligned, ari_v, nmi_v, flip_v, c_shift, sil_v, float(shares.min()), float(shares.max()), ext_v)

        ckpt_path = cache_dir / f"{strategy_id}_ckpt.npz"
        start = 0
        if not pilot and ckpt_path.exists():
            with np.load(ckpt_path) as ckpt:
                if ckpt["identity"].item() == identity["code_digest"] and ckpt["ari_mat"].shape[1] == n_stages:
                    start = min(rounds, int(ckpt["completed"].item()))
                    ari_mat[:start] = ckpt["ari_mat"][:start]
                    nmi_mat[:start] = ckpt["nmi_mat"][:start]
                    flip_mat[:start] = ckpt["flip_mat"][:start]
                    shift_mat[:start] = ckpt["shift_mat"][:start]
                    sil_mat[:start] = ckpt["sil_mat"][:start]
                    min_sh_mat[:start] = ckpt["min_sh_mat"][:start]
                    max_sh_mat[:start] = ckpt["max_sh_mat"][:start]
                    step_ari_mat[:start] = ckpt["step_ari_mat"][:start]
                    ext_eta2[:start] = ckpt["ext_eta2"][:start]
                    ext_diff[:start] = ckpt["ext_diff"][:start]

        for r in range(start, rounds):
            order = tie_permutation(cut_rank, strategy_id, r)
            if pilot:
                assert np.array_equal(cut_rank[order], cut_rank)
                assert len(np.unique(order)) == len(cut_rank)
                prev_idx_set: set[int] = set()
            prev_round_labels = None
            for col, st in enumerate(stages):
                n = st["training_n"]
                if col in unsplit_cache:
                    ev_aligned, ari_v, nmi_v, flip_v, c_shift, sil_v, mns, mxs, ext_v = unsplit_cache[col]
                    if pilot:
                        cur_set = set(order[:n])
                        assert prev_idx_set < cur_set
                        prev_idx_set = cur_set
                else:
                    idx = order[:n]
                    if pilot:
                        cur_set = set(idx)
                        assert len(cur_set) == n and prev_idx_set < cur_set
                        earlier_fixed = set(np.flatnonzero(cut_rank < st["cutoff_idx"]))
                        assert earlier_fixed <= cur_set
                        prev_idx_set = cur_set
                    canon_idx = idx[np.argsort(chrono_ids[idx], kind="mergesort")]
                    z_tr, z_ev, mean_tr, scale_tr = preprocess_train_eval(chrono_x[canon_idx], eval_x)
                    _, ev_raw, c_raw, ok = fit_and_predict_clusters(
                        algorithm, k, z_tr, z_ev, fit_seed=SEED + r, is_reference_or_release=False
                    )
                    if not ok:
                        prev_round_labels = None
                        continue
                    remap, c_shift = align_by_training_centroids(c_raw, mean_tr, scale_tr, ref_centroids, mean_ref, scale_ref)
                    ev_aligned = remap[ev_raw]
                    ari_v = float(adjusted_rand_score(ref_ev_labels, ev_aligned))
                    nmi_v = float(normalized_mutual_info_score(ref_ev_labels, ev_aligned))
                    flip_v = float(np.mean(ev_aligned != ref_ev_labels))
                    sil_v = float(silhouette_score(z_ev, ev_aligned)) if len(np.unique(ev_aligned)) >= 2 else math.nan
                    shares = np.bincount(ev_aligned, minlength=k) / eval_n
                    mns, mxs = float(shares.min()), float(shares.max())
                    ext_v = external_validity_metrics(ev_aligned, k, anchors)

                ari_mat[r, col] = ari_v
                nmi_mat[r, col] = nmi_v
                flip_mat[r, col] = flip_v
                shift_mat[r, col] = c_shift
                sil_mat[r, col] = sil_v
                min_sh_mat[r, col] = mns
                max_sh_mat[r, col] = mxs
                for ai, aname in enumerate(anchor_names):
                    ext_eta2[r, col, ai], ext_diff[r, col, ai] = ext_v[aname]
                if col > 0 and prev_round_labels is not None:
                    step_ari_mat[r, col - 1] = float(adjusted_rand_score(prev_round_labels, ev_aligned))
                prev_round_labels = ev_aligned

            if not pilot and ((r + 1) % 250 == 0 or r + 1 == rounds):
                save_arrays(
                    ckpt_path,
                    identity=identity["code_digest"],
                    completed=r + 1,
                    ari_mat=ari_mat[: r + 1],
                    nmi_mat=nmi_mat[: r + 1],
                    flip_mat=flip_mat[: r + 1],
                    shift_mat=shift_mat[: r + 1],
                    sil_mat=sil_mat[: r + 1],
                    min_sh_mat=min_sh_mat[: r + 1],
                    max_sh_mat=max_sh_mat[: r + 1],
                    step_ari_mat=step_ari_mat[: r + 1],
                    ext_eta2=ext_eta2[: r + 1],
                    ext_diff=ext_diff[: r + 1],
                )

        elapsed = time.perf_counter() - started
        if pilot:
            return {
                "strategy": strategy_id,
                "release_points": len(release_rows),
                "checked_stages": n_stages,
                "first_cut": stages[0]["verified_cut"],
                "degenerate_draws": int(np.isnan(ari_mat).sum()),
                "fit_seconds": round(elapsed, 3),
                "ref_2y_moca_eta2": round(ref_ext["2y_moca"][0], 4),
                "ref_2y_pm_any_diff": round(ref_ext["2y_pm_any"][1], 4),
            }

        def summarize_vec(arr: np.ndarray) -> tuple[float, float, float, float, int]:
            fin = arr[np.isfinite(arr)]
            if not len(fin):
                return math.nan, math.nan, math.nan, math.nan, 0
            p025, p50, p975 = np.percentile(fin, [2.5, 50.0, 97.5])
            return float(fin.mean()), float(p025), float(p50), float(p975), int(len(fin))

        chrono_rows = []
        chrono_ext_rows = []
        for col, st in enumerate(stages):
            ari_m, ari_lo, ari_med, ari_hi, valid_n = summarize_vec(ari_mat[:, col])
            nmi_m, nmi_lo, _, nmi_hi, _ = summarize_vec(nmi_mat[:, col])
            flp_m, flp_lo, _, flp_hi, _ = summarize_vec(flip_mat[:, col])
            shf_m, shf_lo, _, shf_hi, _ = summarize_vec(shift_mat[:, col])
            sil_m, _, _, _, _ = summarize_vec(sil_mat[:, col])
            mns_m, _, _, _, _ = summarize_vec(min_sh_mat[:, col])
            mxs_m, _, _, _, _ = summarize_vec(max_sh_mat[:, col])
            chrono_rows.append(
                {
                    "strategy": strategy_id,
                    "panel": panel_id,
                    "algorithm": algorithm,
                    "k": k,
                    "training_n": st["training_n"],
                    "evaluation_n": eval_n,
                    "verified_cut": st["verified_cut"],
                    "observation_through": st["observation_through"],
                    "earlier_fixed_n": st["earlier_fixed_n"],
                    "boundary_group_n": st["boundary_group_n"],
                    "used_from_boundary_group": st["used_from_boundary_group"],
                    "boundary_group_split": st["boundary_group_split"],
                    "attempted_rounds": rounds,
                    "valid_rounds": valid_n,
                    "degenerate_rounds": rounds - valid_n,
                    "ari_mean": ari_m,
                    "ari_p025": ari_lo,
                    "ari_median": ari_med,
                    "ari_p975": ari_hi,
                    "nmi_mean": nmi_m,
                    "nmi_p025": nmi_lo,
                    "nmi_p975": nmi_hi,
                    "flip_rate_mean": flp_m,
                    "flip_rate_p025": flp_lo,
                    "flip_rate_p975": flp_hi,
                    "centroid_shift_mean": shf_m,
                    "centroid_shift_p025": shf_lo,
                    "centroid_shift_p975": shf_hi,
                    "silhouette_mean": sil_m,
                    "min_cluster_share_mean": mns_m,
                    "max_cluster_share_mean": mxs_m,
                }
            )
            for ai, aname in enumerate(anchor_names):
                e_m, e_lo, _, e_hi, e_v = summarize_vec(ext_eta2[:, col, ai])
                d_m, d_lo, _, d_hi, d_v = summarize_vec(ext_diff[:, col, ai])
                chrono_ext_rows.append(
                    {
                        "strategy": strategy_id,
                        "panel": panel_id,
                        "algorithm": algorithm,
                        "k": k,
                        "training_n": st["training_n"],
                        "verified_cut": st["verified_cut"],
                        "anchor": aname,
                        "valid_eta2_rounds": e_v,
                        "eta2_mean": e_m,
                        "eta2_p025": e_lo,
                        "eta2_p975": e_hi,
                        "valid_contrast_rounds": d_v,
                        "severe_minus_mild_mean": d_m,
                        "severe_minus_mild_p025": d_lo,
                        "severe_minus_mild_p975": d_hi,
                    }
                )

        step_rows = []
        for col in range(1, n_stages):
            s_m, s_lo, s_med, s_hi, s_v = summarize_vec(step_ari_mat[:, col - 1])
            step_rows.append(
                {
                    "strategy": strategy_id,
                    "panel": panel_id,
                    "algorithm": algorithm,
                    "k": k,
                    "smaller_n": stages[col - 1]["training_n"],
                    "larger_n": stages[col]["training_n"],
                    "smaller_cut": stages[col - 1]["verified_cut"],
                    "larger_cut": stages[col]["verified_cut"],
                    "valid_paired_rounds": s_v,
                    "step_ari_mean": s_m,
                    "step_ari_p025": s_lo,
                    "step_ari_median": s_med,
                    "step_ari_p975": s_hi,
                }
            )

        print(f"{strategy_id:42s} {rounds}r {elapsed:5.2f}s (deg={int(np.isnan(ari_mat).sum())})", flush=True)
        return {
            "strategy": strategy_id,
            "releases": release_rows,
            "release_ext": release_ext_rows,
            "chrono": chrono_rows,
            "chrono_ext": chrono_ext_rows,
            "steps": step_rows,
            "fit_seconds": elapsed,
        }


def _worker(args):
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        return evaluate_strategy(*args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    data_dir = reg.resolve_data_dir(args.data_dir)
    results_dir = args.results_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    identity = run_identity(data_dir)
    started = time.perf_counter()

    inv = prepare_panels(data_dir, results_dir, cache_dir, identity)

    jobs = [
        (panel.id, alg, k, cache_dir, PILOT_ROUNDS if args.pilot else ROUNDS, identity, args.pilot)
        for panel in reg.PANELS
        for alg in reg.ALGORITHMS
        for k in reg.CLUSTER_COUNTS
    ]
    # Run heavier GMM jobs first for better parallel load balance
    alg_weight = {"gmm": 0, "kmeans": 1, "ward": 2}
    ordered = sorted(range(len(jobs)), key=lambda i: (alg_weight[jobs[i][1]], -jobs[i][2]))

    def run_jobs():
        if args.workers == 1:
            return [_worker(job) for job in jobs]
        out = [None] * len(jobs)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_worker, jobs[i]): i for i in ordered}
            for fut, i in futs.items():
                out[i] = fut.result()
        return out

    if args.pilot:
        t0 = time.perf_counter()
        checks = run_jobs()
        fit_s = time.perf_counter() - t0
        est_full = fit_s * ROUNDS / PILOT_ROUNDS
        save_json(
            cache_dir / "pilot.json",
            {
                "status": "passed",
                "identity": identity,
                "pilot_rounds": PILOT_ROUNDS,
                "workers": args.workers,
                "strategies": len(checks),
                "release_points": sum(x["release_points"] for x in checks),
                "chronological_stages": sum(x["checked_stages"] for x in checks),
                "degenerate_draws": sum(x["degenerate_draws"] for x in checks),
                "fit_elapsed_seconds": round(fit_s, 2),
                "estimated_full_seconds": round(est_full, 1),
                "tasks": checks,
            },
        )
        print(
            f"Pilot passed: {len(checks)} strategies, {sum(x['release_points'] for x in checks)} release points, "
            f"{sum(x['checked_stages'] for x in checks)} verified stages, "
            f"deg={sum(x['degenerate_draws'] for x in checks)}; fit={fit_s:.1f}s, est_1000={est_full:.0f}s",
            flush=True,
        )
        return

    receipt = read_json(cache_dir / "pilot.json")
    assert receipt["identity"] == identity and receipt["status"] == "passed", "Run --pilot first after any code/data change."

    outputs = run_jobs()
    rel_df = pd.DataFrame([r for o in outputs for r in o["releases"]])
    rel_ext_df = pd.DataFrame([r for o in outputs for r in o["release_ext"]])
    chr_df = pd.DataFrame([r for o in outputs for r in o["chrono"]])
    chr_ext_df = pd.DataFrame([r for o in outputs for r in o["chrono_ext"]])
    step_df = pd.DataFrame([r for o in outputs for r in o["steps"]])

    results_dir.mkdir(parents=True, exist_ok=True)
    rel_df.to_csv(results_dir / "official_release_stability.csv", encoding="utf-8-sig", index=False)
    rel_ext_df.to_csv(results_dir / "official_release_external_validity.csv", encoding="utf-8-sig", index=False)
    chr_df.to_csv(results_dir / "chronological_size_stability.csv", encoding="utf-8-sig", index=False)
    chr_ext_df.to_csv(results_dir / "chronological_external_validity.csv", encoding="utf-8-sig", index=False)
    step_df.to_csv(results_dir / "chronological_step_stability.csv", encoding="utf-8-sig", index=False)

    save_json(
        results_dir / "run_summary.json",
        {
            "status": "complete",
            "identity": identity,
            "strategies": len(outputs),
            "official_release_points": len(rel_df),
            "chronological_stages": len(chr_df),
            "total_degenerate_draws": int(chr_df["degenerate_rounds"].sum()),
            "panels": list(inv["panels"]),
            "algorithms": list(reg.ALGORITHMS),
            "cluster_counts": list(reg.CLUSTER_COUNTS),
            "external_anchors": [a[0] for a in reg.EXTERNAL_ANCHORS],
            "alignment_rule": "Training-only Hungarian centroid alignment projected into 2026-08 reference standardized space",
            "tie_handling": f"{ROUNDS} nested within-snapshot permutations; earlier snapshots fixed",
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "pilot": receipt,
        },
    )
    print(
        f"Complete: {len(outputs)} strategies; {len(rel_df)} release points; "
        f"{len(chr_df)} chronological stages; {time.perf_counter() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
