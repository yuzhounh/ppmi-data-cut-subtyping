"""Build publication-ready figures and summary tables from saved dual-trajectory results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_IN_CODE = _HERE.name == "code"
PROJECT = _HERE.parent if _IN_CODE else _HERE

DEFAULT_RESULTS = (
    PROJECT / "results" / "unified_subtyping_trajectories"
    if _IN_CODE
    else PROJECT / "outputs" / "results"
)
DEFAULT_FIGURES = (
    PROJECT / "results" / "unified_subtyping_trajectories" / "figures"
    if _IN_CODE
    else PROJECT / "outputs" / "figures"
)
DEFAULT_TABLES = (
    PROJECT / "results" / "unified_subtyping_trajectories" / "tables"
    if _IN_CODE
    else PROJECT / "outputs" / "tables"
)

PANEL_ORDER = (
    "P1_multidomain_11",
    "P2_compact_7",
    "P3_multidomain_plus_upsit_12",
)
PANEL_SHORT = {
    "P1_multidomain_11": "P1: Multi-domain (11 features)",
    "P2_compact_7": "P2: Compact (7 features)",
    "P3_multidomain_plus_upsit_12": "P3: Multi-domain + UPSIT (12 features)",
}
PANEL_TAG = {
    "P1_multidomain_11": "P1 (11f)",
    "P2_compact_7": "P2 (7f)",
    "P3_multidomain_plus_upsit_12": "P3 (12f+UPSIT)",
}
ALG_ORDER = ("kmeans", "gmm", "ward")
ALG_LABEL = {
    "kmeans": "K-means",
    "gmm": "GMM (diag)",
    "ward": "Ward hierarchical",
}
ALG_COLOR = {
    "kmeans": "#0072B2",  # Okabe-Ito blue
    "gmm": "#E69F00",     # Okabe-Ito orange
    "ward": "#D55E00",    # Okabe-Ito vermilion
}
ALG_MARKER = {
    "kmeans": "o",
    "gmm": "s",
    "ward": "^",
}
ANCHOR_META = {
    "2y_moca": ("Two-year MoCA", "pts"),
    "2y_updrs3": ("Two-year MDS-UPDRS III", "pts"),
    "2y_updrs2": ("Two-year MDS-UPDRS II", "pts"),
    "2y_adl": ("Two-year Schwab-England ADL", "pts"),
    "2y_pm_any": ("Two-year progression milestone", "pp"),
    "2y_cogstate_abn": ("Two-year cognitive impairment", "pp"),
    "2y_adl_lt80": ("Two-year ADL < 80", "pp"),
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def fmt_version(v: int | str) -> str:
    s = str(v)
    return f"{s[:4]}-{s[4:6]}"


def save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig1_release_stability_k3(rel: pd.DataFrame, out_dir: Path) -> None:
    """Main Figure 1: Official 11-release ARI and held-out flip rate trajectories at clinical primary K=3."""
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.8), sharex=True)
    versions = sorted(rel["version"].unique())
    x_pos = np.arange(len(versions))
    x_labels = [fmt_version(v) for v in versions]

    for col, panel in enumerate(PANEL_ORDER):
        ax_ari = axes[0, col]
        ax_flp = axes[1, col]
        sub = rel[(rel["panel"] == panel) & (rel["k"] == 3)]

        for alg in ALG_ORDER:
            s = sub[sub["algorithm"] == alg].sort_values("version")
            ax_ari.plot(
                x_pos,
                s["ari_to_ref"],
                color=ALG_COLOR[alg],
                marker=ALG_MARKER[alg],
                markersize=5,
                linewidth=1.9,
                label=ALG_LABEL[alg],
                zorder=3 if alg == "kmeans" else 2,
            )
            ax_flp.plot(
                x_pos,
                s["flip_rate_to_ref"] * 100.0,
                color=ALG_COLOR[alg],
                marker=ALG_MARKER[alg],
                markersize=5,
                linewidth=1.9,
                label=ALG_LABEL[alg],
                zorder=3 if alg == "kmeans" else 2,
            )

        ax_ari.axhline(0.80, color="#666666", linestyle="--", linewidth=0.9, alpha=0.8)
        ax_flp.axhline(10.0, color="#666666", linestyle="--", linewidth=0.9, alpha=0.8)

        ax_ari.set_title(PANEL_SHORT[panel], pad=8)
        ax_ari.set_ylim(-0.02, 1.06)
        ax_flp.set_ylim(-2.0, 64.0)
        ax_flp.set_xticks(x_pos)
        ax_flp.set_xticklabels(x_labels, rotation=45, ha="right")
        ax_flp.set_xlabel("Official PPMI curated data cut")

        if col == 0:
            ax_ari.set_ylabel("ARI vs. 2026-08 reference")
            ax_flp.set_ylabel("Held-out flip rate (%)")
            ax_ari.text(0.10, 0.72, "ARI = 0.80", fontsize=7.5, color="#555555")
            ax_flp.text(0.10, 11.8, "10% flip", fontsize=7.5, color="#555555")

    # Crisp non-overlapping callouts for key empirical phenomena
    axes[0, 0].annotate(
        "GMM jump (+3 pts)\nARI=0.23",
        xy=(5, 0.230),
        xytext=(3.2, 0.05),
        fontsize=7.5,
        color="#9C6B00",
        arrowprops={"arrowstyle": "->", "color": "#9C6B00", "lw": 0.9},
    )
    axes[0, 1].annotate(
        "K-means basin\nswitching",
        xy=(7, 0.332),
        xytext=(5.2, 0.06),
        fontsize=7.5,
        color="#005282",
        arrowprops={"arrowstyle": "->", "color": "#005282", "lw": 0.9},
    )
    axes[0, 2].annotate(
        "GMM 3-cut\nregime jump",
        xy=(5, 0.216),
        xytext=(1.6, 0.04),
        fontsize=7.5,
        color="#9C6B00",
        arrowprops={"arrowstyle": "->", "color": "#9C6B00", "lw": 0.9},
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, out_dir, "fig1_release_stability_k3")


def make_fig2_chronological_dissociation(
    rel: pd.DataFrame,
    rext: pd.DataFrame,
    ch: pd.DataFrame,
    cext: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Main Figure 2: Sample-size doubling vs official releases on P1 K=3 showing membership-prognosis dissociation."""
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.8), sharex=True)
    panel = "P1_multidomain_11"
    k = 3

    ch_sub = ch[(ch["panel"] == panel) & (ch["k"] == k)]
    rel_sub = rel[(rel["panel"] == panel) & (rel["k"] == k)]
    cext_sub = cext[(cext["panel"] == panel) & (cext["k"] == k)]
    rext_sub = rext[(rext["panel"] == panel) & (rext["k"] == k)]

    ticks = sorted(ch_sub["training_n"].unique())

    panels_spec = [
        (axes[0, 0], "A. Individual membership agreement (ARI vs. 2026-08)", "ARI", "ari", 1.0, (0.0, 1.06)),
        (axes[0, 1], "B. Individual patient reclassification (flip rate)", "Held-out flip rate (%)", "flip", 100.0, (-2.0, 78.0)),
        (axes[1, 0], "C. External 2-year cognitive separation (MoCA)", "Two-year MoCA variance explained (η²)", "moca", 1.0, (0.0, 0.27)),
        (axes[1, 1], "D. External 2-year progression milestone risk gap", "Severe minus Mild risk difference (pp)", "pm", 100.0, (-2.0, 72.0)),
    ]

    for ax, title, ylabel, mode, scale, ylim in panels_spec:
        for alg in ALG_ORDER:
            color = ALG_COLOR[alg]
            marker = ALG_MARKER[alg]

            if mode in ("ari", "flip"):
                s = ch_sub[ch_sub["algorithm"] == alg].sort_values("training_n")
                r = rel_sub[rel_sub["algorithm"] == alg].sort_values("training_n")
                col_m = "ari_mean" if mode == "ari" else "flip_rate_mean"
                col_lo = "ari_p025" if mode == "ari" else "flip_rate_p025"
                col_hi = "ari_p975" if mode == "ari" else "flip_rate_p975"
                col_r = "ari_to_ref" if mode == "ari" else "flip_rate_to_ref"
            elif mode == "moca":
                s = cext_sub[(cext_sub["algorithm"] == alg) & (cext_sub["anchor"] == "2y_moca")].sort_values("training_n")
                r = rext_sub[(rext_sub["algorithm"] == alg) & (rext_sub["anchor"] == "2y_moca")].sort_values("training_n")
                col_m, col_lo, col_hi, col_r = "eta2_mean", "eta2_p025", "eta2_p975", "eta2"
            else:
                s = cext_sub[(cext_sub["algorithm"] == alg) & (cext_sub["anchor"] == "2y_pm_any")].sort_values("training_n")
                r = rext_sub[(rext_sub["algorithm"] == alg) & (rext_sub["anchor"] == "2y_pm_any")].sort_values("training_n")
                col_m, col_lo, col_hi, col_r = (
                    "severe_minus_mild_mean",
                    "severe_minus_mild_p025",
                    "severe_minus_mild_p975",
                    "severe_minus_mild",
                )

            xs = s["training_n"].to_numpy(dtype=float)
            ym = s[col_m].to_numpy(dtype=float) * scale
            ylo = s[col_lo].to_numpy(dtype=float) * scale
            yhi = s[col_hi].to_numpy(dtype=float) * scale

            ax.fill_between(xs, ylo, yhi, color=color, alpha=0.14, linewidth=0)
            ax.plot(
                xs,
                ym,
                color=color,
                marker=marker,
                markersize=5,
                linewidth=1.9,
                label=f"{ALG_LABEL[alg]} (1,000-round chrono)",
                zorder=3,
            )
            # Overlay the 10 non-reference official releases as crisp open diamonds
            r_nr = r[r["version"] != 20260810]
            ax.scatter(
                r_nr["training_n"].to_numpy(dtype=float),
                r_nr[col_r].to_numpy(dtype=float) * scale,
                facecolors="white",
                edgecolors=color,
                marker="D",
                s=28,
                linewidths=1.1,
                zorder=4,
            )

        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xticklabels([str(int(t)) for t in ticks])
        ax.set_xlim(32, 1380)
        ax.set_ylim(*ylim)
        ax.set_title(title, loc="left", pad=6)
        ax.set_ylabel(ylabel)
        if mode == "ari":
            ax.axhline(0.80, color="#666666", linestyle="--", linewidth=0.85, alpha=0.8)
        elif mode == "flip":
            ax.axhline(10.0, color="#666666", linestyle="--", linewidth=0.85, alpha=0.8)

    axes[1, 0].set_xlabel("Development PD sample size (log₂ scale)")
    axes[1, 1].set_xlabel("Development PD sample size (log₂ scale)")

    # Legend with extra proxy for official release diamonds
    handles, labels = axes[0, 0].get_legend_handles_labels()
    diamond_proxy = plt.Line2D(
        [0],
        [0],
        marker="D",
        color="none",
        markerfacecolor="white",
        markeredgecolor="#333333",
        markersize=5.5,
        markeredgewidth=1.1,
        label="Official curated cuts (2023-06 to 2026-05)",
    )
    handles.append(diamond_proxy)
    labels.append("Official curated cuts (2023-06 to 2026-05)")
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, out_dir, "fig2_chronological_dissociation")


def make_fig3_matrix_and_dissociation(rel: pd.DataFrame, rext: pd.DataFrame, out_dir: Path) -> None:
    """Main Figure 3: Full 27-strategy heatmap of median ARI (max flip%) + membership-vs-prognosis dissociation scatter."""
    fig, (ax_hm, ax_sc) = plt.subplots(1, 2, figsize=(13.0, 5.6), gridspec_kw={"width_ratios": [1.05, 1.15]})

    non_ref = rel[rel["version"] != 20260810]
    row_keys = [(p, k) for p in PANEL_ORDER for k in (2, 3, 4)]
    row_labels = [f"{PANEL_TAG[p]}  K={k}" for p, k in row_keys]

    mat_ari = np.zeros((len(row_keys), len(ALG_ORDER)))
    mat_flip = np.zeros((len(row_keys), len(ALG_ORDER)))
    for i, (p, k) in enumerate(row_keys):
        for j, alg in enumerate(ALG_ORDER):
            s = non_ref[(non_ref["panel"] == p) & (non_ref["algorithm"] == alg) & (non_ref["k"] == k)]
            mat_ari[i, j] = s["ari_to_ref"].median()
            mat_flip[i, j] = s["flip_rate_to_ref"].max() * 100.0

    im = ax_hm.imshow(mat_ari, cmap="YlGnBu", vmin=0.10, vmax=0.95, aspect="auto")
    ax_hm.set_xticks(np.arange(len(ALG_ORDER)))
    ax_hm.set_xticklabels([ALG_LABEL[a] for a in ALG_ORDER], fontweight="bold")
    ax_hm.set_yticks(np.arange(len(row_labels)))
    ax_hm.set_yticklabels(row_labels)
    ax_hm.set_title("A. 10-cut median ARI (max flip %)", loc="left", pad=8)
    ax_hm.grid(False)
    for y_sep in (2.5, 5.5):
        ax_hm.axhline(y_sep, color="white", linewidth=2.2)

    for i in range(len(row_keys)):
        for j in range(len(ALG_ORDER)):
            val = mat_ari[i, j]
            flp = mat_flip[i, j]
            txt_color = "white" if val > 0.62 else "#1A1A1A"
            ax_hm.text(
                j,
                i,
                f"{val:.2f}\n({flp:.0f}%)",
                ha="center",
                va="center",
                fontsize=7.8,
                color=txt_color,
                fontweight="bold" if val >= 0.80 else "normal",
            )
    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
    cbar.set_label("Median ARI vs. 2026-08", fontsize=8.5)

    # Panel B: Dissociation scatter (Median Flip% vs 11-release Two-year progression milestone risk diff, with min-max bars)
    panel_shapes = {"P1_multidomain_11": "o", "P2_compact_7": "s", "P3_multidomain_plus_upsit_12": "^"}
    for (p, alg, k), grp in non_ref.groupby(["panel", "algorithm", "k"]):
        eg = rext[(rext["panel"] == p) & (rext["algorithm"] == alg) & (rext["k"] == k) & (rext["anchor"] == "2y_pm_any")]
        x_med = grp["flip_rate_to_ref"].median() * 100.0
        y_med = eg["severe_minus_mild"].median() * 100.0
        y_min = eg["severe_minus_mild"].min() * 100.0
        y_max = eg["severe_minus_mild"].max() * 100.0
        size = {2: 38, 3: 68, 4: 105}[k]
        ax_sc.vlines(x_med, y_min, y_max, color=ALG_COLOR[alg], alpha=0.35, linewidth=1.2, zorder=1)
        ax_sc.scatter(
            x_med,
            y_med,
            color=ALG_COLOR[alg],
            marker=panel_shapes[p],
            s=size,
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )

    ax_sc.axvline(10.0, color="#666666", linestyle="--", linewidth=0.9, alpha=0.8)
    ax_sc.axhline(0.0, color="#333333", linestyle="-", linewidth=0.7, alpha=0.6)
    ax_sc.set_xlim(0.0, 36.5)
    ax_sc.set_ylim(-2.0, 96.0)
    ax_sc.set_xlabel("10-release median held-out flip rate (%)")
    ax_sc.set_ylabel("2-year progression milestone risk gap\n(Severe minus Mild, pp; median & 11-cut range)")
    ax_sc.set_title("B. Reclassification churn vs. 2-year prognostic gap", loc="left", pad=8)

    ax_sc.text(0.8, 86.0, "Low churn (<10%)\n& steady prognosis", fontsize=7.8, color="#005282")
    ax_sc.text(11.0, 82.0, "High churn (15–33%)\nstill >0 risk gap\n(wide release swing)", fontsize=7.8, color="#A84300")

    # Clean legend for Panel B placed in the open mid-upper region (x=11..22, y=50..78)
    leg_items = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=ALG_COLOR[a], markersize=7, label=ALG_LABEL[a])
        for a in ALG_ORDER
    ] + [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555", markersize=6, label="P1 (11f)"),
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor="#555555", markersize=6, label="P2 (7f)"),
        plt.Line2D([0], [0], marker="^", color="none", markerfacecolor="#555555", markersize=6, label="P3 (12f+UPSIT)"),
    ]
    ax_sc.legend(handles=leg_items, loc="upper left", bbox_to_anchor=(0.30, 0.82), fontsize=7.6, frameon=True, framealpha=0.92)

    fig.tight_layout()
    save_fig(fig, out_dir, "fig3_algorithm_k_panel_matrix")


def make_figS1_all27_releases(rel: pd.DataFrame, out_dir: Path) -> None:
    """Supplementary Figure S1: Full 3x3 official-release ARI atlas across K=2,3,4 and panels P1/P2/P3."""
    fig, axes = plt.subplots(3, 3, figsize=(13.2, 9.4), sharex=True, sharey=True)
    versions = sorted(rel["version"].unique())
    x_pos = np.arange(len(versions))
    x_labels = [fmt_version(v) for v in versions]

    for row, k in enumerate((2, 3, 4)):
        for col, panel in enumerate(PANEL_ORDER):
            ax = axes[row, col]
            sub = rel[(rel["panel"] == panel) & (rel["k"] == k)]
            for alg in ALG_ORDER:
                s = sub[sub["algorithm"] == alg].sort_values("version")
                ax.plot(
                    x_pos,
                    s["ari_to_ref"],
                    color=ALG_COLOR[alg],
                    marker=ALG_MARKER[alg],
                    markersize=4.2,
                    linewidth=1.7,
                    label=ALG_LABEL[alg],
                )
            ax.axhline(0.80, color="#666666", linestyle="--", linewidth=0.8, alpha=0.75)
            ax.set_ylim(-0.05, 1.06)
            if row == 0:
                ax.set_title(PANEL_SHORT[panel], pad=7)
            if col == 0:
                ax.set_ylabel(f"K = {k}\nARI vs. 2026-08")
            if row == 2:
                ax.set_xticks(x_pos)
                ax.set_xticklabels(x_labels, rotation=45, ha="right")
                ax.set_xlabel("Official PPMI curated cut")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(fig, out_dir, "figS1_all_27_release_trajectories")


def make_figS2_all27_chronological(rel: pd.DataFrame, ch: pd.DataFrame, out_dir: Path) -> None:
    """Supplementary Figure S2: Full 3x3 chronological 1,000-round ARI atlas with 95% bands and official overlays."""
    fig, axes = plt.subplots(3, 3, figsize=(13.2, 9.4), sharey=True)
    for row, k in enumerate((2, 3, 4)):
        for col, panel in enumerate(PANEL_ORDER):
            ax = axes[row, col]
            sub_c = ch[(ch["panel"] == panel) & (ch["k"] == k)]
            sub_r = rel[(rel["panel"] == panel) & (rel["k"] == k) & (rel["version"] != 20260810)]
            ticks = sorted(sub_c["training_n"].unique())
            for alg in ALG_ORDER:
                s = sub_c[sub_c["algorithm"] == alg].sort_values("training_n")
                r = sub_r[sub_r["algorithm"] == alg].sort_values("training_n")
                xs = s["training_n"].to_numpy(dtype=float)
                ax.fill_between(xs, s["ari_p025"], s["ari_p975"], color=ALG_COLOR[alg], alpha=0.14, linewidth=0)
                ax.plot(
                    xs,
                    s["ari_mean"],
                    color=ALG_COLOR[alg],
                    marker=ALG_MARKER[alg],
                    markersize=4.2,
                    linewidth=1.7,
                    label=ALG_LABEL[alg],
                )
                ax.scatter(
                    r["training_n"],
                    r["ari_to_ref"],
                    facecolors="white",
                    edgecolors=ALG_COLOR[alg],
                    marker="D",
                    s=20,
                    linewidths=0.95,
                    zorder=4,
                )
            ax.axhline(0.80, color="#666666", linestyle="--", linewidth=0.8, alpha=0.75)
            ax.set_xscale("log", base=2)
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.set_xticklabels([str(int(t)) for t in ticks], fontsize=7.5)
            ax.set_xlim(32, 1380)
            ax.set_ylim(-0.05, 1.06)
            if row == 0:
                ax.set_title(PANEL_SHORT[panel], pad=7)
            if col == 0:
                ax.set_ylabel(f"K = {k}\nARI vs. 2026-08")
            if row == 2:
                ax.set_xlabel("Development PD sample size (log₂)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    diamond_proxy = plt.Line2D(
        [0],
        [0],
        marker="D",
        color="none",
        markerfacecolor="white",
        markeredgecolor="#333333",
        markersize=5,
        markeredgewidth=1.0,
        label="Official curated cuts",
    )
    handles.append(diamond_proxy)
    labels.append("Official curated cuts")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(fig, out_dir, "figS2_all_27_chronological_trajectories")


def make_figS3_external_anchors(rext: pd.DataFrame, out_dir: Path) -> None:
    """Supplementary Figure S3: All 7 external 2-year prognostic anchors across 11 official releases for P1 K=3."""
    anchors = list(ANCHOR_META.keys())
    fig, axes = plt.subplots(2, 4, figsize=(14.0, 6.8))
    axes_flat = axes.ravel()
    versions = sorted(rext["version"].unique())
    x_pos = np.arange(len(versions))
    x_labels = [fmt_version(v) for v in versions]

    sub = rext[(rext["panel"] == "P1_multidomain_11") & (rext["k"] == 3)]
    for idx, anc in enumerate(anchors):
        ax = axes_flat[idx]
        title, unit = ANCHOR_META[anc]
        scale = 100.0 if unit == "pp" else 1.0
        for alg in ALG_ORDER:
            s = sub[(sub["algorithm"] == alg) & (sub["anchor"] == anc)].sort_values("version")
            ax.plot(
                x_pos,
                s["severe_minus_mild"] * scale,
                color=ALG_COLOR[alg],
                marker=ALG_MARKER[alg],
                markersize=4.2,
                linewidth=1.7,
                label=ALG_LABEL[alg],
            )
        ax.axhline(0.0, color="#555555", linestyle="--", linewidth=0.8, alpha=0.8)
        ax.set_title(title, fontsize=9, pad=5)
        ax.set_ylabel(f"Severe − Mild ({unit})", fontsize=8)
        ax.set_xticks(x_pos)
        if idx >= 3:
            ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7.5)
        else:
            ax.set_xticklabels([])

    axes_flat[-1].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    axes_flat[-1].legend(handles, labels, loc="center", frameon=False, fontsize=10, title="Algorithm (P1, K = 3)")
    fig.tight_layout()
    save_fig(fig, out_dir, "figS3_external_anchors_release_stability")


def build_tables(
    inv: dict,
    rel: pd.DataFrame,
    rext: pd.DataFrame,
    ch: pd.DataFrame,
    cext: pd.DataFrame,
    step: pd.DataFrame,
    tables_dir: Path,
) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Table 1: Panel cohort & subgroup evolution overview
    t1_rows = []
    for p in PANEL_ORDER:
        pm = inv["panels"][p]
        rc0 = pm["release_counts"]["20230612"]
        rc1 = pm["release_counts"]["20260810"]
        s0 = rc0["subgroups"].get("Sporadic", 0) / rc0["dev_n"] * 100.0
        s1 = rc1["subgroups"].get("Sporadic", 0) / rc1["dev_n"] * 100.0
        t1_rows.append(
            {
                "panel": PANEL_TAG[p],
                "n_features": len(pm["features"]),
                "dev_202306_n": rc0["dev_n"],
                "dev_202608_n": rc1["dev_n"],
                "growth_pct": (rc1["dev_n"] / rc0["dev_n"] - 1.0) * 100.0,
                "sporadic_202306_pct": s0,
                "sporadic_202608_pct": s1,
                "verified_202306_n": pm["dev_202306_verified_n"],
                "fixed_eval_n": pm["eval_n"],
            }
        )
    t1 = pd.DataFrame(t1_rows)
    t1.to_csv(tables_dir / "table1_panel_cohort_overview.csv", index=False, encoding="utf-8-sig")

    # Table 2: All 27 strategies summary across 10 non-reference official releases + 2026-08 external validity
    non_ref = rel[rel["version"] != 20260810]
    t2_rows = []
    for p in PANEL_ORDER:
        for k in (2, 3, 4):
            for alg in ALG_ORDER:
                g = non_ref[(non_ref["panel"] == p) & (non_ref["algorithm"] == alg) & (non_ref["k"] == k)]
                g_all = rel[(rel["panel"] == p) & (rel["algorithm"] == alg) & (rel["k"] == k)]
                e_moca = rext[(rext["panel"] == p) & (rext["algorithm"] == alg) & (rext["k"] == k) & (rext["anchor"] == "2y_moca")]
                e_up2 = rext[(rext["panel"] == p) & (rext["algorithm"] == alg) & (rext["k"] == k) & (rext["anchor"] == "2y_updrs2")]
                e_pm = rext[(rext["panel"] == p) & (rext["algorithm"] == alg) & (rext["k"] == k) & (rext["anchor"] == "2y_pm_any")]
                t2_rows.append(
                    {
                        "panel": PANEL_TAG[p],
                        "k": k,
                        "algorithm": ALG_LABEL[alg],
                        "ari_202306": g.sort_values("version").iloc[0]["ari_to_ref"],
                        "ari_median": g["ari_to_ref"].median(),
                        "ari_min": g["ari_to_ref"].min(),
                        "flip_median_pct": g["flip_rate_to_ref"].median() * 100.0,
                        "flip_max_pct": g["flip_rate_to_ref"].max() * 100.0,
                        "step_ari_median": g_all["step_ari"].median(),
                        "step_ari_min": g_all["step_ari"].min(),
                        "shift_median": g["centroid_shift_l2"].median(),
                        "silhouette_median": g_all["silhouette_eval"].median(),
                        "moca_eta2_ref": e_moca[e_moca["version"] == 20260810]["eta2"].iloc[0],
                        "moca_eta2_min": e_moca["eta2"].min(),
                        "moca_eta2_max": e_moca["eta2"].max(),
                        "updrs2_eta2_ref": e_up2[e_up2["version"] == 20260810]["eta2"].iloc[0],
                        "pm_diff_ref_pp": e_pm[e_pm["version"] == 20260810]["severe_minus_mild"].iloc[0] * 100.0,
                        "pm_diff_min_pp": e_pm["severe_minus_mild"].min() * 100.0,
                        "pm_diff_max_pp": e_pm["severe_minus_mild"].max() * 100.0,
                    }
                )
    t2 = pd.DataFrame(t2_rows)
    t2.to_csv(tables_dir / "table2_official_release_summary_27.csv", index=False, encoding="utf-8-sig")

    # Table 3: Chronological doubling table for P1 K=3 (1,000 rounds)
    t3_rows = []
    for alg in ALG_ORDER:
        cs = ch[(ch["panel"] == "P1_multidomain_11") & (ch["algorithm"] == alg) & (ch["k"] == 3)].sort_values("training_n")
        st = step[(step["panel"] == "P1_multidomain_11") & (step["algorithm"] == alg) & (step["k"] == 3)].sort_values("larger_n")
        st_map = {int(r["larger_n"]): (r["step_ari_mean"], r["step_ari_p025"], r["step_ari_p975"]) for _, r in st.iterrows()}
        for _, r in cs.iterrows():
            n = int(r["training_n"])
            em = cext[(cext["panel"] == "P1_multidomain_11") & (cext["algorithm"] == alg) & (cext["k"] == 3) & (cext["training_n"] == n)]
            moca = em[em["anchor"] == "2y_moca"].iloc[0]
            up2 = em[em["anchor"] == "2y_updrs2"].iloc[0]
            pm = em[em["anchor"] == "2y_pm_any"].iloc[0]
            s_m, s_lo, s_hi = st_map.get(n, (np.nan, np.nan, np.nan))
            t3_rows.append(
                {
                    "algorithm": ALG_LABEL[alg],
                    "training_n": n,
                    "verified_cut": fmt_version(int(r["verified_cut"])),
                    "ari_mean": r["ari_mean"],
                    "ari_p025": r["ari_p025"],
                    "ari_p975": r["ari_p975"],
                    "flip_mean_pct": r["flip_rate_mean"] * 100.0,
                    "flip_p025_pct": r["flip_rate_p025"] * 100.0,
                    "flip_p975_pct": r["flip_rate_p975"] * 100.0,
                    "step_ari_mean": s_m,
                    "step_ari_p025": s_lo,
                    "step_ari_p975": s_hi,
                    "moca_eta2_mean": moca["eta2_mean"],
                    "moca_eta2_p025": moca["eta2_p025"],
                    "moca_eta2_p975": moca["eta2_p975"],
                    "updrs2_eta2_mean": up2["eta2_mean"],
                    "pm_diff_mean_pp": pm["severe_minus_mild_mean"] * 100.0,
                    "pm_diff_p025_pp": pm["severe_minus_mild_p025"] * 100.0,
                    "pm_diff_p975_pp": pm["severe_minus_mild_p975"] * 100.0,
                }
            )
    t3 = pd.DataFrame(t3_rows)
    t3.to_csv(tables_dir / "table3_chronological_p1_k3_summary.csv", index=False, encoding="utf-8-sig")
    return t1, t2, t3


def build_latex_tables(t1: pd.DataFrame, t2: pd.DataFrame, t3: pd.DataFrame, latex_dir: Path) -> None:
    """Emit publication-formatted booktabs LaTeX table bodies directly from verified summary tables."""
    latex_dir.mkdir(parents=True, exist_ok=True)
    alg_short = {"K-means": "K-means", "GMM (diag)": "GMM", "Ward hierarchical": "Ward"}
    pan_short = {"P1 (11f)": "P1 (11f)", "P2 (7f)": "P2 (7f)", "P3 (12f+UPSIT)": "P3 (+UPSIT)"}

    # Table 1: Panel cohort overview
    lines = [
        r"{\ManuscriptTableSize{8.5}{10.5}\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lrrrrrrr@{}}",
        r"\toprule",
        r"\ManuscriptHeader{Panel} & \ManuscriptHeader{Feat.} & \ManuscriptHeader{2023-06 $n$} & \ManuscriptHeader{2026-08 $n$} & \ManuscriptHeader{Growth} & \ManuscriptHeader{Sporadic (\%)} & \ManuscriptHeader{Verif. 2023-06} & \ManuscriptHeader{Eval $n$} \\",
        r"\midrule",
    ]
    for _, r in t1.iterrows():
        lines.append(
            f"{pan_short[r['panel']]} & {int(r['n_features'])} & {int(r['dev_202306_n']):,} & {int(r['dev_202608_n']):,} & "
            f"+{r['growth_pct']:.1f}\\% & {r['sporadic_202306_pct']:.1f}$\\rightarrow${r['sporadic_202608_pct']:.1f} & "
            f"{int(r['verified_202306_n']):,} & {int(r['fixed_eval_n']):,} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}}"])
    (latex_dir / "table1_cohort.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Table 2: Primary K=3 release stability & 2-year external validity summary (9 strategies)
    sub_k3 = t2[t2["k"] == 3]
    lines = [
        r"{\ManuscriptTableSize{7.6}{9.6}\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabular}{@{}llrrrrrr@{}}",
        r"\toprule",
        r"\ManuscriptHeader{Panel} & \ManuscriptHeader{Alg.} & \ManuscriptHeader{ARI med [min]} & \ManuscriptHeader{Flip \% [max]} & \ManuscriptHeader{Step} & \ManuscriptHeader{Shift} & \ManuscriptHeader{MoCA $\eta^2$ (range)} & \ManuscriptHeader{PM, pp (range)} \\",
        r"\midrule",
    ]
    for idx, (_, r) in enumerate(sub_k3.iterrows()):
        if idx > 0 and idx % 3 == 0:
            lines.append(r"\addlinespace[4pt]")
        lines.append(
            f"{pan_short[r['panel']]} & {alg_short[r['algorithm']]} & {r['ari_median']:.3f} [{r['ari_min']:.3f}] & "
            f"{r['flip_median_pct']:.1f} [{r['flip_max_pct']:.1f}] & {r['step_ari_median']:.3f} & "
            f"{r['shift_median']:.3f} & {r['moca_eta2_ref']:.3f} ({r['moca_eta2_min']:.3f}--{r['moca_eta2_max']:.3f}) & "
            f"+{r['pm_diff_ref_pp']:.1f} (+{r['pm_diff_min_pp']:.1f}--+{r['pm_diff_max_pp']:.1f}) \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}}"])
    (latex_dir / "table2_k3_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Table S1: Complete 27-strategy official-release summary (longtable)
    lines = [
        r"{\ManuscriptTableSize{8}{10}\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{@{}lclrrrrrrr@{}}",
        r"\caption{\textbf{Complete 27-strategy summary across 10 non-reference official PPMI releases (June 2023 to May 2026) and August 2026 two-year external anchors.}}\label{tab:s1_all27} \\",
        r"\toprule",
        r"\ManuscriptHeader{Panel} & \ManuscriptHeader{$K$} & \ManuscriptHeader{Alg.} & \ManuscriptHeader{ARI med [min]} & \ManuscriptHeader{Flip \% [max]} & \ManuscriptHeader{Step} & \ManuscriptHeader{Shift} & \ManuscriptHeader{Sil.} & \ManuscriptHeader{MoCA $\eta^2$} & \ManuscriptHeader{PM (pp)} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\ManuscriptHeader{Panel} & \ManuscriptHeader{$K$} & \ManuscriptHeader{Alg.} & \ManuscriptHeader{ARI med [min]} & \ManuscriptHeader{Flip \% [max]} & \ManuscriptHeader{Step} & \ManuscriptHeader{Shift} & \ManuscriptHeader{Sil.} & \ManuscriptHeader{MoCA $\eta^2$} & \ManuscriptHeader{PM (pp)} \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for idx, (_, r) in enumerate(t2.iterrows()):
        if idx > 0 and idx % 3 == 0:
            lines.append(r"\addlinespace[3pt]")
        lines.append(
            f"{pan_short[r['panel']]} & {int(r['k'])} & {alg_short[r['algorithm']]} & {r['ari_median']:.3f} [{r['ari_min']:.3f}] & "
            f"{r['flip_median_pct']:.1f} [{r['flip_max_pct']:.1f}] & {r['step_ari_median']:.3f} & "
            f"{r['shift_median']:.3f} & {r['silhouette_median']:.3f} & {r['moca_eta2_ref']:.3f} & "
            f"+{r['pm_diff_ref_pp']:.1f} \\\\"
        )
    lines.extend([r"\end{longtable}}"])
    (latex_dir / "tableS1_all27.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Table S2: Chronological 1,000-round doubling table for P1 K=3
    lines = [
        r"{\ManuscriptTableSize{7.8}{9.8}\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lrcrrrrr@{}}",
        r"\toprule",
        r"\ManuscriptHeader{Alg.} & \ManuscriptHeader{Dev $n$} & \ManuscriptHeader{Cut} & \ManuscriptHeader{ARI (95\%)} & \ManuscriptHeader{Flip \% (95\%)} & \ManuscriptHeader{Step} & \ManuscriptHeader{MoCA $\eta^2$ (95\%)} & \ManuscriptHeader{PM gap, pp (95\%)} \\",
        r"\midrule",
    ]
    for idx, (_, r) in enumerate(t3.iterrows()):
        if idx > 0 and idx % 6 == 0:
            lines.append(r"\addlinespace[4pt]")
        step_str = "---" if pd.isna(r["step_ari_mean"]) else f"{r['step_ari_mean']:.3f}"
        pm_lo = f"{r['pm_diff_p025_pp']:+.1f}"
        pm_hi = f"{r['pm_diff_p975_pp']:+.1f}"
        lines.append(
            f"{alg_short[r['algorithm']]} & {int(r['training_n']):,} & {r['verified_cut']} & "
            f"{r['ari_mean']:.3f} ({r['ari_p025']:.3f}--{r['ari_p975']:.3f}) & "
            f"{r['flip_mean_pct']:.1f} ({r['flip_p025_pct']:.1f}--{r['flip_p975_pct']:.1f}) & {step_str} & "
            f"{r['moca_eta2_mean']:.3f} ({r['moca_eta2_p025']:.3f}--{r['moca_eta2_p975']:.3f}) & "
            f"{r['pm_diff_mean_pp']:+.1f} ({pm_lo}, {pm_hi}) \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}}"])
    (latex_dir / "tableS2_chronological_p1_k3.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--latex-tables-dir", type=Path, default=None)
    args = parser.parse_args()

    res = args.results_dir.resolve()
    fig_dir = args.figures_dir.resolve()
    tab_dir = args.tables_dir.resolve()

    inv = json.loads((res / "panel_inventory.json").read_text(encoding="utf-8"))
    rel = pd.read_csv(res / "official_release_stability.csv")
    rext = pd.read_csv(res / "official_release_external_validity.csv")
    ch = pd.read_csv(res / "chronological_size_stability.csv")
    cext = pd.read_csv(res / "chronological_external_validity.csv")
    step = pd.read_csv(res / "chronological_step_stability.csv")

    setup_style()
    make_fig1_release_stability_k3(rel, fig_dir)
    make_fig2_chronological_dissociation(rel, rext, ch, cext, fig_dir)
    make_fig3_matrix_and_dissociation(rel, rext, fig_dir)
    make_figS1_all27_releases(rel, fig_dir)
    make_figS2_all27_chronological(rel, ch, fig_dir)
    make_figS3_external_anchors(rext, fig_dir)
    t1, t2, t3 = build_tables(inv, rel, rext, ch, cext, step, tab_dir)
    if args.latex_tables_dir is not None:
        build_latex_tables(t1, t2, t3, args.latex_tables_dir.resolve())

    print(f"Built 6 figures in {fig_dir} and 3 tables in {tab_dir}", flush=True)


if __name__ == "__main__":
    main()
