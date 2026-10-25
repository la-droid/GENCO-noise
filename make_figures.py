"""Write-up figures from the per-checkpoint CSVs written by genco_noise.py.

    python make_figures.py [clean|perturbed]

Writes genco_out/<tag>/figures/: fig1_noise.png, fig2_error.png, fig3_what_tracks.png, fig4_fallback.png and
results_table.csv (the numbers behind them).
Needs only pandas, numpy and matplotlib. The four constants below must match genco_noise.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIN_SIGMA, FALSE_ALARM, TOP_SHARE, LARGE_ERROR = 0.2, 0.05, 0.10, 0.90
TAG = sys.argv[1] if len(sys.argv) > 1 else "clean"
OUT = Path("genco_out") / TAG
FIG = OUT / "figures"

INK, INK2 = "#0b0b0b", "#52514e"
DETECTORS = {"emb": ("embedding distance", "#2a78d6"),
             "vec": ("all-loads input distance", "#eb6834"),
             "pbe": ("physics residual (PBE)", "#1baf7a")}
SCORES = {"bus_prior": ("bus prior (usual error of the bus)", INK2),
          "emb_score": ("embedding distance", "#2a78d6"),
          "PBE": ("physics residual (PBE)", "#1baf7a")}
# grids: palette slots not used by the detectors (violet on the key 118-bus line, best contrast); each line also
# keeps its own dash style and marker, since yellow and magenta are light on white
GRID_STYLE = {"case14_ieee": ("-", "o", "#eda100"), "case30_ieee": ("--", "s", "#e87ba4"),
              "case118_ieee": ("-.", "^", "#6250d6")}
LABEL_BOX = dict(facecolor="#fcfcfb", edgecolor="none", alpha=0.85, pad=1.2)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})


def load(name):
    """One table over all checkpoint folders, with the checkpoint's size and seed as columns."""
    frames = []
    for d in sorted(OUT.glob("*_seed*")):
        if not (d / f"{name}.csv").exists():   # a checkpoint that has not computed this table yet
            continue
        f = pd.read_csv(d / f"{name}.csv")
        f["size"], f["seed"] = d.name.split("_", 1)
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def short(grid):
    return grid.replace("_ieee", "")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    me, det, rank = load("model_error"), load("detection"), load("error_ranking")
    grids = [g for g in GRID_STYLE if g in set(me.grid)]
    n_ckpt = me.groupby(["size", "seed"]).ngroups
    sig_max = me.noise.max()
    rng = np.random.default_rng(0)

    # ---------------- figure 1: does the model get worse, and which detector notices? ----------------
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for grid in grids:
        ls, mk, color = GRID_STYLE[grid]
        g = me[me.grid == grid].groupby("noise").ratio.agg(["mean", "min", "max"])
        a.fill_between(g.index, g["min"], g["max"], color=color, alpha=0.15, lw=0)
        a.plot(g.index, g["mean"], ls=ls, marker=mk, color=color, lw=2.2, ms=5, label=short(grid))
    a.axhline(1, color=INK, lw=0.8, ls=":")
    a.axvline(TRAIN_SIGMA, color=INK, lw=0.8, ls=":")
    a.set_xticks(sorted(me.noise.unique()))
    a.set_title("(a) Model error under load noise", loc="left", fontsize=12)
    a.set_ylabel(f"median |Va error|, relative to sigma={TRAIN_SIGMA:g}\n(1x = unchanged)")
    a.set_xlabel(f"load noise sigma (training sigma = {TRAIN_SIGMA:g})\nline = mean of {n_ckpt} checkpoints, band = min to max")
    a.legend(fontsize=9, loc="upper left")

    mean = {}
    for key, (label, color) in DETECTORS.items():
        g = det[det.detector == key].groupby("noise").flagged_pct.agg(["mean", "min", "max"])
        b.fill_between(g.index, g["min"], g["max"], color=color, alpha=0.12, lw=0)
        b.plot(g.index, g["mean"], "o-", color=color, lw=2.2, ms=5, label=label)
        mean[key] = g["mean"]

    def value_label(key, sigma, dx, dy, ha="center", text=None):
        b.annotate(text or f"{mean[key][sigma]:.0f}%", (sigma, mean[key][sigma]), xytext=(dx, dy), textcoords="offset points",
                   ha=ha, va="center", fontsize=9.5, color=INK, bbox=LABEL_BOX, zorder=5)

    first = min(x for x in mean["emb"].index if x > TRAIN_SIGMA)   # first noise level above training
    value_label("vec", first, 0, 11)                     # above the point
    value_label("emb", first, 9, -10, ha="left")         # right-below, away from both rising lines
    value_label("pbe", first, 0, 11)
    if abs(mean["vec"][sig_max] - mean["emb"][sig_max]) < 2:   # both near 100%: one shared label
        value_label("vec", sig_max, -4, 11, ha="right", text=f"both {mean['vec'][sig_max]:.0f}%")
    else:
        value_label("vec", sig_max, 0, 11)
        value_label("emb", sig_max, 0, -12)
    value_label("pbe", sig_max, 0, 11)
    b.axhline(100 * FALSE_ALARM, color=INK, lw=0.8, ls=":")
    b.axvline(TRAIN_SIGMA, color=INK, lw=0.8, ls=":")
    b.set_ylim(-3, 112)
    b.set_title("(b) Share of samples flagged as unusual", loc="left", fontsize=12)
    b.set_ylabel(f"samples flagged (%)\nalarm threshold: {100 * FALSE_ALARM:.0f}% of normal samples flagged (dotted)")
    b.set_xlabel(f"load noise sigma (training sigma = {TRAIN_SIGMA:g})\nline = mean of {len(grids)} grids x {n_ckpt} checkpoints, band = min to max")
    b.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig(FIG / "fig1_noise.png", bbox_inches="tight", dpi=170)
    plt.close()

    # ---------------- figure 2: does a score point at the buses with large error? ----------------
    per_ckpt = rank.groupby(["size", "seed", "grid", "score"])[["overall", "within_bus"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, col, scores, title, ylabel in [
            (axes[0], "overall", ["bus_prior", "emb_score", "PBE"], "(a) Rank all buses together",
             f"mean |Va error| of the top {100 * TOP_SHARE:.0f}% ranked buses,\nrelative to all buses  (1x = no better than random)"),
            (axes[1], "within_bus", ["emb_score", "PBE"], "(b) Rank each bus against its own other samples",
             f"mean |Va error| of the top {100 * TOP_SHARE:.0f}% of a bus's samples,\nrelative to that bus's mean  (1x = no better than random)")]:
        w = 0.8 / len(scores)
        for j, s in enumerate(scores):
            label, color = SCORES[s]
            xs = np.arange(len(grids)) + (j - (len(scores) - 1) / 2) * w
            vals = [per_ckpt[(per_ckpt.score == s) & (per_ckpt.grid == g)][col] for g in grids]
            ax.bar(xs, [v.mean() for v in vals], width=w * 0.92, color=color, alpha=0.85, label=label)
            for x, v in zip(xs, vals):
                ax.scatter(x + rng.uniform(-w / 6, w / 6, len(v)), v, s=14, color=INK, zorder=3)
                ax.text(x, v.max() + 0.03, f"{v.mean():.1f}x", ha="center", va="bottom", fontsize=9, color=INK)
        ax.axhline(1, color=INK, lw=0.8, ls=":")
        ax.set_xticks(range(len(grids)), [short(g) for g in grids])
        ax.set_ylim(0, per_ckpt[col].max() * 1.22)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(f"bar = mean of {n_ckpt} checkpoints, dots = each checkpoint (averaged over sigma levels)")
        ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig(FIG / "fig2_error.png", bbox_inches="tight", dpi=170)
    plt.close()

    # ---------------- figure 3: does a score follow the noise or the error? ----------------
    tracks, deciles = load("what_tracks"), load("error_by_decile")
    n_tr = tracks.groupby(["size", "seed"]).ngroups
    fig, axes = plt.subplots(1, 1 + len(grids), figsize=(17, 4.6), gridspec_kw={"width_ratios": [1.35] + [1] * len(grids)})
    ax = axes[0]
    kinds = [("corr_noise", "follows the\nnoise level"), ("corr_error", "follows the error\nat a fixed noise level")]
    for j, s in enumerate(["emb_score", "PBE"]):
        label, color = SCORES[s]
        xs = np.arange(len(kinds)) + (j - 0.5) * 0.36
        vals = [tracks[tracks.score == s][k] for k, _ in kinds]
        ax.bar(xs, [v.mean() for v in vals], width=0.34, color=color, alpha=0.85, label=label)
        for x, v in zip(xs, vals):
            ax.scatter(x + rng.uniform(-0.05, 0.05, len(v)), v, s=10, color=INK, zorder=3)
            ax.text(x, max(v.max(), 0) + 0.03, f"{v.mean():.2f}", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(len(kinds)), [k[1] for k in kinds])
    ax.set_ylim(min(-0.3, tracks[["corr_noise", "corr_error"]].min().min() - 0.05), 1.12)
    ax.set_title("(a) What each score follows", loc="left", fontsize=12)
    ax.set_ylabel("Spearman correlation inside a bus\n(0 = none, 1 = perfect)")
    ax.set_xlabel(f"bar = mean of {len(grids)} grids x {n_tr} checkpoints, dots = each")
    ax.legend(fontsize=9, loc="upper right")

    top = deciles.groupby(["grid", "score", "decile", "size", "seed"]).rel_err.mean().max()
    for ax, grid, letter in zip(axes[1:], grids, "bcd"):
        for s in ["emb_score", "PBE"]:
            label, color = SCORES[s]
            t = deciles[(deciles.grid == grid) & (deciles.score == s)].groupby("decile").rel_err.agg(["mean", "min", "max"])
            ax.fill_between(t.index, t["min"], t["max"], color=color, alpha=0.15, lw=0)
            ax.plot(t.index, t["mean"], "o-", color=color, lw=2.2, ms=4, label=label)
            ax.annotate(f"{t['mean'].iloc[-1]:.2f}x", (t.index[-1], t["mean"].iloc[-1]), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=9.5, color=INK, bbox=LABEL_BOX)
        ax.axhline(1, color=INK, lw=0.8, ls=":")
        ax.set_ylim(0.4, top * 1.12)
        ax.set_xticks(range(1, 11))
        ax.set_title(f"({letter}) {short(grid)}: error by score decile", loc="left", fontsize=12)
        ax.set_xlabel("score decile within the bus\n(1 = lowest, 10 = highest)")
    axes[1].set_ylabel("|Va error| relative to the bus's\naverage at that noise level (1x = average)")
    plt.tight_layout()
    plt.savefig(FIG / "fig3_what_tracks.png", bbox_inches="tight", dpi=170)
    plt.close()

    # ---------------- figure 4: the AC fallback - which samples to re-solve? ----------------
    fb = load("fallback")
    n_fb = fb.groupby(["size", "seed"]).ngroups
    fig, axes = plt.subplots(1, len(grids), figsize=(15.5, 4.6), sharey=True)
    for ax, grid, letter in zip(axes, grids, "abc"):
        ax.plot([0, 100], [0, 100], color=INK, lw=1, ls=":", label="random choice")
        at20 = []
        for key in ("pbe", "emb", "vec"):
            label, color = DETECTORS[key]
            t = fb[(fb.grid == grid) & (fb.score == key)].groupby("budget").caught.agg(["mean", "min", "max"]) * 100
            x = t.index * 100
            ax.fill_between(x, t["min"], t["max"], color=color, alpha=0.15, lw=0)
            ax.plot(np.r_[0, x], np.r_[0, t["mean"]], "-", color=color, lw=2.2, label=label)
            v = t.loc[0.2, "mean"]
            ax.plot(20, v, "o", color=color, ms=6, zorder=4)
            at20.append(f"{label.split(' (')[0]}: {v:.0f}%")
        ax.axvline(20, color=INK, lw=0.8, ls=":")
        ax.text(97, 4, "re-solving 20% catches\n" + "\n".join(at20) + "\nrandom choice: 20%", ha="right", va="bottom",
                fontsize=9.5, color=INK, bbox=dict(facecolor="#fcfcfb", edgecolor="#d8d7d3", pad=4))
        ax.set_xlim(0, 100); ax.set_ylim(0, 102)
        ax.set_title(f"({letter}) {short(grid)}", loc="left", fontsize=12)
        ax.set_xlabel("samples re-solved with AC (%)")
    axes[0].set_ylabel("large errors caught (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.text(0.01, -0.12, f"Large error: worst-bus |Va error| above the worst {100 * (1 - LARGE_ERROR):.0f}% at training noise. "
             f"All noise levels pooled. "
             f"Line = mean of {n_fb} checkpoints, band = min to max; points mark a 20% AC budget.", fontsize=9.5, color=INK2)
    plt.tight_layout()
    plt.savefig(FIG / "fig4_fallback.png", bbox_inches="tight", dpi=170)
    plt.close()

    # ---------------- the numbers behind the figures ----------------
    def agg(df, value, keys, metric):
        t = df.groupby(keys)[value].agg(["mean", "min", "max", "count"]).reset_index()
        t.insert(0, "metric", metric)
        return t
    table = pd.concat([
        agg(me[me.noise == sig_max], "ratio", ["grid"], f"median |Va error| at sigma={sig_max:g} / at sigma={TRAIN_SIGMA:g}"),
        agg(det[det.noise.isin([0.3, sig_max])], "flagged_pct", ["grid", "detector", "noise"], "% of samples flagged"),
        agg(per_ckpt, "overall", ["grid", "score"], f"error of top {100 * TOP_SHARE:.0f}% ranked buses vs all (x), overall"),
        agg(per_ckpt.dropna(subset=["within_bus"]), "within_bus", ["grid", "score"], "same, ranked within each bus"),
        agg(tracks, "corr_noise", ["grid", "score"], "Spearman with noise level, inside a bus"),
        agg(tracks, "corr_error", ["grid", "score"], "Spearman with |Va error| at fixed noise, inside a bus"),
        agg(deciles[deciles.decile == 10], "rel_err", ["grid", "score"], "error in the top score decile vs the bus average (x)"),
        agg(fb[fb.budget.isin([0.1, 0.2, 0.3])], "caught", ["grid", "score", "budget"], "share of large errors caught with this AC budget"),
    ], ignore_index=True)
    table.to_csv(FIG / "results_table.csv", index=False)
    print(table.round(2).to_string(index=False))
    print(f"\nWritten to {FIG.resolve()}")


if __name__ == "__main__":
    main()
