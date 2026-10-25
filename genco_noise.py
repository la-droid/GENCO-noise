"""GENCO under load noise: when is the model wrong, and which signals warn about it?

Runs the pretrained GENCO power-flow checkpoints (HF gridfm/genco-pf-datakit, loaded as in the gridfm
GENCO explorer notebook) on datakit-generated data with increasing per-bus load noise, and writes six
small CSVs per checkpoint to genco_out/<clean|perturbed>/<size>_<seed>/:

    model_error.csv      median |Va error| per grid and noise level
    detection.csv        share of samples each detector flags as unusual
    error_ranking.csv    how much larger the error is on the buses each score ranks as most suspicious
    what_tracks.csv      whether each score follows the noise level or the error
    error_by_decile.csv  error per score decile, each bus compared with itself
    fallback.csv         share of large-error samples caught when a budget of samples is re-solved with AC

Finished tables are never overwritten; delete a checkpoint folder to recompute it. Then `python make_figures.py`
builds the write-up figures.

    conda activate genco && python genco_noise.py        (needs gridfm-graphkit, gridfm-datakit + Julia setup)
"""
import shutil, subprocess, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import lightning as L
from huggingface_hub import list_repo_files, hf_hub_download
from sklearn.covariance import LedoitWolf

from gridfm_graphkit.io.config_version import resolve_config
from gridfm_graphkit.io.param_handler import NestedNamespace, get_task
from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
from gridfm_graphkit.cli import _normalize_loaded_state_dict_keys

warnings.filterwarnings("ignore")

# ---------------- knobs you might change ----------------
GRIDS = ["case14_ieee", "case30_ieee", "case118_ieee"]
SIGMAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]   # per-bus load-noise strength (datakit load.sigma)
TRAIN_SIGMA = 0.2                # noise level the checkpoints were trained at; anchor for every comparison
PERTURB_GRID = False             # False: only the load noise varies. True: also random N-1 outages and
                                 # +-20% line-parameter noise per sample, as in the checkpoints' training data
N_LOAD_SCENARIOS = 300           # samples per (grid, sigma) when PERTURB_GRID is False
N_TOPOLOGY_VARIANTS = 2          # outage variants per load scenario (only used if PERTURB_GRID)
N_REF_LOAD_SCENARIOS = 600       # reference set at TRAIN_SIGMA that the detectors are fitted on
CKPT_SIZES = ["tiny", "small", "base"]
CKPT_SEEDS = ["seed0", "seed1", "seed42"]
FALSE_ALARM = 0.05               # a detector's alarm threshold flags this share of normal (TRAIN_SIGMA) samples
TOP_SHARE = 0.10                 # "most suspicious" = this share of buses with the highest score
LARGE_ERROR = 0.90               # fallback test: a sample's error is "large" above this quantile at TRAIN_SIGMA
SEED, REF_SEED = 0, 1            # data-generation seeds (test sets / reference set)
DATA_DIR, CKPT_DIR = Path("genco_data"), Path("genco_ckpt")
HF_REPO = "gridfm/genco-pf-datakit"
# --------------------------------------------------------

RUN_TAG = "perturbed" if PERTURB_GRID else "clean"
OUT_DIR = Path("genco_out") / RUN_TAG
ACCEL = "gpu" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# ======================= checkpoints (as in the gridfm GENCO explorer notebook) =======================

def list_available_runs():
    files = list_repo_files(HF_REPO)
    runs = sorted({str(Path(f).parent) for f in files
                   if f.endswith("best_model_state_dict.pt") and not f.startswith("mlflow/")})
    return files, pd.DataFrame([r.split("/")[:3] for r in runs], columns=["grid", "size", "seed"])


def download_run(files, avail, grid, size, ckpt_seed):
    """Download weights, config and normalizer stats of one published run."""
    run = avail[(avail.grid == grid) & (avail["size"] == size) & (avail.seed == ckpt_seed)]
    assert len(run), f"No {size}/{ckpt_seed} checkpoint for {grid}"
    run_dir = "/".join(run.iloc[0][["grid", "size", "seed"]])
    want = [f for f in files if f.startswith(run_dir + "/") and f.endswith((".yaml", ".pt")) and not f.endswith("last.pt")]
    return {Path(f).name: Path(hf_hub_download(HF_REPO, f, local_dir=CKPT_DIR)) for f in want}


def load_genco(local, grid, data_dir, n_samples):
    """Build the datamodule + GENCO task from a downloaded run, evaluating on `data_dir`."""
    cfg_path = next(p for k, p in local.items() if k.endswith(".yaml"))
    cfg = resolve_config(yaml.safe_load(cfg_path.read_text()), config_path=str(cfg_path), conversion_mode="auto_inline")
    cfg["data"]["networks"] = [grid]
    cfg["data"]["scenarios"] = [int(n_samples)]
    cfg["data"]["val_ratio"], cfg["data"]["test_ratio"] = 0.1, 0.8     # not training: most samples go to test
    cfg["data"]["workers"] = 0
    cfg["data"].pop("split_from_existing_files", None)
    cfg["training"].update(accelerator=ACCEL, devices=1, strategy="auto", batch_size=64)

    args = NestedNamespace(**cfg)
    args.get_embeddings = True       # predict_step then also returns the bus embeddings, keyed by (scenario, bus)
    dm = LitGridHeteroDataModule(args, str(data_dir), normalizer_stats_path=str(local["normalizer_stats.pt"]))
    dm.setup("predict")

    task = get_task(args, dm.data_normalizers)
    state = _normalize_loaded_state_dict_keys(torch.load(local["best_model_state_dict.pt"], map_location="cpu"))
    task.load_state_dict(state, strict=True)
    task.eval()
    return task, dm


# ======================= data (gridfm-datakit) =======================

def data_dir(sigma=None):
    """Data folder of one noise level; sigma=None is the reference set."""
    if sigma is None:
        return DATA_DIR / f"reference_sigma_{TRAIN_SIGMA}_n{N_REF_LOAD_SCENARIOS}_{RUN_TAG}"
    return DATA_DIR / f"sigma_{sigma}_n{N_LOAD_SCENARIOS}_{RUN_TAG}"


def generate_data(grid, sigma, out_dir, n_load_scenarios, seed):
    """AC-solved samples whose per-bus loads carry noise of strength `sigma` (cached on disk)."""
    raw = out_dir / grid / "raw"
    if (raw / "bus_data.parquet").exists():
        return raw
    cfg = {
        "network": {"name": grid, "source": "pglib"},
        "load": {"generator": "agg_load_profile", "agg_profile": "default", "scenarios": n_load_scenarios,
                 "sigma": float(sigma), "change_reactive_power": True, "global_range": 0.4,
                 "max_scaling_factor": 4.0, "step_size": 0.1, "start_scaling_factor": 1.0},
        "topology_perturbation": ({"type": "random", "k": 1, "n_topology_variants": N_TOPOLOGY_VARIANTS,
                                   "elements": ["branch", "gen"]} if PERTURB_GRID else {"type": "none"}),
        "generation_perturbation": {"type": "none"},
        "admittance_perturbation": {"type": "random_perturbation", "sigma": 0.2} if PERTURB_GRID else {"type": "none"},
        "settings": {"num_processes": 4, "data_dir": str(out_dir), "large_chunk_size": 50, "overwrite": True,
                     "mode": "pf", "include_dc_res": False, "enable_solver_logs": False, "pf_fast": True,
                     "dcpf_fast": True,  # datakit requires it even without DC results
                     "max_iter": 200, "seed": seed},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / f"datagen_{grid}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    exe = shutil.which("gridfm_datakit")
    assert exe, "gridfm_datakit CLI not found - is the genco env activated?"
    t0 = time.time()
    subprocess.run([exe, "generate", str(cfg_path)], check=True)
    print(f"Generated {grid} sigma={sigma} in {time.time() - t0:.0f}s")
    return raw


# ======================= prediction and scores =======================

def predict_on(local, grid, sigma, n_load_scenarios, seed):
    """Run GENCO on one dataset. Returns the per-bus table and the (row-aligned) bus embeddings."""
    out_dir = data_dir(sigma)
    raw = generate_data(grid, sigma if sigma is not None else TRAIN_SIGMA, out_dir, n_load_scenarios, seed)
    n_samples = pd.read_parquet(raw / "bus_data.parquet", columns=["scenario"]).scenario.nunique()
    task, dm = load_genco(local, grid, out_dir, n_samples)

    trainer = L.Trainer(accelerator=ACCEL, devices=1, logger=False, enable_progress_bar=False, enable_model_summary=False)
    preds = trainer.predict(task, datamodule=dm)
    pb = pd.DataFrame({c: np.concatenate([p["bus"][c] for p in preds]) for c in preds[0]["bus"]})
    emb = pd.DataFrame({c: np.concatenate([p["bus_embeddings"][c] for p in preds]) for c in preds[0]["bus_embeddings"]})
    assert (pb.scenario.values == emb.scenario.values).all() and (pb.bus.values == emb.bus.values).all()
    # Va error only where Va is unknown (every bus except the slack bus); Va is in radians in graphkit's tables
    pb["va_err_deg"] = np.rad2deg(np.abs(pb.Va_pred - pb.Va_target)).where(pb.REF != 1)
    return pb, emb.filter(like="emb_").to_numpy()


class BusMahalanobis:
    """One Gaussian per bus id, fitted on reference vectors; score = Mahalanobis distance to it.

    Ledoit-Wolf shrinkage keeps the covariance usable when a bus has fewer reference samples than dimensions.
    """

    def fit(self, X, bus):
        self.mean, self.prec = {}, {}
        for b in np.unique(bus):
            lw = LedoitWolf().fit(X[bus == b])
            self.mean[b], self.prec[b] = lw.location_, np.linalg.pinv(lw.covariance_)
        return self

    def score(self, X, bus):
        out = np.full(len(X), np.nan)
        for b in np.unique(bus):
            sel = bus == b
            d = X[sel] - self.mean[b]
            out[sel] = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, self.prec[b], d), 0))
        return out


def load_vectors(grid, sigma=None):
    """One row per scenario: Pd and Qd of every bus, straight from the generated data."""
    b = pd.read_parquet(data_dir(sigma) / grid / "raw" / "bus_data.parquet", columns=["scenario", "bus", "Pd", "Qd"])
    return pd.concat([b.pivot(index="scenario", columns="bus", values=c).add_prefix(c) for c in ("Pd", "Qd")], axis=1)


def load_distance(grid):
    """Mahalanobis distance of a sample's whole load vector (Pd, Qd of every bus) to the reference set:
    the input-only baseline, fitted once per grid. Returns score(sigma, scenarios)."""
    ref = load_vectors(grid)
    keep = ref.columns[ref.std() > 1e-9]                      # buses without load carry no information
    lw = LedoitWolf().fit(ref[keep].to_numpy())
    prec = np.linalg.pinv(lw.covariance_)

    def score(sigma, scenarios):
        diff = load_vectors(grid, sigma).loc[scenarios, keep].to_numpy() - lw.location_
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", diff, prec, diff), 0))
    return score


# ======================= the result tables =======================

def model_error(bus_df):
    """Median |Va error| per grid and sigma, and its ratio to the value at TRAIN_SIGMA."""
    g = bus_df.groupby(["grid", "noise"]).agg(median_va_err_deg=("va_err_deg", "median"),
                                              n_samples=("scenario", "nunique")).reset_index()
    base = g[g.noise == TRAIN_SIGMA].set_index("grid").median_va_err_deg
    g["ratio"] = g.median_va_err_deg / g.grid.map(base)
    return g


def detection(bus_df):
    """Share of samples each detector flags at each sigma; threshold flags FALSE_ALARM of the TRAIN_SIGMA samples.

    emb: mean embedding distance over the buses of a sample. pbe: mean PBE. vec: the all-loads input distance, the
    baseline the embedding has to beat, since the embedding is computed from the same inputs.
    """
    rows = []
    for grid, d in bus_df.groupby("grid"):
        vec = load_distance(grid)
        scores = []
        for sigma, x in d.groupby("noise"):
            s = x.groupby("scenario").agg(emb=("emb_score", "mean"), pbe=("PBE", "mean"))
            s["vec"] = vec(sigma, s.index)
            s["noise"] = sigma
            scores.append(s)
        s = pd.concat(scores)
        for det in ("emb", "vec", "pbe"):
            thr = s.loc[s.noise == TRAIN_SIGMA, det].quantile(1 - FALSE_ALARM)
            for sigma, x in s.groupby("noise"):
                rows.append({"grid": grid, "detector": det, "noise": sigma, "flagged_pct": 100 * (x[det] > thr).mean()})
    return pd.DataFrame(rows)


def error_ranking(bus_df):
    """Mean |Va error| of the TOP_SHARE highest-scored buses relative to all buses (1x = no better than random).

    `overall` ranks all buses of a sigma level together. `within_bus` ranks each bus against its own other samples
    and compares with that bus's own mean error, so bus identity cannot help any more. `bus_prior` is the static
    baseline: rank buses by their usual error at TRAIN_SIGMA (cross-fitted by scenario parity at that sigma).
    """
    rows = []
    for grid, d in bus_df.groupby("grid"):
        d = d.dropna(subset=["va_err_deg"]).copy()
        at = d[d.noise == TRAIN_SIGMA]
        d["bus_prior"] = d.bus.map(at.groupby("bus").va_err_deg.mean())
        for parity in (0, 1):
            fit = at[at.scenario % 2 == 1 - parity].groupby("bus").va_err_deg.mean()
            m = (d.noise == TRAIN_SIGMA) & (d.scenario % 2 == parity)
            d.loc[m, "bus_prior"] = d.bus[m].map(fit)
        for sigma, x in d.groupby("noise"):
            bus_mean = x.groupby("bus").va_err_deg.transform("mean")
            for score in ("bus_prior", "emb_score", "PBE"):
                top = x[score].rank(method="first", pct=True) >= 1 - TOP_SHARE
                row = {"grid": grid, "noise": sigma, "score": score, "overall": x.va_err_deg[top].mean() / x.va_err_deg.mean()}
                if score != "bus_prior":   # constant per bus, so there is nothing to rank within a bus
                    top_w = x.groupby("bus")[score].rank(method="first", pct=True) >= 1 - TOP_SHARE
                    row["within_bus"] = (x.va_err_deg / bus_mean)[top_w].mean()
                rows.append(row)
    return pd.DataFrame(rows)


def what_tracks(bus_df):
    """Does a score follow the noise level or the error? Spearman correlations inside each bus: with the noise
    level (all samples of the bus), and with the error (samples of the bus at one noise level), then averaged."""
    rows = []
    d = bus_df.dropna(subset=["va_err_deg"])
    for grid, g in d.groupby("grid"):
        for score in ("emb_score", "PBE"):
            rows.append({"grid": grid, "score": score,
                         "corr_noise": np.nanmean([x[score].corr(x.noise, method="spearman") for _, x in g.groupby("bus")]),
                         "corr_error": np.nanmean([x[score].corr(x.va_err_deg, method="spearman")
                                                   for _, x in g.groupby(["bus", "noise"])])})
    return pd.DataFrame(rows)


def error_by_decile(bus_df):
    """Mean error per score decile, with each bus compared with itself at the same noise level (1 = its average)."""
    d = bus_df.dropna(subset=["va_err_deg"]).copy()
    grp = d.groupby(["grid", "bus", "noise"])
    d["rel_err"] = d.va_err_deg / grp.va_err_deg.transform("mean")
    out = []
    for score in ("emb_score", "PBE"):
        decile = (grp[score].rank(method="first", pct=True) * 10).clip(upper=9.999).astype(int) + 1   # 1..10
        t = d.assign(decile=decile).groupby(["grid", "decile"]).rel_err.mean().reset_index()
        t["score"] = score
        out.append(t)
    return pd.concat(out, ignore_index=True)


def fallback(bus_df):
    """The AC fallback the GENCO paper proposes: re-solve the samples a score ranks highest with an AC solver.

    Per sample: score = max PBE over buses, mean embedding distance, or the all-loads input distance; error =
    worst-bus |Va error|. A sample has a large error if it exceeds the LARGE_ERROR quantile of the TRAIN_SIGMA
    samples. All noise levels are pooled, as an operator would see them. For each AC budget (share of samples
    re-solved) the table gives the share of large-error samples caught; random picking catches exactly the budget.
    """
    d = bus_df.dropna(subset=["va_err_deg"])
    s = d.groupby(["grid", "noise", "scenario"]).agg(pbe=("PBE", "max"), emb=("emb_score", "mean"),
                                                     err=("va_err_deg", "max")).reset_index()
    rows = []
    for grid, g in s.groupby("grid"):
        vec = load_distance(grid)
        g = g.copy()
        for sigma, x in g.groupby("noise"):
            g.loc[x.index, "vec"] = vec(sigma, x.scenario.values)
        large = g.err > g.loc[g.noise == TRAIN_SIGMA, "err"].quantile(LARGE_ERROR)
        for score in ("pbe", "emb", "vec"):
            order = g[score].rank(method="first", ascending=False, pct=True)   # 0..1, highest score first
            for budget in np.round(np.arange(0.05, 1.001, 0.05), 2):
                rows.append({"grid": grid, "score": score, "budget": budget, "share_large": large.mean(),
                             "caught": (large & (order <= budget)).sum() / large.sum()})
    return pd.DataFrame(rows)


RESULTS = {"model_error": model_error, "detection": detection, "error_ranking": error_ranking,
           "what_tracks": what_tracks, "error_by_decile": error_by_decile, "fallback": fallback}


# ======================= pipeline =======================

def run_checkpoint(files, avail, size, ckpt_seed, out, tables_to_write):
    L.seed_everything(SEED, verbose=False)
    tables = []
    for grid in GRIDS:
        local = download_run(files, avail, grid, size, ckpt_seed)
        ref_pb, ref_E = predict_on(local, grid, None, N_REF_LOAD_SCENARIOS, REF_SEED)
        emb_ref = BusMahalanobis().fit(ref_E, ref_pb.bus.values)
        for sigma in SIGMAS:
            pb, E = predict_on(local, grid, sigma, N_LOAD_SCENARIOS, SEED)
            pb["emb_score"] = emb_ref.score(E, pb.bus.values)
            pb["grid"], pb["noise"] = grid, sigma
            tables.append(pb[["grid", "noise", "scenario", "bus", "va_err_deg", "emb_score", "PBE"]])
    bus_df = pd.concat(tables, ignore_index=True)

    out.mkdir(parents=True, exist_ok=True)
    for name in tables_to_write:          # existing results are never overwritten
        RESULTS[name](bus_df).to_csv(out / f"{name}.csv", index=False)


def main():
    files, avail = list_available_runs()
    for size in CKPT_SIZES:
        for ckpt_seed in CKPT_SEEDS:
            out = OUT_DIR / f"{size}_{ckpt_seed}"
            missing = [name for name in RESULTS if not (out / f"{name}.csv").exists()]
            if not missing:
                print(f"{size} {ckpt_seed}: done already ({out}) - delete that folder to recompute.")
                continue
            print(f"===== checkpoint {size} {ckpt_seed}: computing {', '.join(missing)} =====")
            run_checkpoint(files, avail, size, ckpt_seed, out, missing)


if __name__ == "__main__":
    main()
