"""Build training features from TRAIN candidates, fit the LightGBM matcher,
tune the decision rule on a held-out group and save everything to models/.

    python -m src.matching.train_model                 # token table + features + model + threshold
    python -m src.matching.train_model --reuse-features

Train Source-1 entities are split by (country, state, city) into
    fit  model training (a sample)   es  early stopping   thr  decision-rule tuning and reported score

Test has ~5.8 S2/S3 records per S1 entity vs ~4.7 in train (its Source 1 looks like
a ~19% smaller sample of entities, the rest of whose records stay as decoys). To
train and validate under the same density, a random 19% of train S1 entities
is hidden: their candidate rows are dropped before any feature is computed and
they are excluded from evaluation, while their S2/S3 records remain.
Feature matrices are written to disk and streamed into LightGBM (lgb.Sequence),
so training fits in 8 GB of RAM.

Model: LightGBM (MIT licence) gradient-boosted trees, far below the 8B-parameter limit.
"""
from __future__ import annotations

import argparse
import gc
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from ..blocking.candidate_generator import candidates_path, countries, load_country
from ..evaluation.validation import assign_groups
from ..preprocessing.io import ROOT, ground_truth_pairs, read_ground_truth
from .feature_engineering import (FEATURES, RECORD_COLUMNS, competition, context_features,
                                  record_columns, string_features)
from .predict import word_stats
from .threshold_tuning import tune

FEATURE_DIR = ROOT / "data" / "features"
MODEL_DIR = ROOT / "models"
FIT_ENTITIES = 700_000          # sampled 'fit' entities (all their candidates)
HIDE_FRAC = 0.19                # train S1 entities hidden to match the test's S1 density
TRAIN_S1_TOTAL = 2_206_821
SEED = 7
CATEGORICAL = ["house_rel", "script_b"]

LGB_PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=255, min_child_samples=100,
    feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    max_bin=255, verbose=-1, num_threads=7, seed=SEED,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _id_number(ids) -> np.ndarray:
    """'S2-681193310' -> 681193310 (vectorised, no Python strings)."""
    arr = pa.array(ids) if not isinstance(ids, (pa.Array, pa.ChunkedArray)) else ids
    return pc.cast(pc.utf8_slice_codeunits(arr, 3), pa.int64()).to_numpy(zero_copy_only=False)


def pair_keys(s1_ids, cand_ids) -> np.ndarray:
    """Unique int64 per (S1 id, S2/S3 id) pair: 30 bits S1 number | 30 bits candidate | 1 bit S3."""
    cand = pa.array(cand_ids) if not isinstance(cand_ids, (pa.Array, pa.ChunkedArray)) else cand_ids
    is_s3 = pc.starts_with(cand, "S3").to_numpy(zero_copy_only=False).astype(np.int64)
    return (_id_number(s1_ids) << 31) | (_id_number(cand) << 1) | is_s3


def _take(col: pd.Series, idx: np.ndarray) -> pa.Array:
    arr = col.array._pa_array if hasattr(col.array, "_pa_array") else pa.array(col.to_numpy())
    return arr.take(pa.array(idx)).combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr.take(pa.array(idx))


def _true_keys() -> np.ndarray:
    gt = ground_truth_pairs(read_ground_truth())
    return np.sort(pair_keys(gt["source1_entity_id"].to_numpy(), gt["entity_id"].to_numpy()))


def _hide(s1: pd.DataFrame, cands: pd.DataFrame, rng: np.random.Generator):
    """Drop HIDE_FRAC of the S1 entities: returns visible s1, re-indexed candidates, hidden ids."""
    hidden = rng.random(len(s1)) < HIDE_FRAC
    new_pos = np.cumsum(~hidden) - 1
    keep = ~hidden[cands["a"].to_numpy()]
    cands = cands[keep].reset_index(drop=True)
    cands["a"] = new_pos[cands["a"].to_numpy()].astype(np.int32)
    return s1[~hidden].reset_index(drop=True), cands, set(s1["entity_id"].to_numpy()[hidden])


def _entity_roles(s1: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Per S1 row: 'fit' (sampled), 'es', 'thr' or '' (unused fit-group entity)."""
    groups = assign_groups(s1)
    roles = np.where(groups == "fit", "", groups).astype(object)
    fit_ents = rng.permutation(np.flatnonzero(groups == "fit"))
    roles[fit_ents[:int(FIT_ENTITIES * len(s1) / TRAIN_S1_TOTAL)]] = "fit"
    return roles


def build_features() -> None:
    """Features for the fit / es / thr pairs of every country -> float16 .npy files (+ meta parquet)."""
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    true_k = _true_keys()
    ws = word_stats("train")
    hidden_all: set = set()
    for country in countries("train"):
        t0 = time.time()
        s1 = load_country("train", (1,), country, RECORD_COLUMNS)
        other = load_country("train", (2, 3), country, RECORD_COLUMNS)
        cands = pd.read_parquet(candidates_path("train", country))
        s1, cands, hidden = _hide(s1, cands, np.random.default_rng(SEED))
        hidden_all |= hidden
        roles = _entity_roles(s1, np.random.default_rng(SEED))
        r = roles[cands["a"].to_numpy()]
        sel = np.flatnonzero(r != "")
        ctx = context_features(cands, competition(cands, s1, other), sel)
        role = r[sel]
        a_idx, b_idx = cands["a"].to_numpy()[sel], cands["b"].to_numpy()[sel]
        s1_ids = _take(cands["s1_id"], sel)          # Arrow gather: no 49M Python strings
        cand_ids = _take(cands["cand_id"], sel)
        labels = np.isin(pair_keys(s1_ids, cand_ids), true_k).astype(np.int8)
        cols_a, cols_b = record_columns(s1), record_columns(other)
        del s1, other, cands
        gc.collect()
        for name in ("fit", "es", "thr"):
            m = role == name
            strf = string_features(a_idx[m], b_idx[m], cols_a, cols_b, ws=ws)
            X = np.hstack([strf, ctx[m]]).astype(np.float16)     # float16 on disk (8 GB laptops)
            np.save(FEATURE_DIR / f"X_{name}_{country}.npy", X)
            meta = pd.DataFrame({"label": labels[m]})
            if name != "fit":
                idx = pa.array(np.flatnonzero(m))
                meta["s1_id"] = s1_ids.take(idx).to_pandas()
                meta["cand_id"] = cand_ids.take(idx).to_pandas()
            meta.to_parquet(FEATURE_DIR / f"meta_{name}_{country}.parquet", index=False)
            log(f"features {country}/{name}: {int(m.sum()):,} pairs ({labels[m].mean():.3f} positive)")
            del X, strf
        del cols_a, cols_b, ctx
        gc.collect()
        log(f"features {country} done ({time.time() - t0:.0f}s)")
    # entity lists per group (incl. entities with no candidates) for honest evaluation
    ents = []
    for country in countries("train"):
        s1 = load_country("train", (1,), country, ["entity_id", "addr_state", "addr_city"])
        groups = np.where(s1["entity_id"].isin(hidden_all), "hidden", assign_groups(s1))
        ents.append(pd.DataFrame({"s1_id": s1["entity_id"], "group": groups, "country": country}))
    pd.concat(ents).to_parquet(FEATURE_DIR / "train_entity_groups.parquet", index=False)


class _NpySeq(lgb.Sequence):
    """Stream a memory-mapped feature matrix into LightGBM."""

    def __init__(self, path, batch_size: int = 500_000):
        self.X = np.load(path, mmap_mode="r")
        self.batch_size = batch_size

    def __getitem__(self, idx):
        return np.asarray(self.X[idx], dtype=np.float64)   # lgb.Sequence requires float64

    def __len__(self) -> int:
        return len(self.X)


def load_group(role: str) -> tuple[np.ndarray, pd.DataFrame]:
    Xs, metas = [], []
    for country in countries("train"):
        Xs.append(np.load(FEATURE_DIR / f"X_{role}_{country}.npy").astype(np.float32))
        metas.append(pd.read_parquet(FEATURE_DIR / f"meta_{role}_{country}.parquet").assign(country=country))
    return np.vstack(Xs), pd.concat(metas, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-features", action="store_true")
    args = ap.parse_args()
    if not args.reuse_features:
        build_features()
    fit_seqs = [_NpySeq(FEATURE_DIR / f"X_fit_{c}.npy") for c in countries("train")]
    fit_y = np.concatenate([pd.read_parquet(FEATURE_DIR / f"meta_fit_{c}.parquet")["label"].to_numpy()
                            for c in countries("train")])
    X_es, meta_es = load_group("es")
    log(f"fit {len(fit_y):,} pairs ({fit_y.mean():.3f} positive), es {len(meta_es):,}")
    dfit = lgb.Dataset(fit_seqs, fit_y, feature_name=FEATURES, categorical_feature=CATEGORICAL,
                       free_raw_data=True)
    des = lgb.Dataset(X_es, meta_es["label"], reference=dfit)
    booster = lgb.train(LGB_PARAMS, dfit, num_boost_round=5000, valid_sets=[des],
                        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(250)])
    booster.save_model(str(MODEL_DIR / "lgbm_matcher.txt"))
    imp = pd.Series(booster.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    log("top features: " + ", ".join(f"{k}={v:.0f}" for k, v in imp.head(20).items()))
    del dfit, des, X_es

    X_thr, meta_thr = load_group("thr")
    meta_thr["p"] = booster.predict(X_thr, num_threads=7)
    groups = pd.read_parquet(FEATURE_DIR / "train_entity_groups.parquet")
    thr_ents = groups.loc[groups["group"] == "thr", "s1_id"]
    truth = ground_truth_pairs(read_ground_truth()).rename(columns={"source1_entity_id": "s1_id",
                                                                    "entity_id": "cand_id"})
    best, table = tune(meta_thr[["s1_id", "cand_id", "p"]], truth, thr_ents)
    log(f"decision rule: {best}")
    config = {"features": FEATURES, "best_iteration": booster.best_iteration, "decision": best,
              "validation": table, "lgb_params": LGB_PARAMS}
    (MODEL_DIR / "matcher_config.json").write_text(json.dumps(config, indent=2, default=float))
    imp.to_csv(MODEL_DIR / "feature_importance.csv", header=["gain"])
    meta_thr[["s1_id", "cand_id", "label", "p", "country"]].to_parquet(FEATURE_DIR / "thr_scored.parquet",
                                                                       index=False)


if __name__ == "__main__":
    main()
