"""Score every blocking candidate of a split with the trained matcher.

    python -m src.matching.predict --split test

Writes data/scores/{split}_{country}.parquet with (s1_id, cand_id, p) for every
candidate pair (the exact set in candidate_pairs.tsv).
"""
from __future__ import annotations

import argparse
import gc
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..blocking.candidate_generator import candidates_path, countries, load_country
from ..preprocessing.io import ROOT
from .feature_engineering import (RECORD_COLUMNS, build_word_stats, competition, context_features,
                                  record_columns, string_features)

MODEL_DIR = ROOT / "models"
SCORE_DIR = ROOT / "data" / "scores"
CHUNK = 3_000_000
EXTRA_MODELS: tuple = ()   # e.g. ("v3",): also score models/lgbm_matcher_v3.txt as column p_v3


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scores_path(split: str, country: str):
    return SCORE_DIR / f"{split}_{country.replace(' ', '_')}.parquet"


def load_model() -> lgb.Booster:
    return lgb.Booster(model_file=str(MODEL_DIR / "lgbm_matcher.txt"))


def word_stats(split: str | None = None) -> dict:
    """Label-free word statistics over train + test candidates together (one shared table for
    training and scoring; ``split`` is ignored). Computed once, then cached."""
    path = ROOT / "data" / "features" / "word_stats_joint.json"
    if path.exists():
        return json.loads(path.read_text())
    ws = build_word_stats(("train", "test"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ws))
    log("word stats (train + test): " + ", ".join(f"{k}={len(v):,}" for k, v in ws.items()))
    return ws


def _extra_models() -> dict:
    """Optional earlier models (models/lgbm_matcher_<v>.txt + matcher_config_<v>.json) also scored,
    for ensembling / comparison. Each uses the subset of current features it was trained on."""
    from .feature_engineering import FEATURES
    out = {}
    for v in EXTRA_MODELS:
        mf, cf = MODEL_DIR / f"lgbm_matcher_{v}.txt", MODEL_DIR / f"matcher_config_{v}.json"
        if mf.exists() and cf.exists():
            feats = json.loads(cf.read_text())["features"]
            if all(f in FEATURES for f in feats):
                out[v] = (lgb.Booster(model_file=str(mf)), [FEATURES.index(f) for f in feats])
    return out


def score_split(split: str, only: list[str] | None = None) -> None:
    booster = load_model()
    extra = _extra_models()
    ws = word_stats(split)
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    for country in countries(split):
        if only and country not in only:
            continue
        t0 = time.time()
        cands = pd.read_parquet(candidates_path(split, country))
        if len(cands) == 0:
            pd.DataFrame({"s1_id": [], "cand_id": [], "p": []}).to_parquet(scores_path(split, country))
            continue
        s1 = load_country(split, (1,), country, RECORD_COLUMNS)
        other = load_country(split, (2, 3), country, RECORD_COLUMNS)
        comp = competition(cands, s1, other)
        cols_a, cols_b = record_columns(s1), record_columns(other)
        del s1, other
        gc.collect()
        p = np.empty(len(cands), np.float32)
        p_extra = {v: np.empty(len(cands), np.float32) for v in extra}
        for off in range(0, len(cands), CHUNK):
            sel = np.arange(off, min(off + CHUNK, len(cands)))
            strf = string_features(cands["a"].to_numpy()[sel], cands["b"].to_numpy()[sel], cols_a, cols_b, ws=ws)
            # same float16 rounding as the stored training features
            X = np.hstack([strf, context_features(cands, comp, sel)]).astype(np.float16).astype(np.float32)
            p[sel] = booster.predict(X, num_threads=7)
            for v, (bst, idx) in extra.items():
                p_extra[v][sel] = bst.predict(X[:, idx], num_threads=7)
            log(f"  {split}/{country}: scored {sel[-1] + 1:,}/{len(cands):,}")
        out = pd.DataFrame({"s1_id": cands["s1_id"], "cand_id": cands["cand_id"], "p": p,
                            **{f"p_{v}": arr for v, arr in p_extra.items()}})
        out.to_parquet(scores_path(split, country), compression="zstd", index=False)
        log(f"{split}/{country}: {len(cands):,} pairs scored ({time.time() - t0:.0f}s)")
        del cols_a, cols_b, cands, comp
        gc.collect()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--countries", nargs="*", help="score only these countries (others keep their scores)")
    args = ap.parse_args()
    score_split(args.split, args.countries)


if __name__ == "__main__":
    main()
