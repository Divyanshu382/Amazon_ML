"""Error analysis on the held-out 'thr' validation group -> reports/error_analysis.md

    python -m src.evaluation.error_analysis

Every true pair of a validation entity ends up in exactly one bucket:
  found                  predicted
  missed: not blocked    never a candidate (blocking recall ceiling)
  missed: low score      candidate, p below the threshold
  missed: one-to-one     candidate above threshold, but assigned to another S1 entity
and every predicted pair that is not true is a false merge (split by whether the
S1 entity is a singleton).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..blocking.candidate_generator import load_country
from ..matching.threshold_tuning import apply_rule
from ..preprocessing.io import ROOT, ground_truth_pairs, read_ground_truth
from .metrics import per_entity_scores, summarize

FEATURE_DIR = ROOT / "data" / "features"
MODEL_DIR = ROOT / "models"
REPORT = ROOT / "reports" / "error_analysis.md"


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(str(v).replace("|", "\\|") for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def main() -> None:
    cfg = json.loads((MODEL_DIR / "matcher_config.json").read_text())["decision"]
    thr = pd.read_parquet(FEATURE_DIR / "thr_scored.parquet")   # written by train_model
    groups = pd.read_parquet(FEATURE_DIR / "train_entity_groups.parquet")
    ents = groups.loc[groups["group"] == "thr", "s1_id"]
    truth = ground_truth_pairs(read_ground_truth()).rename(columns={"source1_entity_id": "s1_id",
                                                                    "entity_id": "cand_id"})
    truth = truth[truth["s1_id"].isin(set(ents))]
    pred = apply_rule(thr[["s1_id", "cand_id", "p"]], cfg)
    scores = per_entity_scores(pred, truth, ents)
    summary = summarize(scores)

    key = ["s1_id", "cand_id"]
    t = truth.merge(thr[key + ["p", "country"]], on=key, how="left")
    t = t.merge(pred[key].assign(pred=1), on=key, how="left")
    best_for_cand = thr.groupby("cand_id")["p"].transform("max")
    lost = thr.loc[thr["p"] < best_for_cand, key].assign(lost=1)
    t = t.merge(lost, on=key, how="left")
    t["bucket"] = np.select(
        [t["pred"] == 1, t["p"].isna(), t["lost"] == 1],
        ["found", "missed: not blocked", "missed: one-to-one"], "missed: low score")
    buckets = t["bucket"].value_counts().rename_axis("bucket").reset_index(name="true pairs")
    buckets["share"] = (buckets["true pairs"] / len(t)).map(lambda v: f"{100 * v:.2f}%")

    fp = pred.merge(truth.assign(true=1), on=key, how="left")
    fp = fp[fp["true"].isna()]
    singleton = set(scores.index[scores["n_true"] == 0])
    fp_tbl = pd.DataFrame({"false merges": [len(fp)],
                           "on singleton entities": [int(fp["s1_id"].isin(singleton).sum())],
                           "predicted pairs": [len(pred)]})

    # examples with raw text
    cols = ["entity_id", "business_name_raw", "business_address_raw"]
    s1 = pd.concat([load_country("train", (1,), c, cols) for c in ("india", "us")]).set_index("entity_id")
    need = set(fp["cand_id"]) | set(t.loc[t["bucket"] != "found", "cand_id"])
    other = pd.concat([load_country("train", (2, 3), c, cols) for c in ("india", "us")])
    other = other[other["entity_id"].isin(need)].set_index("entity_id")

    def examples(df: pd.DataFrame, n: int = 12) -> str:
        df = df.sample(min(n, len(df)), random_state=1) if len(df) else df
        rows = [{"S1 name": s1.at[r.s1_id, "business_name_raw"], "S1 address": s1.at[r.s1_id, "business_address_raw"][:60],
                 "candidate name": other.at[r.cand_id, "business_name_raw"],
                 "candidate address": other.at[r.cand_id, "business_address_raw"][:60],
                 "p": "-" if pd.isna(r.p) else f"{r.p:.2f}"} for r in df.itertuples()]
        return _md(pd.DataFrame(rows)) if rows else "_none_"

    per_country = []
    country_of = thr.drop_duplicates("s1_id").set_index("s1_id")["country"]
    for c, idx in scores.groupby(scores.index.map(country_of).fillna("no candidates")).groups.items():
        per_country.append({"country": c, **{k: (round(v, 4) if isinstance(v, float) else v)
                                              for k, v in summarize(scores.loc[idx]).items()}})
    md = f"""# Error analysis (validation group 'thr', {len(ents):,} train entities)

Decision rule: {cfg}

## Summary

{_md(pd.DataFrame([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()}]))}

## By country

{_md(pd.DataFrame(per_country))}

## Where the true pairs go

{_md(buckets)}

## False merges

{_md(fp_tbl)}

### Examples: false merges
{examples(fp.assign(p=fp["p"]))}

### Examples: missed, not blocked
{examples(t[t["bucket"] == "missed: not blocked"])}

### Examples: missed, low score
{examples(t[t["bucket"] == "missed: low score"])}

### Examples: missed, one-to-one conflict
{examples(t[t["bucket"] == "missed: one-to-one"])}
"""
    REPORT.write_text(md, encoding="utf-8")
    print(md[:3000])


if __name__ == "__main__":
    main()
