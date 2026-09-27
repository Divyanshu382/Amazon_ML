"""Competition metric: F0.5 computed per Source-1 entity, then macro-averaged.

Per entity (T = true matches, P = predicted matches):
    T = {} and P = {}  -> 1.0   (correct singleton)
    T = {} xor P = {}  -> 0.0
    otherwise          -> 1.25 * prec * rec / (0.25 * prec + rec), 0 if no overlap
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def per_entity_scores(pred: pd.DataFrame, truth: pd.DataFrame, entities) -> pd.DataFrame:
    """pred / truth: DataFrames with columns (s1_id, cand_id). entities: every S1 id evaluated."""
    ents = pd.Index(pd.unique(np.asarray(entities)), name="s1_id")
    pred = pred[["s1_id", "cand_id"]].drop_duplicates()
    truth = truth[["s1_id", "cand_id"]].drop_duplicates()
    pred = pred[pred["s1_id"].isin(ents)]
    truth = truth[truth["s1_id"].isin(ents)]
    n_pred = pred.groupby("s1_id").size().reindex(ents, fill_value=0)
    n_true = truth.groupby("s1_id").size().reindex(ents, fill_value=0)
    tp = pred.merge(truth, on=["s1_id", "cand_id"]).groupby("s1_id").size().reindex(ents, fill_value=0)
    p = np.where(n_pred > 0, tp / np.maximum(n_pred, 1), 0.0)
    r = np.where(n_true > 0, tp / np.maximum(n_true, 1), 0.0)
    denom = 0.25 * p + r
    f = np.where(denom > 0, 1.25 * p * r / np.where(denom > 0, denom, 1), 0.0)
    f = np.where((n_true == 0) & (n_pred == 0), 1.0, f)
    return pd.DataFrame({"n_true": n_true.to_numpy(), "n_pred": n_pred.to_numpy(), "tp": tp.to_numpy(),
                         "precision": p, "recall": r, "f05": f}, index=ents)


def summarize(scores: pd.DataFrame) -> dict:
    single = scores["n_true"] == 0
    has = ~single
    return {
        "entities": int(len(scores)),
        "macro_f05": float(scores["f05"].mean()),
        "macro_precision (entities with predictions)": float(scores.loc[scores["n_pred"] > 0, "precision"].mean()),
        "macro_recall (entities with matches)": float(scores.loc[has, "recall"].mean()),
        "micro_precision": float(scores["tp"].sum() / max(scores["n_pred"].sum(), 1)),
        "micro_recall": float(scores["tp"].sum() / max(scores["n_true"].sum(), 1)),
        "singletons": int(single.sum()),
        "singleton_accuracy (predicted empty)": float((scores.loc[single, "n_pred"] == 0).mean()) if single.any() else float("nan"),
        "f05_non_singletons": float(scores.loc[has, "f05"].mean()),
    }


def f05_macro(pred: pd.DataFrame, truth: pd.DataFrame, entities) -> float:
    return float(per_entity_scores(pred, truth, entities)["f05"].mean())
