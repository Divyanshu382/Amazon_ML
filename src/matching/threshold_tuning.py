"""Decision rules on top of match probabilities, tuned for macro F0.5.

threshold rule (``decide``)
    1. one-to-one: each S2/S3 record goes to the single S1 entity with the
       highest probability (the ground truth never links one record to two S1s);
    2. keep the pair if p >= t;
    3. optionally drop pairs far below the entity's best pair: p >= r * max_p(entity).

expected-F rule (``decide_expected_f``)
    after the one-to-one step, for every S1 entity pick the prefix of its
    candidates (sorted by p) that maximises the EXPECTED per-entity F0.5, treating
    the candidates as independent Bernoulli(p * calibration). The empty set is
    one of the options: it scores 1 exactly when the entity has no true match,
    which is how singletons earn their credit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluation.metrics import per_entity_scores, summarize

T_GRID = [round(x, 3) for x in np.arange(0.40, 0.96, 0.025)]
R_GRID = [0.0, 0.5]
CAL_GRID = [0.5, 0.6, 0.7, 0.8]
MAX_CANDS = 12          # candidates per entity considered by the expected-F rule (sorted by p)
P_FLOOR = 0.005         # below this a candidate is treated as certainly false


def assign_one_to_one(scored: pd.DataFrame) -> pd.DataFrame:
    """Keep, for every candidate record, only its highest-probability S1 entity."""
    best = scored.groupby("cand_id")["p"].transform("max")
    return scored[scored["p"] >= best].drop_duplicates("cand_id")


def decide(scored: pd.DataFrame, t: float, r: float = 0.0, one_to_one: bool = True) -> pd.DataFrame:
    """scored: (s1_id, cand_id, p) -> kept (s1_id, cand_id, p)."""
    d = assign_one_to_one(scored) if one_to_one else scored
    d = d[d["p"] >= t]
    if r > 0 and len(d):
        d = d[d["p"] >= r * d.groupby("s1_id")["p"].transform("max")]
    return d


def _poisson_binomial(p: np.ndarray) -> np.ndarray:
    """Distribution of the number of successes of independent Bernoulli(p_i)."""
    dist = np.zeros(len(p) + 1)
    dist[0] = 1.0
    for i, pi in enumerate(p):
        dist[1:i + 2] = dist[1:i + 2] * (1 - pi) + dist[0:i + 1] * pi
        dist[0] *= 1 - pi
    return dist


def best_prefix(p_sorted: np.ndarray) -> int:
    """Number of top candidates maximising expected F0.5 (0 = predict no match)."""
    n = len(p_sorted)
    best_k, best_e = 0, float(np.prod(1 - p_sorted))          # empty set: P(no true match)
    for k in range(1, n + 1):
        tp = _poisson_binomial(p_sorted[:k])                    # true matches inside the set
        fn = _poisson_binomial(p_sorted[k:])                    # true matches left out
        t = np.arange(k + 1)[:, None]
        x = np.arange(n - k + 1)[None, :]
        f = 1.25 * t / (0.25 * (t + x) + k)
        e = float((tp[:, None] * fn[None, :] * f).sum())
        if e > best_e:
            best_k, best_e = k, e
    return best_k


def decide_expected_f(scored: pd.DataFrame, cal: float = 1.0) -> pd.DataFrame:
    """Expected-F0.5-optimal subset per entity (after the one-to-one step)."""
    d = assign_one_to_one(scored)
    d = d[d["p"] > P_FLOOR].sort_values(["s1_id", "p"], ascending=[True, False])
    d = d.groupby("s1_id", sort=False).head(MAX_CANDS)
    p = np.clip(d["p"].to_numpy(np.float64) * cal, 0.0, 1.0)
    ids = d["s1_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], len(ids)]
    keep = np.zeros(len(d), bool)
    for s, e in zip(starts, ends):
        ps = p[s:e]
        if ps[0] > 0.999 and (e - s == 1 or ps[1] < 0.01):  # fast path: one sure match
            keep[s] = True
            continue
        keep[s:s + best_prefix(ps)] = True
    return d[keep]


def tune(scored: pd.DataFrame, truth: pd.DataFrame, entities,
         rules: tuple = ("threshold",)) -> tuple[dict, list[dict]]:
    """Grid-search decision rules on macro F0.5. truth: (s1_id, cand_id) of ALL true pairs.

    Only the threshold rule is selected by default: the expected-F rule relies on calibrated
    probabilities, and on the leaderboard it was clearly worse than the threshold while
    validation had them within noise. It is still reported in the table."""
    table = []
    for r in R_GRID:
        for t in T_GRID:
            s = summarize(per_entity_scores(decide(scored, t, r), truth, entities))
            table.append({"rule": "threshold", "t": t, "r": r, **s})
    for cal in CAL_GRID:
        s = summarize(per_entity_scores(decide_expected_f(scored, cal), truth, entities))
        table.append({"rule": "expected_f", "cal": cal, **s})
    best = max((row for row in table if row["rule"] in rules), key=lambda x: x["macro_f05"])
    keys = ("rule", "t", "r", "cal", "macro_f05")
    return {k: best[k] for k in keys if k in best}, table


def apply_rule(scored: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if cfg.get("rule") == "expected_f":
        return decide_expected_f(scored, cfg.get("cal", 1.0))
    return decide(scored, cfg["t"], cfg.get("r", 0.0))
