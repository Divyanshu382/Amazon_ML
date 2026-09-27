"""Validation split of the TRAIN Source-1 entities.

Each S2/S3 record matches at most one S1 entity, and the decision rule assigns
every candidate record to its best S1 entity only. To validate that honestly,
entities that compete for the same records (same name, same area) must land in
the same split, so we split by a hash of (country, state, city), not by entity.

    group "es"  (~4%)  early stopping of the model
    group "thr" (~4%)  decision-threshold tuning and the reported validation score
    group "fit" (rest) model training
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

ES_PCT, THR_PCT = 4, 4


def _bucket(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16) % 100


def assign_groups(s1: pd.DataFrame) -> np.ndarray:
    """s1 needs country_norm, addr_state, addr_city, entity_id -> array of 'fit'/'es'/'thr'."""
    keys = np.where(
        s1["addr_state"].to_numpy() != "",
        s1["country_norm"] + "|" + s1["addr_state"] + "|" + s1["addr_city"],
        s1["entity_id"],
    )
    uniq, inv = np.unique(keys.astype(str), return_inverse=True)
    buckets = np.array([_bucket(k) for k in uniq])[inv]
    return np.where(buckets < ES_PCT, "es", np.where(buckets < ES_PCT + THR_PCT, "thr", "fit"))
