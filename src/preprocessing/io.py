"""Raw TSV loading helpers shared by the whole pipeline.

All challenge files are tab-separated, unquoted, and use empty strings for
missing values. We therefore read with ``quoting=QUOTE_NONE`` and
``keep_default_na=False`` so that names like "NA" or "None" are never turned
into NaN behind our back.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "dataset"
PROCESSED_DIR = ROOT / "data" / "processed"

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]

_READ_KW = dict(sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                quoting=csv.QUOTE_NONE, encoding="utf-8")


def source_path(split: str, source: int | str) -> Path:
    """``source_path("train", 2)`` -> dataset/train/train_source2.tsv"""
    s = str(source).upper().lstrip("S")
    return DATASET_DIR / split / f"{split}_source{s}.tsv"


def gt_path() -> Path:
    return DATASET_DIR / "train" / "train_ground_truth.tsv"


def read_source(split: str, source: int | str, nrows: int | None = None,
                usecols: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(source_path(split, source), nrows=nrows, usecols=usecols, **_READ_KW)
    return df


def iter_source(split: str, source: int | str, chunksize: int = 500_000,
                usecols: list[str] | None = None) -> Iterator[pd.DataFrame]:
    yield from pd.read_csv(source_path(split, source), chunksize=chunksize,
                           usecols=usecols, **_READ_KW)


def read_ground_truth(nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(gt_path(), nrows=nrows, **_READ_KW)


def split_ids(value: str) -> list[str]:
    """Parse a comma-separated ID list; empty string -> []."""
    return [x for x in value.split(",") if x] if value else []


def ground_truth_pairs(gt: pd.DataFrame) -> pd.DataFrame:
    """Explode the ground truth into one (source1_entity_id, entity_id) row per match."""
    s = gt.set_index("source1_entity_id")["matched_entity_ids"].map(split_ids).explode().dropna()
    return s.rename("entity_id").reset_index()


def source_of(entity_id: str) -> str:
    """'S2-123' -> 'S2'."""
    return entity_id.split("-", 1)[0]
