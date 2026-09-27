"""Write output/matching_results.tsv and output/candidate_pairs.tsv, then run the
official validator.

    python -m src.evaluation.submission --split test

* every Source-1 entity of the split appears exactly once (file order);
* matched ids are a subset of the candidate ids (the decision rule only keeps
  scored candidates), S2/S3 only, no duplicates.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

import pandas as pd

from ..blocking.candidate_generator import countries
from ..matching.predict import MODEL_DIR, scores_path
from ..matching.threshold_tuning import apply_rule
from ..preprocessing.io import DATASET_DIR, ROOT, read_source

OUTPUT_DIR = ROOT / "output"


def _write(path, s1_ids: pd.Series, pairs: pd.DataFrame, col: str) -> None:
    lists = pairs.groupby("s1_id", sort=False)["cand_id"].agg(lambda x: ",".join(dict.fromkeys(x)))
    joined = s1_ids.map(lists).fillna("")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for sid, ids in zip(s1_ids, joined):
            f.write(f"{sid}\t{ids}\n")


def make_submission(split: str = "test") -> dict:
    cfg = json.loads((MODEL_DIR / "matcher_config.json").read_text())["decision"]
    scored = pd.concat([pd.read_parquet(scores_path(split, c)) for c in countries(split)], ignore_index=True)
    matches = apply_rule(scored, cfg)
    s1_ids = read_source(split, 1, usecols=["entity_id"])["entity_id"]
    OUTPUT_DIR.mkdir(exist_ok=True)
    scored = scored.sort_values(["s1_id", "p"], ascending=[True, False], kind="stable")
    _write(OUTPUT_DIR / "candidate_pairs.tsv", s1_ids, scored, "candidate_entity_ids")
    _write(OUTPUT_DIR / "matching_results.tsv", s1_ids, matches, "matched_entity_ids")
    stats = {
        "s1_entities": int(len(s1_ids)),
        "candidate_pairs": int(len(scored)),
        "matched_pairs": int(len(matches)),
        "entities_with_matches": int(matches["s1_id"].nunique()),
        "entities_predicted_singleton": int(len(s1_ids) - matches["s1_id"].nunique()),
        "decision": cfg,
    }
    print(json.dumps(stats, indent=2))
    return stats


def run_validator(split: str = "test") -> int:
    cmd = [sys.executable, str(ROOT / "utils" / "validate_submission.py"),
           "--matching", str(OUTPUT_DIR / "matching_results.tsv"),
           "--candidate", str(OUTPUT_DIR / "candidate_pairs.tsv"),
           "--test-dir", str(DATASET_DIR / split)]
    return subprocess.call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    make_submission(args.split)
    sys.exit(run_validator(args.split))


if __name__ == "__main__":
    main()
