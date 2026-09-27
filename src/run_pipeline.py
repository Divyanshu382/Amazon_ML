"""End-to-end pipeline: raw TSV -> output/matching_results.tsv + output/candidate_pairs.tsv.

    python -m src.run_pipeline                 # everything (about 1.5-2 h on an 8-core laptop)
    python -m src.run_pipeline --from blocking # resume from a stage

Stages
  mine        data-driven lookup tables (src/preprocessing/resources, committed)
  preprocess  normalise all six source files -> data/processed/*.parquet
  blocking    top-30 candidates per Source-1 entity -> data/candidates/
  train       features + LightGBM + decision-rule tuning -> models/
  predict     score every test candidate -> data/scores/
  submit      write output/*.tsv and run utils/validate_submission.py
"""
from __future__ import annotations

import argparse
import sys
import time

STAGES = ["mine", "preprocess", "blocking", "train", "predict", "submit"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", choices=STAGES, default="preprocess",
                    help="first stage to run (default: preprocess; the mined tables are committed)")
    ap.add_argument("--to", dest="end", choices=STAGES, default="submit")
    args = ap.parse_args()
    stages = STAGES[STAGES.index(args.start):STAGES.index(args.end) + 1]
    for stage in stages:
        t0 = time.time()
        print(f"=== {stage} ===", flush=True)
        if stage == "mine":
            sys.argv = ["mine_resources"]
            from .preprocessing import mine_resources
            mine_resources.main()
        elif stage == "preprocess":
            sys.argv = ["preprocessing"]
            from .preprocessing import preprocessing
            preprocessing.main()
        elif stage == "blocking":
            from .blocking.candidate_generator import run_blocking
            run_blocking("train")
            run_blocking("test")
        elif stage == "train":
            sys.argv = ["train_model"]
            from .matching import train_model
            train_model.main()
        elif stage == "predict":
            from .matching.predict import score_split
            score_split("test")
        elif stage == "submit":
            from .evaluation.submission import make_submission, run_validator
            make_submission("test")
            code = run_validator("test")
            if code:
                sys.exit(code)
        print(f"=== {stage} done in {time.time() - t0:.0f}s ===", flush=True)


if __name__ == "__main__":
    main()
