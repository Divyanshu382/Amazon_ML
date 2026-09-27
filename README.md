# Business Entity Resolution: reproducible pipeline

Amazon ML Challenge 2026. Raw challenge TSVs -> `output/matching_results.tsv` + `output/candidate_pairs.tsv`.

| | macro F0.5 |
|---|---|
| Validation (46,818 held-out training entities) | **0.98695** (India 0.9874, US 0.9867) |
| Precision / recall / singletons left empty | 0.9968 / 0.9664 / 98.9% |
| Public leaderboard, best so far (v7) | **0.98340** |

The official validator reports PASS on the test outputs. The methodology is in
[`Documentation_template.md`](Documentation_template.md).

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
# macOS only: LightGBM needs the OpenMP runtime
brew install libomp
```

Put the challenge data at `dataset/train/*.tsv` and `dataset/test/*.tsv` (the layout from the
student resource zip) and the organisers' validator at `utils/validate_submission.py`. Neither
is in this repository.

## Run

```bash
.venv/bin/python -m src.run_pipeline                    # preprocess -> blocking -> train -> predict -> submit
.venv/bin/python -m src.run_pipeline --from predict     # reuse the shipped model in models/, only score the test set
.venv/bin/python -m src.run_pipeline --from mine        # also re-mine the lookup tables (shipped in resources/)
```

| stage | module | output | time (8-core laptop, 8 GB RAM) |
|---|---|---|---|
| preprocess | `src/preprocessing/preprocessing.py` | `data/processed/*.parquet` | ~12 min |
| blocking | `src/blocking/candidate_generator.py` | `data/candidates/*.parquet` | ~45 min |
| train | `src/matching/train_model.py` | `models/lgbm_matcher.txt`, `models/matcher_config.json` | ~40 min |
| predict | `src/matching/predict.py` | `data/scores/*.parquet` | ~35 min |
| submit | `src/evaluation/submission.py` | `output/*.tsv` + validator run | ~6 min |

Blocking and scoring can be re-run for single countries, e.g.
`python -m src.blocking.candidate_generator --split test --countries france` and
`python -m src.matching.predict --split test --countries france`.

Other commands:
- `python -m src.preprocessing.eda` writes `reports/eda_report.md`.
- `python -m src.evaluation.error_analysis` writes `reports/error_analysis.md`.
- `python -m src.evaluation.package --team <name>` builds the final submission zip.
- `python -m pytest tests -q` runs the 69 unit tests.

## Layout

```
src/
  preprocessing/   raw TSV -> standard records (names, addresses, countries; 9 Indic scripts)
    resources/     tables mined from the provided data (Indic token map, vocabularies, ...)
  blocking/        multi-pass key blocking with per-signal candidate quotas
  matching/        pairwise features, word statistics, LightGBM training, test scoring, decision rule
  evaluation/      F0.5 metric, validation split, error analysis, submission writer, packager
  run_pipeline.py  entry point
models/            trained LightGBM matcher + tuned decision rule
reports/           EDA and error-analysis reports
tests/             unit tests
```

## Rules compliance

* **No external data.** There are no APIs, no geocoding and no downloaded datasets. The lookup
  tables are mined from the provided files. Anything that uses labels uses train only; the
  unsupervised word statistics and the French city -> region table use the text of the provided
  files only.
* **Model:** LightGBM 4.7 (MIT licence), a gradient-boosted tree ensemble of 912 trees (~25 MB),
  far below the 8B-parameter limit. rapidfuzz (MIT) supplies string similarities.
* **Country is an open set.** Countries are discovered from the data. France, which is absent
  from train, flows through the same pipeline.
* Every final match is a subset of `candidate_pairs.tsv` (the exact set the model scores).
