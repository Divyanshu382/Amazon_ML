# Status and handoff (2026-09-27)

## Scores

| version | validation macro F0.5 | public leaderboard |
|---|---|---|
| v1 | 0.9714 | 0.96200 |
| v2 | 0.9860 | 0.97855 |
| v3 | 0.98636 | 0.97960 |
| v5 | 0.98674 | 0.97806 |
| **v7** | **0.98695** | **0.98340** (best) |

**1 leaderboard submission is left.** The leaderboard keeps the best score.

## Files NOT in this repository (copy them by hand)

The repository is public, so it holds no challenge data and no predictions. Anyone could upload our
predictions as their own.
- `dataset/`: download again from the challenge portal. `utils/validate_submission.py` comes from the
  same kit.
- `output/matching_results_v7.tsv` and the matching `output/candidate_pairs.tsv` (808 MB), from the
  old laptop. **If v8 does not beat 0.98340, these two files go into the final zip**, since the zip
  must contain the files behind the final leaderboard score.

## Code state

Everything in `main` is the current code. Since v7 was scored, it gained these France-only
normalisation rules:
- names: "et" → "and";
- addresses: "Nord" → region code; "10 B" / "10 T" → bis / ter; q → quai; psg / pass → passage;
  appt / app / apt / apartment → appartement.

The shipped model (`models/lgbm_matcher.txt`) is the v6/v7 model. It was trained before the French
acronym rule existed; US/India are unaffected by the new rules.

## Next steps (v8)

1. **Setup:** `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`. Put the data
   in `dataset/` and the validator in `utils/`.
2. **Full run:** `python -m src.run_pipeline`. It preprocesses, blocks, retrains, scores and writes
   `output/`. With more RAM, try a larger `FIT_ENTITIES` in `src/matching/train_model.py` (700k now;
   train has ~2.2M S1 entities). LightGBM runs on CPU; a GPU is not needed.
3. **Already tried, no gain (skip):**
   - `experiments/second_model_blend.py`: a second LightGBM (511 leaves, feature fraction 0.5) blended
     with the main model scored 0.98685–0.98693 against 0.98695 for the main model alone.
   - The French rules added after v7 change only 0.08% of French pairs (687 removed, 715 added).
4. **Before uploading, compare against v7.**
   - Load `output/matching_results_v7.tsv` and the new file, and count per country the share of v7
     pairs removed and added.
   - US/India should change very little unless the model changed.
   - Large removals are a warning: v5 removed 1.6% of pairs on test vs 0.4% on validation, and it
     scored worse.
   - Upload only if validation improved and the test changes look like the validation changes.
5. **Final zip:** `python -m src.evaluation.package --team <team name>`. Also fill in the team name and
   date in `Documentation_template.md`.

## Lessons

- France (15% of test, no labels) is the main lever. Making French records look like the training
  countries (region fill) gave most of the v7 gain.
- Word statistics must be one table computed over train and test together. Computing them per split
  shifted on test (v5).
- The expected-F0.5 decision rule lost to the plain threshold rule on the leaderboard.
- Address-less copies of names shared by several businesses are coin flips. Blocking caps validation
  at ~0.996.
