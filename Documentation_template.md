# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]  
**Team Members:** Divyanshu (data & preprocessing), Aditya (blocking & candidate generation), Santosh (matching model), Vivek (evaluation, integration & submission)  
**Submission Date:** [Date]

---

## 1. Executive Summary

Every record is first normalised into a canonical form:
- nine Indian scripts are transliterated through a token dictionary mined from the training matches;
- legal forms and address abbreviations are canonicalised;
- glued domain names and handles are split.

A vectorised multi-pass key blocker then generates about 37 candidates per Source-1 entity. Separate
quotas reserve slots for name-only, address-only and address-less evidence.

A LightGBM classifier scores each candidate on 78 features. Besides name and address similarity,
these include two kinds of label-free signal:
- **Word statistics:** does a word that the candidate adds or drops behave like copy noise or like
  a different, sibling business?
- **Competition and duplicate counts:** how contested a record is, and how common a name is among
  Source-1 entities.

Finally, a one-to-one rule assigns each Source-2/3 record to its single best Source-1 entity. It
keeps the pair only above a threshold tuned for macro F0.5.

On 46,818 held-out training entities the pipeline scores **macro F0.5 = 0.98695**:
- India 0.9874, US 0.9867;
- precision 0.997, recall 0.966;
- 98.9% of singletons correctly left empty.

---

## 2. Methodology

### 2.1 Problem Analysis

Full EDA: `reports/eda_report.md`.

* **Scale:** test has 1.73M S1 entities against 9.97M S2/S3 records, which is 1.7 x 10^13 possible pairs.
* **Source 1 is clean** (ASCII, title case). All the noise is in S2 and S3.
  * **Names:**
    - 9–11% are fully or partly written in Indian scripts;
    - injected accents (6–8%) and look-alike digits (`5tar`, `Mem0rial`);
    - alias markers `<synthetic brand> dba|f/k/a|formerly <real name>` (2.4% of S3);
    - `#12345` / `(ID: n)` / `| www.site` tails, and glued domain names (3.5%);
    - shuffled words, typos, and brand-only names that keep nothing of the real name (2.2%).
  * **Addresses:**
    - the state is written as a code, a full name or in native script;
    - components are reordered, and 3.4% are missing or placeholders (`<NULL>`, `N/A`);
    - injected PMB boxes, zero-padded numbers, and fake prepended numbers (`DOOR NO #904 284`).
* **Ground truth:**
  * 3.46 matches per entity on average, and 5.6% of entities are singletons.
  * No S2/S3 record is ever matched to two S1 entities. This is the key structural fact behind our
    decision rule.
  * 26% of S2/S3 records match nothing (distractors), but only 2.3% of address-less records do.
* **Decoy siblings.** Many distractors copy a real business's name with a word added and sit a few
  doors up the same street. Examples are "X Holdings" and "X International" in the US, and
  "X Groupe" and "X Développement" in France. True copies instead keep the house number while adding
  noise words ("X Services", "X & Fils").
* **Names are not unique:** 32% of S1 records share their normalised name with another S1 entity
  in the same country. Only the address separates them.
* **Density:** test has 5.8 S2/S3 records per S1 entity against 4.7 in train, consistent with about
  19% of the test's businesses being absent from Source 1. We reproduce this in training and validation
  by hiding 19% of the training S1 entities, so their copies become distractors.
* **Country is an open set:** train has US and India; test adds France (15% of test S1).
  * A third of French S2/S3 addresses stop at the city ("77 AV LEON JOUHAUX, LILLE"). US and Indian
    copies almost always keep the state.

### 2.2 Solution Strategy

**Approach Type:** Blocking + classifier, with a collective (one-to-one) decision rule.  
**Core Innovation:**
1. **Data-mined normalisation.** A native-script-to-English token dictionary (1,417 tokens across
   9 scripts) is aligned on training matches. Together with a phonetic fallback, it covers 96% of native-script tokens in test.
   Brand syllables are mined from alias markers, so that glued real names are split but synthetic
   brand words are not.
2. **Label-free word statistics.** For every word a candidate adds to or drops from the Source-1
   name, and for every change of legal form, we count over all candidate pairs how often the two
   records still share the house number. Noise words keep it (≈0.75); decoy words do not (≈0.05).
   There is a loose version (any shared address word) and a strict one (same street). The table is
   computed once over train and test candidates together, so it means the same thing at training
   and scoring time and works for French words without French labels.
3. **Competition-aware matching.** Features describe how contested a candidate record is across S1
   entities (`b_rank`, `b_score_gap`, `b_deg`) and how common its name is among S1 entities. The
   one-to-one rule turns "a record belongs to one entity" into precision.

---

## 3. Candidate Generation (Blocking)

Module: `src/blocking/candidate_generator.py`. Every record emits keys, all scoped to the record's
country, which is read from the data.

| key | example |
|---|---|
| core-name token + state; phonetic token + state | `sun\|mh`, `sftvr\|mh` |
| core-name token (country-wide); any pair of core tokens | `sun`, `software sun` |
| whole core name; + state; + any address word | `sun software`, `sun software\|mh` |
| core-name token + address number; core-name token + city | `sun\|5`, `sun\|thane` |
| address number + state; address number + address word | `5\|mh`, `5\|krishna` |
| address word + state; adjacent address-word bigram | `dahisar\|mh`, `krishna shopping` |
| core-name token / whole name / token pair, S2/S3 records without an address only | reaches address-less records |

* Keys whose S2/S3 block exceeds a per-kind cap (150–600) are dropped.
* Each pair is scored by the summed IDF of its shared keys: in total, over name keys only, and over
  address keys only.
* A candidate is kept if it is in the entity's
  - top 30 by total score;
  - top 15 by address score (brand-only or renamed copies);
  - top 5 by name score (copies with a rewritten address);
  - or top 3 address-less candidates by name score.
* Implementation is sorted numpy key arrays plus `searchsorted`. A whole country runs chunk by
  chunk in bounded memory on an 8 GB laptop.
* **Candidate pairs generated (test):** 63,987,416 (36.9 per entity; France 9.7M, India 29.2M,
  US 25.1M). The reduction ratio versus all pairs is 0.999996.
* **How we ensured true matches were not lost:** recall was measured against training ground truth
  and every miss category was examined. Each fix is a separate key kind or quota:
  * generic names → name + address compound keys;
  * injected fake numbers → every number becomes a key;
  * missing addresses → state-free keys and the address-less quota;
  * renamed / brand-only copies → the address-score quota;
  * Telangana/Andhra Pradesh → treated as one state.

  On the held-out group, **98.8% of true pairs are in the candidate set** (India 98.5%, US 99.0%).
  The remainder is mostly address-less copies whose name carries several typos.

---

## 4. Matching Model

Module: `src/matching/`. Preprocessing: `src/preprocessing/` (see its README).

**Features used (78):**
- **Name (24):**
  - exact match of the core and normalised name; token Jaccard;
  - rapidfuzz ratio, token-set, token-sort and partial ratio; Jaro-Winkler;
  - character-trigram Jaccard; ratio on the glued (space-free) name;
  - phonetic-key Jaccard and exact match; first-token match;
  - acronym match (`JGIP` = Jai Gold Impex; French acronyms skip articles and keep "Ets", so
    `EL` = Ets Lavant SARL);
  - similarity to the alias part of the candidate; token containment in both directions;
  - token counts; legal-form Jaccard and conflict.
- **Address (18):**
  - candidate address missing; state agreement (AP/TG treated as equal); house-number agreement;
  - address-number Jaccard, containment and count;
  - address token Jaccard, token-set and plain ratio; street similarity;
  - city agreement and similarity; postcode agreement; shared address words;
  - **house-number relation:** equal, leading or trailing digit dropped, one digit changed,
    permuted, a few doors up or down, or other. The difference and cross-containment are also
    included, since noise drops digits while decoy siblings sit a few doors up.

  Missing information is encoded as **-1 (unknown)**, not as a mismatch.
- **Word statistics (17):**
  - number of words added and dropped; best typo similarity between them;
  - min / max / mean house-number agreement rate of the added words, the dropped words and the
    legal-form change;
  - each in the loose and the strict (same-street) version;
  - number of added words never seen before.
- **Context (19):**
  - blocking score (total, name, address), rank, and number of shared keys; candidates per entity;
  - for the candidate record: how many S1 entities retrieved it, and its rank and score gap versus
    the best of them;
  - source (S2/S3), `name_known_ratio` (0 = brand-only name), glued-domain flag, script;
  - how many S1 entities share the S1 name (core; core + state; full name with legal form);
  - how many S1 entities carry the candidate's name.

**Model type:** LightGBM gradient-boosted trees (MIT licence):
- 912 trees, 255 leaves, learning rate 0.05;
- feature fraction 0.7, bagging 0.8;
- early-stopped on log-loss.

**Training data:**
* 21.3M candidate pairs from 700k training entities (9.1% positive).
* Negatives are the blocking candidates themselves, so they are realistic hard negatives: same name
  tokens, same street, same house number.
* 19% of training S1 entities are hidden, to reproduce the test's density.

**Validation split:**
* By hash of (country, state, city), so that S1 entities competing for the same records stay in the
  same split.
* 4% of cities are used for early stopping, and 4% for threshold tuning and reporting.

**Threshold selection method:** grid search on the held-out group, maximising macro F0.5 over the
threshold t ∈ [0.40, 0.95] and a relative-to-best ratio r ∈ {0, 0.5}.
* Selected: **one-to-one assignment, t = 0.75, r = 0**.
* Every final match is a scored candidate, so matches ⊆ candidates by construction.

**France (no training labels):**
* The word statistics and duplicate counts are label-free, so they are computed on French data like
  on any other country.
* Missing French regions are filled from the city → region table of the clean Source-1 records
  (unambiguous cities only). This made the region fields of 455k French S2/S3 addresses look like
  the US/India records the model was trained on.
* On test, France gets 3.2 matches per entity and 6.1% empty predictions (US 3.4 / 5.8%).

---

## 5. Results & Error Analysis

Full report: `reports/error_analysis.md` (held-out group, 46,818 entities, 162,530 true pairs).

**F_0.5 Score (macro):** **0.98695** overall.
- By country: India 0.9874, US 0.9867.
- Precision 0.9968, recall 0.9664, singleton accuracy 0.989.

**Public leaderboard history:**

| version | change | validation | leaderboard |
|---|---|---|---|
| v1 | first pipeline | 0.9714 | 0.96200 |
| v2 | normalisation, blocking and feature improvements | 0.9860 | 0.97855 |
| v3 | further normalisation and feature improvements | 0.98636 | 0.97960 |
| v5 | same-street word statistics per split + expected-F rule | 0.98674 | 0.97806 |
| v6/v7 | one joint word-statistics table, threshold rule; France region fill and acronyms | 0.98695 | pending |

The leaderboard sits about 0.007 below validation. France has no labels, but it shows about twice
the US share of candidate records with an uncertain probability. We believe it accounts for most
of that gap.

**Where the true pairs go:**
- 96.6% found;
- 1.9% scored below the threshold;
- 1.2% never blocked;
- 0.3% lost in the one-to-one assignment.

**By candidate type:**
- Copies with an address are recalled at 98.5% (Latin script) and 98.9% (native script).
- Address-less copies are 4.4% of true pairs, but recall on them is only 53%, and they cause 60% of
  all misses. Most of them carry a name that several S1 entities share, so nothing in the record
  says which entity they belong to.

**Common false positives (wrong merges):** 493 of 157k predicted pairs (0.3%).
- 40% belong to a Source-1 entity that is absent (hidden), 21% to another entity, and 39% to no
  entity.
- Typical cases are decoy siblings with a nearly identical name a few doors away
  (`Piedmont Holdings Holdings`, 132 vs 1127 Cadillac Avenue), and brand-only names at the exact
  address of a different business.

**Common false negatives (missed matches):**
- address-less copies of names shared by several S1 entities;
- address-less copies with typos in several name words (`Christ Cahtdeibrgl`), which blocking
  cannot reach;
- copies whose address carries a different house number together with a renamed business word.

---

## 6. Conclusion

Most of the score comes from three things:
- careful normalisation of a heavily noised, multi-script source;
- blocking designed around measured recall misses;
- using the data's structure. This means the one-to-one constraint as both features and decision
  rule, plus label-free word statistics that tell copy noise from decoy siblings.

**What did not help:**
- An expected-F0.5 decision rule tied the threshold rule on validation but lost on the leaderboard.
- Word statistics computed separately per split shifted between train and test (v5), which is why
  we switched to one joint table.
- Blending the final model with an earlier one, and a separate threshold for address-less copies,
  both gave no gain on validation.
- A second-stage re-scoring model over each entity's full candidate list added only +0.0003.

**Remaining headroom:**
- about 0.004 in blocking misses;
- address-less copies of shared names, which are inherently ambiguous;
- French decoy siblings, which have to be judged without French labels.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:

```
src/run_pipeline.py              entry point: python -m src.run_pipeline
src/preprocessing/               normalisation (names, addresses, 9 Indian scripts), French region fill, EDA, mining of lookup tables
src/blocking/                    multi-pass key blocking with per-signal quotas
src/matching/                    features, word statistics, LightGBM training, scoring, decision rule
src/evaluation/                  F0.5 metric, validation split, error analysis, submission writer + validator, packager
models/                          trained matcher + tuned decision rule (lets `--from predict` skip training)
tests/                           69 unit tests
reports/                         EDA and error-analysis reports
README.md, requirements.txt
```

**Reproduce:**
1. `pip install -r requirements.txt` (on macOS also `brew install libomp`).
2. Place the data in `dataset/`.
3. Run `python -m src.run_pipeline`.

This takes about 2–2.5 h on an 8-core, 8 GB laptop. The shipped model was trained before the French
region fill and acronym rule were added. A full re-run retrains with them and recomputes the word
statistics on the filled French addresses, so a small number of decisions can differ from the
submitted file.

### B. Additional Results

* Normalisation effect on true matches (raw → normalised):
  - exact name match 10.7% → 42.1% (69.3% on the core name);
  - name token Jaccard 0.62 → 0.79; address token Jaccard 0.60 → 0.79.

  On random non-matches these stay at ≤ 0.06.
* Top features by gain: `block_score`, `block_rank`, `b_rank`, `addr_tsr`, `block_n_keys`,
  `street_ratio`, `heqs_extra_min`, `heq_extra_min`, `heq_extra_mean`, `b_score_gap`
  (`models/feature_importance.csv`).
