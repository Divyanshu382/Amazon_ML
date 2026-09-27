# Preprocessing (raw TSV -> standardised records)

Owner: Divyanshu. Used by blocking (Aditya) and matching (Santosh).

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# 1. mine lookup tables from the provided data (about 3 min; outputs are committed in resources/)
.venv/bin/python -m src.preprocessing.mine_resources
# 2. normalise every file -> data/processed/{split}_source{n}.parquet (about 8 min on 7 cores)
.venv/bin/python -m src.preprocessing.preprocessing
# 3. EDA report -> reports/eda_report.md (+ figures, eda_stats.json)
.venv/bin/python -m src.preprocessing.eda
# tests
.venv/bin/python -m pytest tests -q
```

`--split test --sources 1` limits the run to one file; `--sample 50000` runs a quick check.

## Use from other modules

```python
from src.preprocessing import load_processed, normalize_name, normalize_address, preprocess_record

s1 = load_processed("test", 1)                          # DataFrame, one row per record
s2 = load_processed("test", 2, columns=["entity_id", "name_core", "addr_state"])

normalize_name("ABC Technologies Pvt. Ltd.")            # 'abc technologies pvt ltd'
normalize_address("2238 WILLIAMSBURG RD, GLEN DALE, WV", "US")  # '2238 williamsburg road glen dale wv'
```

## Output columns

| column | example (S2 record) | notes |
|---|---|---|
| `entity_id`, `source` | `S2-164974600`, `S2` | |
| `business_name_raw` | `Apex పవర్ ప్రైవేట్ లిమిటెడ్` | untouched |
| `business_name_norm` | `apex power pvt ltd` | canonical full name |
| `name_core` | `apex power` | minus legal forms, honorifics (Shri, Smt, M/s, Dr, The) and "and"; never empty if the name is not |
| `name_legal` | `ltd pvt` | sorted canonical legal forms |
| `name_alt` | `nexsolzeph` | part before a `dba` / `f/k/a` / `formerly` marker (a synthetic brand in this data) |
| `name_is_domain` | `False` | name contained a glued domain / handle (`tristateguild.com`, `#rajkotbiotech`); it is segmented to `tri state guild` / `rajkot biotech` |
| `name_known_ratio` | `1.0` | share of `name_core` words seen in clean S1 names; `0.0` flags brand-only or garbled names (`Drexvio`, `Korkelo`): rely on the address for these |
| `name_script` | `mixed` | `latin`, `indic:<script>`, `mixed`, `other` (raw name) |
| `name_phonetic` | `apks pvr` | consonant skeleton per `name_core` token |
| `business_address_raw` | `SY NO 789, PLOT NO 4, SANGAREDDY, Andhra Pradesh` | untouched |
| `business_address_norm` | `sy 789 plot 4 sangareddy ap` | |
| `country`, `country_norm` | `India`, `india` | open set: unknown labels pass through lower-cased |
| `addr_street` | `sy 789` | first component with a number |
| `addr_city` | `sangareddy` | last non-state component without a number |
| `addr_state` | `ap` | per-country code (`tx`, `ka`, French region `hdf`); `india:ka` if the country is unknown |
| `addr_house_no` | `789` | first house-number-like token, zero padding removed (`00478/1` -> `478/1`) |
| `addr_numbers` | `4 789` | all digit runs, unique, space-separated |
| `addr_postcode` | `` | 6-digit PIN / 5-digit ZIP / French CP when present |
| `addr_is_missing` | `False` | empty or placeholder (`<NULL>`, `N/A`) address |

## Files

| file | what |
|---|---|
| `io.py` | TSV reading (always `sep="\t"`, no quoting, no NA conversion), ground-truth parsing |
| `text_utils.py` | Unicode cleanup, accent folding, script detection, Indic transliteration (one table for all nine Brahmic scripts), phonetic key |
| `name_normalizer.py` | `normalize_name`, `parse_name` |
| `address_normalizer.py` | `normalize_address`, `parse_address` |
| `gazetteer.py` | country aliases, per-country state/region tables, renamed cities (hand-written constants) |
| `resources.py` / `resources/` | loaders and the mined tables: `indic_token_map`, `native_state_map`, `name_vocab`, `address_vocab`, `brand_pieces` |
| `mine_resources.py` | builds `resources/` from the provided data |
| `preprocessing.py` | parallel, chunked driver, `preprocess_record`, `load_processed` |
| `eda.py` | builds `reports/eda_report.md` |

## Fair-play notes

* Everything is derived from the provided files. There are no APIs, no geocoding, and no downloaded
  datasets.
* Label-based mining (`indic_token_map` alignment, `native_state_map`) uses **train** ground truth only.
  Test files are read only as unlabelled text: Source-1 vocabulary counts, the list of native-script
  tokens for the phonetic fallback, and synthetic-brand syllables.
* `gazetteer.py` holds generic spelling conventions typed in by hand: postal state codes, French
  region/department names, renamed Indian cities. It contains no business data.
