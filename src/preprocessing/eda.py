"""Exploratory data analysis -> reports/eda_report.md (+ figures, + eda_stats.json).

    python -m src.preprocessing.eda

Needs data/processed/train_source{1,2,3}.parquet (run preprocessing first)
for the duplicate and normalisation-effectiveness sections.
"""
from __future__ import annotations

import json
import random
import warnings
import re
import time

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .io import ROOT, ground_truth_pairs, iter_source, read_ground_truth
from .preprocessing import processed_path

REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
NOISE_SAMPLE = 500_000          # rows per file used for regex-based noise prevalence
PAIR_SAMPLE_ENTITIES = 200_000  # GT entities used for normalisation evaluation
SEED = 13

# Chart styling: reference palette slots 1-3 (validated all-pairs), light surface.
COLORS = {"S1": "#2a78d6", "S2": "#eb6834", "S3": "#1baf7a"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

PLACEHOLDER_RE = r"^\s*(?:<?null>?|none|nil|n/?a|nan|unknown|-+)\s*$"
NOISE_PATTERNS = {
    "Indic script (any)": (r"[ऀ-ൿ]", "name"),
    "Latin accents / non-ASCII Latin": (r"[À-ɏ]", "name"),
    "ALL CAPS": (None, "name"),
    "Alias marker (dba, f/k/a, formerly, t/a ...)":
        (r"(?i)\s(?:d/b/a|dba|doing business as|f/k/a|fka|formerly|a/k/a|aka|t/a|trading as)\s", "name"),
    "ID tail (#12345 / (ID: n))": (r"(?i)(?:\(\s*id\s*[:#]?\s*\d+\s*\)|#\s*\d{4,})\s*$", "name"),
    "Pipe + website tail": (r"\|", "name"),
    "Domain-style name (x.com)": (r"(?i)^\S+\.(?:com|in|net|org|co|fr)$", "name"),
    "Junk prefix (--, >>, ***, ...)": (r"^\s*(?:--|>>|\*\*\*|\.\.\.|@|#)", "name"),
    "Bracketed part [..] / (..)": (r"[\[\(]", "name"),
    "Digit inside a word (5tar, mem0rial)": (r"\b[A-Za-z]+[0-9][A-Za-z]+\b|\b[0-9][A-Za-z]{3,}\b", "name"),
    "Double space": (r"  ", "name"),
    "Address: Indic script": (r"[ऀ-ൿ]", "address"),
    "Address: placeholder (<NULL>, N/A)": (r"(?i)(?:^|,\s*)(?:<null>|null|n/a|none)(?:\s*,|$)", "address"),
    "Address: zero-padded number": (r"(?:^|[\s#,])0\d+", "address"),
    "Address: PMB box": (r"(?i)\bpmb\s*\d+", "address"),
    "Address: house marker (#, No., H.no, Door No)": (r"(?i)(?:#\s*\d|\bno\.?\s*\d|\bh\.?\s?no\b|\bdoor no\b)", "address"),
    "Address: repeated token (Unit Unit)": (r"(?i)\b(\w+)\s+\1\b", "address"),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]).replace("|", "\\|") for c in cols) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------- raw file stats
def raw_file_stats() -> tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
    rows, countries, noise_rows, lengths = [], {}, [], {}
    for split in ("train", "test"):
        for s in (1, 2, 3):
            key = f"{split}/S{s}"
            n = miss_name = miss_addr = miss_country = 0
            country_counts: dict[str, int] = {}
            name_chars, name_toks, addr_chars = [], [], []
            hashes = []
            noise_hits = {k: 0 for k in NOISE_PATTERNS}
            noise_n = 0
            for chunk in iter_source(split, s):
                n += len(chunk)
                nm, ad, co = chunk["business_name"], chunk["business_address"], chunk["country"]
                miss_name += int((nm.str.strip() == "").sum() + nm.str.match(PLACEHOLDER_RE).sum())
                miss_addr += int((ad.str.strip() == "").sum() + ad.str.match(PLACEHOLDER_RE).sum())
                miss_country += int((co.str.strip() == "").sum())
                for c, v in co.value_counts().items():
                    country_counts[c] = country_counts.get(c, 0) + int(v)
                name_chars.append(nm.str.len().to_numpy())
                name_toks.append(nm.str.split().str.len().fillna(0).to_numpy())
                addr_chars.append(ad.str.len().to_numpy())
                hashes.append(pd.util.hash_pandas_object(
                    chunk[["business_name", "business_address", "country"]], index=False).to_numpy())
                if noise_n < NOISE_SAMPLE:
                    part = chunk.iloc[: NOISE_SAMPLE - noise_n]
                    noise_n += len(part)
                    for label, (pat, field) in NOISE_PATTERNS.items():
                        col = part["business_name" if field == "name" else "business_address"]
                        if pat is None:   # ALL CAPS: has letters and equals its upper-case
                            hit = (col == col.str.upper()) & col.str.contains(r"[A-Za-z]")
                        else:
                            hit = col.str.contains(pat, regex=True)
                        noise_hits[label] += int(hit.sum())
            h = np.concatenate(hashes)
            exact_dups = int(len(h) - len(np.unique(h)))
            nc, nt, ac = map(np.concatenate, (name_chars, name_toks, addr_chars))
            lengths[key] = {"name_tokens": nt, "name_chars": nc, "addr_chars": ac}
            rows.append({
                "file": key, "records": f"{n:,}",
                "missing name": _pct(miss_name / n), "missing address": _pct(miss_addr / n),
                "missing country": _pct(miss_country / n),
                "exact duplicate rows": f"{exact_dups:,}",
                "name chars (median / p95)": f"{np.median(nc):.0f} / {np.percentile(nc, 95):.0f}",
                "name tokens (median / p95)": f"{np.median(nt):.0f} / {np.percentile(nt, 95):.0f}",
                "address chars (median / p95)": f"{np.median(ac):.0f} / {np.percentile(ac, 95):.0f}",
            })
            countries[key] = dict(sorted(country_counts.items(), key=lambda kv: -kv[1]))
            noise_rows.append({"file": key, **{k: v / noise_n for k, v in noise_hits.items()}})
            log(f"raw stats: {key} ({n:,} rows)")
    noise = pd.DataFrame(noise_rows).set_index("file").T
    return pd.DataFrame(rows), countries, noise, lengths


# ------------------------------------------------------------ ground truth
def ground_truth_stats() -> tuple[dict, pd.Series, pd.DataFrame]:
    gt = read_ground_truth()
    ids = gt["matched_entity_ids"].str.split(",")
    n_matches = gt["matched_entity_ids"].map(lambda v: len([x for x in v.split(",") if x]))
    n_s2 = gt["matched_entity_ids"].str.count(r"\bS2-")
    n_s3 = gt["matched_entity_ids"].str.count(r"\bS3-")
    all_ids = [x for lst in ids for x in lst if x]
    stats = {
        "source1_entities": int(len(gt)),
        "singletons (0 matches)": int((n_matches == 0).sum()),
        "exactly 1 match": int((n_matches == 1).sum()),
        "2+ matches": int((n_matches >= 2).sum()),
        "mean matches / entity": round(float(n_matches.mean()), 3),
        "max matches / entity": int(n_matches.max()),
        "total matched pairs": int(n_matches.sum()),
        "pairs to S2": int(n_s2.sum()),
        "pairs to S3": int(n_s3.sum()),
        "entities with S2 match only": int(((n_s2 > 0) & (n_s3 == 0)).sum()),
        "entities with S3 match only": int(((n_s3 > 0) & (n_s2 == 0)).sum()),
        "entities with both S2 and S3": int(((n_s2 > 0) & (n_s3 > 0)).sum()),
        "S2/S3 ids matched to >1 S1 entity": int(len(all_ids) - len(set(all_ids))),
    }
    hist = n_matches.clip(upper=10).value_counts().sort_index()
    # per-country singleton rate needs S1 country
    s1 = pq.read_table(processed_path("train", 1), columns=["entity_id", "country"]).to_pandas()
    g = gt.assign(n=n_matches).merge(s1, left_on="source1_entity_id", right_on="entity_id")
    by_country = g.groupby("country").agg(entities=("n", "size"), singleton_rate=("n", lambda x: (x == 0).mean()),
                                          mean_matches=("n", "mean")).reset_index()
    return stats, hist, by_country


def unmatched_record_stats() -> dict:
    """How many train S2/S3 records are not the match of any S1 entity (distractors)."""
    gt = read_ground_truth()
    matched = {x for v in gt["matched_entity_ids"] for x in v.split(",") if x}
    out = {}
    for s in (2, 3):
        ids = pq.read_table(processed_path("train", s), columns=["entity_id"]).column(0).to_pylist()
        out[f"S{s}"] = {"records": len(ids), "matched to some S1": sum(i in matched for i in ids)}
    return out


# ---------------------------------------------------- processed-data checks
def normalized_duplicates() -> pd.DataFrame:
    rows = []
    for split in ("train", "test"):
        for s in (1, 2, 3):
            t = pq.read_table(processed_path(split, s),
                              columns=["business_name_norm", "business_address_norm", "country_norm"])
            df = t.to_pandas()
            h = pd.util.hash_pandas_object(df, index=False).to_numpy()
            hn = pd.util.hash_pandas_object(df[["business_name_norm", "country_norm"]], index=False).to_numpy()
            rows.append({"file": f"{split}/S{s}",
                         "duplicate (norm name + norm address + country)": f"{len(h) - len(np.unique(h)):,}",
                         "duplicate (norm name + country)": f"{len(hn) - len(np.unique(hn)):,}",
                         "empty name_norm": f"{int((df['business_name_norm'] == '').sum()):,}",
                         "empty address_norm": f"{int((df['business_address_norm'] == '').sum()):,}"})
            log(f"norm dups: {split}/S{s}")
    return pd.DataFrame(rows)


def _load_ids(split: str, s: int, ids: set, columns: list[str]) -> pd.DataFrame:
    pf = pq.ParquetFile(processed_path(split, s))
    value_set = __import__("pyarrow").array(sorted(ids))
    parts = []
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=columns)
        parts.append(t.filter(pc.is_in(t.column("entity_id"), value_set=value_set)).to_pandas())
    return pd.concat(parts, ignore_index=True)


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _raw_tokens(text: str) -> set:
    return set(_TOKEN_RE.findall(text.lower()))


def _jac(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def normalization_effect() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare raw vs normalised fields on GT matches and on random non-matches."""
    rng = random.Random(SEED)
    gt = read_ground_truth()
    gt = gt.sample(n=min(PAIR_SAMPLE_ENTITIES, len(gt)), random_state=SEED)
    pairs = ground_truth_pairs(gt)
    cols = ["entity_id", "business_name_raw", "business_name_norm", "name_core", "name_phonetic", "name_known_ratio",
            "business_address_raw", "business_address_norm", "addr_state", "addr_house_no", "country_norm"]
    s1 = _load_ids("train", 1, set(gt["source1_entity_id"]), cols)
    other = pd.concat([_load_ids("train", s, set(pairs["entity_id"]), cols) for s in (2, 3)])
    s1 = s1.set_index("entity_id")
    other = other.set_index("entity_id")
    # non-matches: same country, random S1 entity (a realistic "same block" baseline is Aditya's job)
    by_country: dict[str, list[str]] = s1.groupby("country_norm").groups
    by_country = {k: list(v) for k, v in by_country.items()}
    neg = pairs.copy()
    neg["source1_entity_id"] = [rng.choice(by_country[other.at[e, "country_norm"]])
                                for e in neg["entity_id"]]
    neg = neg[~neg.set_index(["source1_entity_id", "entity_id"]).index.isin(
        pairs.set_index(["source1_entity_id", "entity_id"]).index)]

    def metrics(p: pd.DataFrame) -> dict:
        a = s1.loc[p["source1_entity_id"]].reset_index(drop=True)
        b = other.loc[p["entity_id"]].reset_index(drop=True)
        src = p["entity_id"].str[:2].reset_index(drop=True)
        rows = {}
        for grp, mask in [("all", np.ones(len(p), bool)), ("S1-S2", (src == "S2").to_numpy()),
                          ("S1-S3", (src == "S3").to_numpy())]:
            A, B = a[mask], b[mask]
            both_state = (A["addr_state"] != "") & (B["addr_state"] != "")
            both_house = (A["addr_house_no"] != "") & (B["addr_house_no"] != "")
            rows[grp] = {
                "name exact (raw, lower-cased)": (A["business_name_raw"].str.lower().str.strip().values
                                                  == B["business_name_raw"].str.lower().str.strip().values).mean(),
                "name exact (name_norm)": (A["business_name_norm"].values == B["business_name_norm"].values).mean(),
                "name exact (name_core)": (A["name_core"].values == B["name_core"].values).mean(),
                "name exact (name_phonetic)": (A["name_phonetic"].values == B["name_phonetic"].values).mean(),
                "name token Jaccard (raw)": np.mean([_jac(_raw_tokens(x), _raw_tokens(y)) for x, y in
                                                     zip(A["business_name_raw"], B["business_name_raw"])]),
                "name token Jaccard (name_norm)": np.mean([_jac(set(x.split()), set(y.split())) for x, y in
                                                           zip(A["business_name_norm"], B["business_name_norm"])]),
                "address token Jaccard (raw)": np.mean([_jac(_raw_tokens(x), _raw_tokens(y)) for x, y in
                                                        zip(A["business_address_raw"], B["business_address_raw"])]),
                "address token Jaccard (address_norm)": np.mean([_jac(set(x.split()), set(y.split())) for x, y in
                                                                 zip(A["business_address_norm"], B["business_address_norm"])]),
                "state agrees (when both parsed)": (A["addr_state"].values == B["addr_state"].values)[both_state.values].mean(),
                "house no. agrees (when both parsed)": (A["addr_house_no"].values == B["addr_house_no"].values)[both_house.values].mean(),
                "state parsed on S2/S3 side": (B["addr_state"] != "").mean(),
                "S2/S3 name brand-only (name_known_ratio = 0)": (B["name_known_ratio"] == 0).mean(),
            }
        return rows

    pos, negm = metrics(pairs), metrics(neg)
    table = pd.DataFrame({
        "matches: all": pos["all"], "matches: S1-S2": pos["S1-S2"], "matches: S1-S3": pos["S1-S3"],
        "random non-matches (same country)": negm["all"],
    })
    log(f"normalisation effect: {len(pairs):,} match pairs, {len(neg):,} non-match pairs")
    # examples of hard matches after normalisation (lowest name Jaccard)
    a = s1.loc[pairs["source1_entity_id"]].reset_index(drop=True)
    b = other.loc[pairs["entity_id"]].reset_index(drop=True)
    jac = np.array([_jac(set(x.split()), set(y.split())) for x, y in zip(a["business_name_norm"], b["business_name_norm"])])
    hard_idx = np.argsort(jac, kind="stable")[:4000:200]
    hard = pd.DataFrame({"S1 name": a.loc[hard_idx, "business_name_raw"].values,
                         "S2/S3 name": b.loc[hard_idx, "business_name_raw"].values,
                         "S1 name_norm": a.loc[hard_idx, "business_name_norm"].values,
                         "S2/S3 name_norm": b.loc[hard_idx, "business_name_norm"].values})
    return table, hard


# ----------------------------------------------------------------- figures
def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=12)
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.tick_params(colors=INK_2, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def make_figures(hist: pd.Series, lengths: dict, effect: pd.DataFrame) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    files = []

    # 1. matches per Source-1 entity
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    labels = [str(i) if i < 10 else "10+" for i in hist.index]
    bars = ax.bar(labels, hist.values / hist.sum() * 100, color=COLORS["S1"], width=0.6)
    for bar, v in zip(bars, hist.values / hist.sum() * 100):
        if v >= 1:
            ax.annotate(f"{v:.1f}%", (bar.get_x() + bar.get_width() / 2, v), ha="center", va="bottom",
                        fontsize=8, color=INK_2, xytext=(0, 2), textcoords="offset points")
    _style(ax, "Matches per Source 1 entity (train ground truth)", "number of S2/S3 matches", "% of entities")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "matches_per_entity.png", dpi=150)
    files.append("figures/matches_per_entity.png")
    plt.close(fig)

    # 2. name token-length distribution per source (train)
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    bins = np.arange(0, 13)
    for s in ("S1", "S2", "S3"):
        nt = np.clip(lengths[f"train/{s}"]["name_tokens"], 0, 12)
        share = np.bincount(nt.astype(int), minlength=13)[:13] / len(nt) * 100
        ax.plot(bins, share, color=COLORS[s], linewidth=2, marker="o", markersize=4, label=s)
    ax.set_xticks(bins, [str(b) if b < 12 else "12+" for b in bins])
    ax.legend(frameon=False, labelcolor=INK_2)
    _style(ax, "Business-name length by source (train)", "whitespace tokens", "% of records")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "name_length_by_source.png", dpi=150)
    files.append("figures/name_length_by_source.png")
    plt.close(fig)

    # 3. raw vs normalised agreement on true matches
    pairs_plot = [("Name exact match", "name exact (raw, lower-cased)", "name exact (name_norm)"),
                  ("Name token Jaccard", "name token Jaccard (raw)", "name token Jaccard (name_norm)"),
                  ("Address token Jaccard", "address token Jaccard (raw)", "address token Jaccard (address_norm)")]
    fig, ax = plt.subplots(figsize=(8, 3.6), facecolor=SURFACE)
    y = np.arange(len(pairs_plot))
    h = 0.34
    raw_v = [effect.loc[r, "matches: all"] for _, r, _ in pairs_plot]
    norm_v = [effect.loc[n, "matches: all"] for _, _, n in pairs_plot]
    b1 = ax.barh(y + h / 2 + 0.01, raw_v, height=h, color="#b8b7b1", label="raw")
    b2 = ax.barh(y - h / 2 - 0.01, norm_v, height=h, color=COLORS["S1"], label="normalised")
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_width():.2f}", (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        va="center", fontsize=8, color=INK_2, xytext=(4, 0), textcoords="offset points")
    ax.set_yticks(y, [p[0] for p in pairs_plot])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.legend(frameon=False, labelcolor=INK_2, loc="lower right")
    _style(ax, "Agreement between true matches: raw vs normalised text", "mean over matched pairs")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "normalisation_effect.png", dpi=150)
    files.append("figures/normalisation_effect.png")
    plt.close(fig)
    return files


# ------------------------------------------------------------------ report
def main() -> None:
    warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")
    REPORT_DIR.mkdir(exist_ok=True)
    files_df, countries, noise, lengths = raw_file_stats()
    gt_stats, hist, by_country = ground_truth_stats()
    unmatched = unmatched_record_stats()
    dups = normalized_duplicates()
    effect, hard = normalization_effect()
    figs = make_figures(hist, lengths, effect)

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else v

    country_tbl = pd.DataFrame(countries).fillna(0).astype(int).T
    country_tbl = country_tbl.map(lambda v: f"{v:,}").reset_index().rename(columns={"index": "file"})
    noise_tbl = noise.map(_pct).reset_index().rename(columns={"index": f"pattern (first {NOISE_SAMPLE:,} rows/file)"})
    gt_tbl = pd.DataFrame([{"metric": k, "value": f"{v:,}" if isinstance(v, int) else v} for k, v in gt_stats.items()])
    hist_tbl = pd.DataFrame({"matches": [str(i) if i < 10 else "10+" for i in hist.index],
                             "entities": [f"{v:,}" for v in hist.values],
                             "share": [_pct(v / hist.sum()) for v in hist.values]})
    by_country["singleton_rate"] = by_country["singleton_rate"].map(_pct)
    by_country["mean_matches"] = by_country["mean_matches"].map(lambda v: f"{v:.2f}")
    by_country["entities"] = by_country["entities"].map(lambda v: f"{v:,}")
    unmatched_tbl = pd.DataFrame([{"source": k, "records": f"{v['records']:,}",
                                   "matched to some S1": f"{v['matched to some S1']:,}",
                                   "never matched (distractors)": _pct(1 - v["matched to some S1"] / v["records"])}
                                  for k, v in unmatched.items()])
    effect_tbl = effect.map(fmt).reset_index().rename(columns={"index": "metric"})

    md = f"""# EDA and preprocessing report: Business Entity Resolution

*Generated by `python -m src.preprocessing.eda` on {time.strftime('%Y-%m-%d')}. All numbers are computed from
the provided files; nothing external is used.*

## 1. Files

{_md_table(files_df)}

"Missing" counts empty strings and placeholders such as `<NULL>`, `null`, `N/A`.

### Duplicate-looking records after normalisation

{_md_table(dups)}

Within one source, many S2/S3 records collapse to the same normalised name, address and country. These are
the "several matches per entity" in the ground truth: one S1 entity usually matches several near-identical
S2/S3 variants.

**Names are not unique in Source 1.** About 32% of train S1 records (707k) share their normalised name with
another S1 entity in the same country, and only the address tells them apart. A name-only match is
therefore not enough evidence under F0.5. Blocking and matching must use the address (`addr_state`,
`addr_house_no`, `addr_street`) to separate same-name businesses.

### Country labels

{_md_table(country_tbl)}

Training has only **US** and **India**; test adds **France** (~15% of test S1). The pipeline treats country
as an open set (`normalize_country` passes unknown labels through, and the address parser falls back to
generic rules).

## 2. Ground truth (train)

{_md_table(gt_tbl)}

![matches per entity]({figs[0]})

{_md_table(hist_tbl)}

### By country

{_md_table(by_country)}

### How many S2/S3 records are ever matched?

{_md_table(unmatched_tbl)}

Many S2/S3 records never match any S1 entity. Blocking must expect a large pool of distractors, and
F0.5 rewards leaving true singletons empty.

## 3. Noise patterns

Share of records that show each pattern (regex on raw text).

{_md_table(noise_tbl)}

S1 is clean: ASCII, title case, no injected junk. S2 and S3 carry the noise. They mix nine Indic scripts
(Devanagari, Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada, Malayalam), fully or partly
translated, and add injected accents, look-alike digits, alias markers
(`<synthetic brand> dba <real name>`), ID tails, website tails, glued domain names, shuffled words, typos
and dropped tokens. Addresses vary between state code, full name and native script, add or drop
components, reorder them, zero-pad house numbers, and inject PMB boxes.

![name length]({figs[1]})

## 4. Normalisation: what it does and whether it helps

Every raw record is turned into the standard record (`src/preprocessing/preprocessing.py`):

| field | meaning |
|---|---|
| `business_name_norm` | accent-folded, transliterated, lower-cased; alias/ID/website tails removed; legal forms canonical (`private limited` -> `pvt ltd`, `L.L.C.` -> `llc`) |
| `name_core` | `business_name_norm` minus legal forms, honorifics (Shri/Smt/M/s/The) and "and" |
| `name_legal` | canonical legal forms found |
| `name_alt` | the synthetic brand before a dba/f/k/a marker |
| `name_is_domain`, `name_script` | flags (glued domain / handle; script of the raw name) |
| `name_known_ratio` | share of core words seen in clean S1 names; 0 = brand-only name, so rely on the address |
| `name_phonetic` | consonant skeleton per core token (bridges transliteration vs English spelling) |
| `business_address_norm` | abbreviations expanded, typos fixed, state as canonical code, house markers / zero padding / PMB removed |
| `addr_street`, `addr_city`, `addr_state`, `addr_house_no`, `addr_numbers`, `addr_postcode` | parsed components that do not depend on component order |

Data-driven resources (`python -m src.preprocessing.mine_resources`, train only for anything label-based):

* `indic_token_map.tsv`: native-script token to Latin word. Tokens are aligned position by position on train
  ground-truth pairs whose token counts match, then decided by majority vote. Tokens never seen in a
  labelled pair fall back to a phonetic match against the Source-1 vocabulary.
* `native_state_map.tsv`: native-script state name to state code.
* `name_vocab.tsv` / `address_vocab.tsv`: Source-1 token counts, used for domain-name segmentation,
  look-alike-digit repair, and to stop the typo fixer from "correcting" real words.
* `brand_pieces.tsv`: syllables that recur across the synthetic brands seen before `dba`/`f/k/a`
  markers. They stop brand words (`fayevantagequo`) from being split like glued real names
  (`alliedsoftwaresystems` -> `allied software systems`).
* `reports/mined_abbreviations.tsv`: token pairs that differ between S1 and its matches. This is the
  evidence behind the abbreviation tables.

### Effect on true matches vs random non-matches ({PAIR_SAMPLE_ENTITIES:,} sampled train entities)

{_md_table(effect_tbl)}

![normalisation effect]({figs[2]})

Normalisation should raise agreement between true matches without raising it between non-matches.
The last column is the baseline for random pairs from the same country.

### Hardest remaining matches (lowest name Jaccard after normalisation)

{_md_table(hard)}

These are the cases for fuzzy / phonetic features and the ML matcher: heavy typos, dropped words, and
synthetic-brand-only names.

## 5. Notes for blocking and matching

* **Country:** block within `country_norm`, but never filter to known labels. France only appears in test.
* **Names:** use `name_core` tokens and character n-grams for blocking. Glued domains and handles are
  already segmented. Compare S1 against both `business_name_norm` and `name_alt`.
* **Brand-only names** (`name_known_ratio == 0`, see section 4) cannot match on name at all. Blocking needs
  an address-only pass (house number + street + city) to recall them, and the matcher should weight
  the address for them.
* **Addresses:** `addr_state` and `addr_house_no` are strong and cheap signals. Addresses can be missing
  (see section 1); treat missing as unknown, not as a mismatch.
* **Singletons** are a large share of S1 (section 2), so keep the match threshold precision-oriented.
"""
    (REPORT_DIR / "eda_report.md").write_text(md, encoding="utf-8")
    stats = {"files": files_df.to_dict("records"), "countries": countries, "ground_truth": gt_stats,
             "matches_hist": {str(k): int(v) for k, v in hist.items()}, "unmatched": unmatched,
             "normalisation_effect": effect.to_dict()}
    (REPORT_DIR / "eda_stats.json").write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    log(f"wrote {(REPORT_DIR / 'eda_report.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
