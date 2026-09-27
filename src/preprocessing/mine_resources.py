"""Mine the data-driven lookup tables used by the normalizers.

    python -m src.preprocessing.mine_resources            # all tables
    python -m src.preprocessing.mine_resources --only vocab

Outputs (src/preprocessing/resources/):
  name_vocab.tsv        clean Source-1 name tokens (train + test S1; no labels used)
  address_vocab.tsv     clean Source-1 address tokens (train + test S1; no labels used)
  brand_pieces.tsv      substrings recurring across synthetic brand names (the part
                        before "dba"/"f/k/a" in S2/S3 names; text only, no labels)
  indic_token_map.tsv   native-script token -> Latin token. Primarily aligned on
                        TRAIN ground-truth pairs (majority vote); tokens never
                        seen in a labelled pair fall back to the closest
                        Source-1 vocabulary word with the same phonetic key
  native_state_map.tsv  native-script state component -> (country, state code), TRAIN only
and reports/mined_abbreviations.tsv (evidence for the hand-curated
abbreviation tables in address_normalizer.py / name_normalizer.py).

Only the provided challenge files are used -- no external data.
"""
from __future__ import annotations

import argparse
import collections
import time

import pandas as pd

from .gazetteer import canonical_state, normalize_country
from .io import ROOT, ground_truth_pairs, iter_source, read_ground_truth, read_source
from .name_normalizer import ALIAS_MARKER_RE, basic_name_tokens, split_alias
from .resources import (ADDRESS_VOCAB, BRAND_PIECES, INDIC_TOKEN_MAP, NAME_VOCAB, NATIVE_STATE_MAP,
                        RESOURCE_DIR)
from .resources import name_vocab
from .text_utils import clean_unicode, fold_accents, has_indic, phonetic_key, transliterate_indic

INDIC_PATTERN = r"[ऀ-ൿ]"
_PUNCT = ".,;:()[]{}\"'!?|-_/\\*#@<>+&"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _write_tsv(path, header: list[str], rows) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
            n += 1
    return n


# ------------------------------------------------------------------ vocab
def mine_vocab(min_count: int = 2) -> None:
    names: collections.Counter = collections.Counter()
    addrs: collections.Counter = collections.Counter()
    for split in ("train", "test"):
        for chunk in iter_source(split, 1, usecols=["business_name", "business_address"]):
            for name in chunk["business_name"]:
                names.update(t for t in basic_name_tokens(name) if t.isalpha() and len(t) > 1)
            for addr in chunk["business_address"]:
                addrs.update(t for t in basic_name_tokens(addr) if t.isalpha() and len(t) > 1)
        log(f"vocab: scanned {split} source1")
    for path, counts in ((NAME_VOCAB, names), (ADDRESS_VOCAB, addrs)):
        rows = [(w, c) for w, c in counts.most_common() if c >= min_count]
        n = _write_tsv(path, ["token", "count"], rows)
        log(f"vocab: wrote {n:,} tokens -> {path.relative_to(ROOT)}")


def mine_brand_pieces(min_brands: int = 50) -> None:
    """Synthetic brands ('Nexsolzeph dba ...') are glued from a small syllable set. Count, for
    every 3-6 letter substring, how many distinct brand words contain it."""
    brands: set[str] = set()
    for split in ("train", "test"):
        for s in (2, 3):
            for chunk in iter_source(split, s, usecols=["business_name"]):
                names = chunk["business_name"]
                for name in names[names.str.contains(ALIAS_MARKER_RE)]:
                    alt = split_alias(clean_unicode(name))[1]
                    brands.update(t for t in basic_name_tokens(alt) if t.isalpha() and len(t) >= 6)
    df: collections.Counter = collections.Counter()
    for b in brands:
        df.update({b[i:i + n] for n in range(3, 7) for i in range(len(b) - n + 1)})
    rows = [(p, c) for p, c in df.most_common() if c >= min_brands]
    n = _write_tsv(BRAND_PIECES, ["piece", "brands"], rows)
    log(f"brands: {len(brands):,} brand words, {n:,} recurring pieces -> {BRAND_PIECES.relative_to(ROOT)}")


# ----------------------------------------------------- labelled pair loading
def _load_indic_pairs() -> pd.DataFrame:
    """Train GT pairs where the S2/S3 record has Indic text in name or address."""
    parts = []
    for s in (2, 3):
        for chunk in iter_source("train", s):
            mask = chunk["business_name"].str.contains(INDIC_PATTERN) | \
                chunk["business_address"].str.contains(INDIC_PATTERN)
            parts.append(chunk[mask])
        log(f"pairs: scanned train source{s}")
    other = pd.concat(parts, ignore_index=True)
    pairs = ground_truth_pairs(read_ground_truth())
    pairs = pairs.merge(other, on="entity_id")
    s1 = read_source("train", 1).rename(columns=lambda c: c if c == "entity_id" else c + "_s1")
    pairs = pairs.merge(s1, left_on="source1_entity_id", right_on="entity_id", suffixes=("", "_y"))
    log(f"pairs: {len(pairs):,} labelled pairs with Indic text")
    return pairs


def _other_tokens(name: str) -> list[str]:
    """Whitespace tokens of a (possibly mixed-script) name; Latin ones lightly normalised."""
    out = []
    for tok in clean_unicode(name).split():
        tok = tok.strip(_PUNCT)
        if not tok:
            continue
        if has_indic(tok):
            out.append(tok)
        else:
            out.extend(basic_name_tokens(tok))
    return out


def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _native_name_tokens() -> collections.Counter:
    """All Indic tokens in S2/S3 business names (train + test; text only, no labels)."""
    counts: collections.Counter = collections.Counter()
    for split in ("train", "test"):
        for s in (2, 3):
            for chunk in iter_source(split, s, usecols=["business_name"]):
                names = chunk["business_name"]
                for name in names[names.str.contains(INDIC_PATTERN)]:
                    counts.update(t for t in (w.strip(_PUNCT) for w in clean_unicode(name).split())
                                  if has_indic(t))
    return counts


def _phonetic_fallback(tokens, min_vocab_count: int = 20) -> list[tuple]:
    """Map a native token to the Source-1 vocabulary word sharing its phonetic key
    (closest by edit distance to the romanisation, then most frequent)."""
    by_key: dict[str, list[tuple[int, str]]] = collections.defaultdict(list)
    for w, c in name_vocab().counts.items():
        if c >= min_vocab_count:
            by_key[phonetic_key(w)].append((c, w))
    rows = []
    for tok in tokens:
        roman = transliterate_indic(tok)
        key = phonetic_key(roman)
        cands = by_key.get(key, [])
        if len(key) < 3 or not cands:
            continue
        dist, neg_c, best = min((_levenshtein(roman, w), -c, w) for c, w in cands)
        if dist <= max(2, len(best) // 3):
            rows.append((tok, best, -neg_c, 0.0, "phonetic"))
    return rows


def mine_indic_tokens(pairs: pd.DataFrame, min_count: int = 2, min_share: float = 0.5) -> None:
    votes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    aligned = 0
    for s1_name, name in zip(pairs["business_name_s1"], pairs["business_name"]):
        if not has_indic(name):
            continue
        a = basic_name_tokens(s1_name)
        b = _other_tokens(name)
        if len(a) != len(b):
            continue
        aligned += 1
        for x, y in zip(b, a):
            if has_indic(x) and y.isalpha():
                votes[x][y] += 1
    rows = []
    for tok, c in votes.items():
        best, n = c.most_common(1)[0]
        total = sum(c.values())
        if n >= min_count and n / total >= min_share:
            rows.append((tok, best, n, round(n / total, 3), "aligned"))
    rows.sort(key=lambda r: -r[2])
    n_aligned = len(rows)
    known = {r[0] for r in rows}
    unseen = [t for t, _ in _native_name_tokens().most_common() if t not in known]
    fallback = _phonetic_fallback(unseen)
    n = _write_tsv(INDIC_TOKEN_MAP, ["native_token", "latin_token", "count", "share", "method"],
                   rows + fallback)
    log(f"indic: {aligned:,} aligned names -> {n_aligned:,} aligned tokens; "
        f"{len(fallback):,}/{len(unseen):,} unseen tokens mapped phonetically; "
        f"wrote {n:,} -> {INDIC_TOKEN_MAP.relative_to(ROOT)}")


def mine_native_states(pairs: pd.DataFrame, min_count: int = 3) -> None:
    votes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for addr_s1, addr, country in zip(pairs["business_address_s1"], pairs["business_address"],
                                      pairs["country_s1"]):
        country = normalize_country(country)
        comps = [c.strip() for c in clean_unicode(addr).split(",") if c.strip()]
        native = [c for c in comps if has_indic(c)]
        if not native:
            continue
        codes = []
        for c in addr_s1.split(","):
            key = " ".join("".join(ch if ch.isalnum() else " " for ch in fold_accents(c).lower()).split())
            code = canonical_state(key, country)
            if code and len(key) > 3:         # full state names only: codes are ambiguous
                codes.append(code)
        if len(set(codes)) == 1:
            for comp in native:
                votes[comp][(country, codes[0])] += 1
    rows = []
    for comp, c in votes.items():
        (country, code), n = c.most_common(1)[0]
        if n >= min_count and n / sum(c.values()) >= 0.8:
            rows.append((comp, country, code, n))
    rows.sort(key=lambda r: -r[3])
    n = _write_tsv(NATIVE_STATE_MAP, ["native_component", "country", "state", "count"], rows)
    log(f"states: kept {n:,} native state spellings -> {NATIVE_STATE_MAP.relative_to(ROOT)}")


# -------------------------------------------------- abbreviation evidence
def mine_abbreviations(n_entities: int = 150_000, top: int = 400) -> None:
    """Token pairs (short, long) that differ between S1 and its matches, where
    the short form is an in-order subsequence of the long one."""
    gt = read_ground_truth(nrows=n_entities)
    pairs = ground_truth_pairs(gt)
    ids = set(pairs["entity_id"])
    other = pd.concat([ch[ch["entity_id"].isin(ids)] for s in (2, 3) for ch in iter_source("train", s)])
    s1 = read_source("train", 1)
    pairs = (pairs.merge(s1, left_on="source1_entity_id", right_on="entity_id", suffixes=("", "_s1"))
                  .merge(other, left_on="entity_id", right_on="entity_id", suffixes=("_s1", "")))

    def subseq(a: str, b: str) -> bool:
        it = iter(b)
        return all(ch in it for ch in a)

    cnt: collections.Counter = collections.Counter()
    for field in ("business_name", "business_address"):
        for a, b, country in zip(pairs[field + "_s1"], pairs[field], pairs["country_s1"]):
            A, B = set(basic_name_tokens(a)), set(basic_name_tokens(b))
            for x in B - A:
                for y in A - B:
                    if x.isdigit() or y.isdigit():
                        continue
                    short, long_ = (x, y) if len(x) < len(y) else (y, x)
                    if short and short[0] == long_[0] and len(short) < len(long_) and subseq(short, long_):
                        cnt[(field.split("_")[1], country, short, long_)] += 1
    out = ROOT / "reports" / "mined_abbreviations.tsv"
    out.parent.mkdir(exist_ok=True)
    rows = [(*k, n) for k, n in cnt.most_common(top)]
    _write_tsv(out, ["field", "country", "short", "long", "count"], rows)
    log(f"abbrev: wrote top {len(rows)} candidates -> {out.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["vocab", "brands", "indic", "abbrev"], help="mine one table group")
    args = ap.parse_args()
    RESOURCE_DIR.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "vocab"):
        mine_vocab()
    if args.only in (None, "brands"):
        mine_brand_pieces()
    if args.only in (None, "indic"):
        pairs = _load_indic_pairs()
        mine_indic_tokens(pairs)
        mine_native_states(pairs)
    if args.only in (None, "abbrev"):
        mine_abbreviations()


if __name__ == "__main__":
    main()
