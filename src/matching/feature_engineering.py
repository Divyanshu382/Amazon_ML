"""Pairwise features for (Source-1 record, Source-2/3 candidate).

``compute_features(pairs, s1, other)`` takes candidate pairs as row positions
into two preprocessed DataFrames and returns a float32 feature matrix. The
string work runs in a process pool (fork: the DataFrames are shared
copy-on-write).

Missing information is encoded explicitly (-1 / NaN) instead of as a mismatch:
an S2/S3 record without an address is "unknown", not "different".

Some features use word statistics computed WITHOUT labels on the candidate pairs
of the same split (``build_word_stats``): for a word the candidate ADDS to the
Source-1 name, a word it DROPS, or a change of legal form, how often the two
records still share the house number. Decoy siblings ("... Holding",
"... International", "inc" -> "corp") sit at a different number, noise copies
("... Services", "... & Fils") keep it, so the rate separates them in any
language, including France (absent from the training labels).
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from ..blocking.candidate_generator import STATE_EQUIV

RECORD_COLUMNS = [
    "entity_id", "business_name_norm", "name_core", "name_phonetic", "name_alt", "name_legal",
    "name_is_domain", "name_known_ratio", "name_script", "business_address_norm", "addr_street",
    "addr_city", "addr_state", "addr_house_no", "addr_numbers", "addr_postcode", "addr_is_missing",
    "country_norm",
]

STRING_FEATURES = [
    "name_core_exact", "name_norm_exact", "name_core_jac", "name_norm_jac", "name_core_ratio",
    "name_core_tsr", "name_core_tsort", "name_core_partial", "name_core_jw", "name_norm_tsr",
    "name_char3_jac", "name_glued_ratio", "name_phon_jac", "name_phon_exact", "name_first_tok_eq",
    "name_acronym", "name_alt_tsr", "name_a_in_b", "name_b_in_a", "name_len_a", "name_len_b",
    "name_len_diff", "legal_jac", "legal_conflict",
    "addr_missing_b", "state_eq", "house_eq", "num_jac", "num_a_in_b", "num_common",
    "addr_jac", "addr_tsr", "addr_ratio", "street_ratio", "city_eq", "city_ratio", "postcode_eq",
    "addr_words_common",
    # house-number relation (noise drops a leading digit; decoy siblings sit a few doors UP)
    "house_rel", "house_diff", "a_lead_in_b_nums", "b_lead_in_a_nums",
    # words added / dropped by the candidate, legal-form change
    "n_extra_tok", "n_missing_tok", "extra_typo_max", "heq_extra_min", "heq_extra_max", "heq_extra_mean",
    "n_extra_unknown", "heq_missing_min", "heq_missing_mean", "heq_legal", "legal_added", "legal_removed",
    # same statistics restricted to pairs on the SAME street (much sharper: decoy words ~0.03,
    # noise words ~0.75, versus ~0.2 vs ~0.4 when any shared address word such as the city counts)
    "heqs_extra_min", "heqs_extra_max", "heqs_extra_mean", "heqs_missing_min", "heqs_legal",
]
CONTEXT_FEATURES = ["block_score", "block_n_keys", "block_rank", "a_n_cands", "b_deg", "b_rank",
                    "b_score_gap", "is_s3", "known_b", "is_domain_b", "script_b", "missing_a",
                    "block_score_name", "block_score_addr",
                    # how ambiguous the name is among Source-1 entities; the *_norm_* counts include
                    # the legal form, which separates twins such as "X SARL" / "X SAS"
                    "a_core_dup", "a_core_state_dup", "b_core_in_s1", "a_norm_dup", "b_norm_in_s1"]
FEATURES = STRING_FEATURES + CONTEXT_FEATURES
_SCRIPT_CODE = {"latin": 0, "mixed": 1, "other": 3}

_A: list | None = None   # one contiguous Arrow array per RECORD_COLUMNS column (fast random take)
_B: list | None = None
_WS: dict = {k: {} for k in ("extra", "missing", "legal", "extra_s", "missing_s", "legal_s")}  # build_word_stats
_NAN = float("nan")
# house_rel codes
_REL = {"missing": 0, "equal": 1, "b_suffix_of_a": 2, "a_suffix_of_b": 3, "first_digit": 4,
        "last_digit": 5, "mid_digit": 6, "permuted": 7, "plus_small": 8, "minus_small": 9, "other": 10,
        "b_prefix_of_a": 11, "a_prefix_of_b": 12}


# French acronyms skip articles/prepositions and keep "Ets" ("Ets Lavant SARL" -> "EL",
# "Comité du Very" -> "CV"); these words are near-absent from US/India names.
_ACRONYM_SKIP = {"de", "du", "des", "la", "le", "les", "l", "d", "et", "and", "en", "au", "aux"}
_ACRONYM_LEGAL_SKIP = {"sarl", "sas", "sasu", "sa", "sci", "eurl", "snc", "scop", "cie", "co", "inc",
                       "llc", "ltd", "pvt", "corp", "llp", "lp", "pc", "pllc", "plc", "opc", "gmbh", "coop"}


def _acronym(act: list, an: str, bc: str) -> bool:
    b = bc.replace(" ", "")
    if len(act) > 1 and "".join(t[0] for t in act) == b:
        return True
    if not 2 <= len(b) <= 6 or not b.isalpha():
        return False
    for toks in ([t for t in act if t not in _ACRONYM_SKIP],
                 [t for t in an.split() if t not in _ACRONYM_SKIP and t not in _ACRONYM_LEGAL_SKIP]):
        if len(toks) > 1 and "".join(t[0] for t in toks) == b:
            return True
    return False


def _jac(a: set, b: set) -> float:
    if not a and not b:
        return -1.0
    return len(a & b) / len(a | b)


def _char3(s: str) -> set:
    s = s.replace(" ", "")
    return {s[i:i + 3] for i in range(len(s) - 2)} if len(s) > 2 else {s}


def _lead_num(h: str) -> str:
    out = ""
    for ch in h:
        if ch.isdigit():
            out += ch
        elif out:
            break
    return out


def _state(s: str) -> str:
    return STATE_EQUIV.get(s, s)


def _house_rel(a: str, b: str) -> tuple[float, float]:
    """(relation code, signed difference b - a) of two leading house numbers."""
    if not a or not b:
        return _REL["missing"], _NAN
    diff = float(max(-30_000, min(30_000, int(b[:9]) - int(a[:9]))))   # fits float16 storage
    if a == b:
        return _REL["equal"], diff
    if b.endswith(a):
        return _REL["a_suffix_of_b"], diff
    if a.endswith(b):
        return _REL["b_suffix_of_a"], diff
    if a.startswith(b):                       # trailing digit(s) dropped: 19753 -> 1975
        return _REL["b_prefix_of_a"], diff
    if b.startswith(a):
        return _REL["a_prefix_of_b"], diff
    if len(a) == len(b):
        pos = [i for i in range(len(a)) if a[i] != b[i]]
        if len(pos) == 1:
            return (_REL["first_digit"] if pos[0] == 0 else _REL["last_digit"] if pos[0] == len(a) - 1
                    else _REL["mid_digit"]), diff
        if sorted(a) == sorted(b):
            return _REL["permuted"], diff
    if 0 < diff <= 12:
        return _REL["plus_small"], diff
    if -12 <= diff < 0:
        return _REL["minus_small"], diff
    return _REL["other"], diff


def _legal_key(la: str, lb: str) -> str:
    return f"{la}>{lb}"


def _word_features(acs: set, bcs: set, al: str, bl: str) -> list[float]:
    extra, missing = bcs - acs, acs - bcs
    te_x, te_m, te_l = _WS["extra"], _WS["missing"], _WS["legal"]
    xs = [te_x[t] for t in extra if t in te_x]
    ms = [te_m[t] for t in missing if t in te_m]
    sx = [_WS["extra_s"][t] for t in extra if t in _WS["extra_s"]]
    sm = [_WS["missing_s"][t] for t in missing if t in _WS["missing_s"]]
    typo = max((JaroWinkler.normalized_similarity(x, m) for x in extra for m in missing), default=_NAN)
    als, bls = set(al.split()), set(bl.split())
    return [
        float(len(extra)), float(len(missing)), typo,
        min(xs) if xs else _NAN, max(xs) if xs else _NAN, sum(xs) / len(xs) if xs else _NAN,
        float(len(extra) - len(xs)),
        min(ms) if ms else _NAN, sum(ms) / len(ms) if ms else _NAN,
        te_l.get(_legal_key(al, bl), _NAN),
        float(len(bls - als)), float(len(als - bls)),
        min(sx) if sx else _NAN, max(sx) if sx else _NAN, sum(sx) / len(sx) if sx else _NAN,
        min(sm) if sm else _NAN, _WS["legal_s"].get(_legal_key(al, bl), _NAN),
    ]


def _pair_features(a: tuple, b: tuple) -> list[float]:
    (_, an, ac, ap, _, al, _, _, _, aa, ast, aci, asta, ah, anum, apc, _, _) = a
    (_, bn, bc, bp, balt, bl, _, _, _, ba, bst, bci, bsta, bh, bnum, bpc, bmiss, _) = b
    act, bct = ac.split(), bc.split()
    acs, bcs = set(act), set(bct)
    ans, bns = set(an.split()), set(bn.split())
    f = [
        float(ac == bc and ac != ""),
        float(an == bn and an != ""),
        _jac(acs, bcs),
        _jac(ans, bns),
        fuzz.ratio(ac, bc) / 100,
        fuzz.token_set_ratio(ac, bc) / 100,
        fuzz.token_sort_ratio(ac, bc) / 100,
        fuzz.partial_ratio(ac, bc) / 100,
        JaroWinkler.normalized_similarity(ac, bc),
        fuzz.token_set_ratio(an, bn) / 100,
        _jac(_char3(ac), _char3(bc)),
        fuzz.ratio(ac.replace(" ", ""), bc.replace(" ", "")) / 100,
        _jac(set(ap.split()), set(bp.split())),
        float(ap == bp and ap != ""),
        float(bool(act) and bool(bct) and act[0] == bct[0]),
        float(_acronym(act, an, bc)),
        fuzz.token_set_ratio(ac, balt) / 100 if balt else -1.0,
        len(acs & bcs) / len(acs) if acs else 0.0,
        len(acs & bcs) / len(bcs) if bcs else 0.0,
        float(len(act)), float(len(bct)), float(abs(len(act) - len(bct))),
    ]
    als, bls = set(al.split()), set(bl.split())
    f += [_jac(als, bls) if als and bls else -1.0,
          float(bool(als) and bool(bls) and not (als & bls))]
    words = _word_features(acs, bcs, al, bl)
    # ------------------------------------------------------------ address
    if bmiss:
        return f + [1.0] + [-1.0] * 13 + [_REL["missing"], _NAN, -1.0, -1.0] + words
    sa, sb = _state(asta), _state(bsta)
    ha, hb = _lead_num(ah), _lead_num(bh)
    na, nb = set(anum.split()), set(bnum.split())
    rel, diff = _house_rel(ha, hb)
    aw = {w for w in aa.split() if len(w) > 2 and w.isalpha()}
    bw = {w for w in ba.split() if len(w) > 2 and w.isalpha()}
    f += [
        0.0,
        (float(sa == sb) if sa and sb else -1.0),
        (float(ha == hb) if ha and hb else -1.0),
        _jac(na, nb) if na and nb else -1.0,
        (len(na & nb) / len(na) if na and nb else -1.0),
        float(len(na & nb)),
        _jac(set(aa.split()), set(ba.split())),
        fuzz.token_set_ratio(aa, ba) / 100,
        fuzz.ratio(aa, ba) / 100,
        (fuzz.token_set_ratio(ast, bst) / 100 if ast and bst else -1.0),
        (float(aci == bci) if aci and bci else -1.0),
        (fuzz.ratio(aci, bci) / 100 if aci and bci else -1.0),
        (float(apc == bpc) if apc and bpc else -1.0),
        float(len(aw & bw)),
        float(rel), diff,
        (float(ha in nb) if ha and nb else -1.0),
        (float(hb in na) if hb and na else -1.0),
    ]
    return f + words


def record_columns(df: pd.DataFrame) -> list:
    """RECORD_COLUMNS as contiguous Arrow arrays (pandas' chunked Arrow strings make row lookups
    ~40x slower). Build once per side; the DataFrame can be freed afterwards."""
    out = []
    for c in RECORD_COLUMNS:
        arr = df[c].array
        out.append(arr._pa_array.combine_chunks() if hasattr(arr, "_pa_array") else pa.array(df[c].to_numpy()))
    return out


def _rows(cols: list, idx: np.ndarray) -> list[tuple]:
    take = pa.array(idx)
    return list(zip(*[c.take(take).to_pylist() for c in cols]))


def _work(chunk: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    ai, bi = chunk
    return np.array([_pair_features(a, b) for a, b in zip(_rows(_A, ai), _rows(_B, bi))], dtype=np.float32)


def string_features(a_idx: np.ndarray, b_idx: np.ndarray, cols_a: list, cols_b: list,
                    workers: int | None = None, chunk: int = 50_000, ws: dict | None = None) -> np.ndarray:
    """Pairwise string features; cols_a / cols_b come from ``record_columns``."""
    global _A, _B, _WS
    _A, _B = cols_a, cols_b
    if ws is not None:
        _WS = ws
    chunks = [(a_idx[i:i + chunk], b_idx[i:i + chunk]) for i in range(0, len(a_idx), chunk)]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if not chunks:
        return np.zeros((0, len(STRING_FEATURES)), np.float32)
    with mp.get_context("fork").Pool(workers) as pool:
        parts = pool.map(_work, chunks)
    return np.vstack(parts)


def competition(cands: pd.DataFrame, s1: pd.DataFrame, other: pd.DataFrame) -> dict:
    """Per-row and per-record arrays behind the context features. ``cands`` must hold ALL
    candidates of the country so that b_deg / b_rank / b_score_gap see every S1 entity."""
    a, b = cands["a"].to_numpy(), cands["b"].to_numpy()
    score = cands["score"].to_numpy(np.float32)
    # rank of each row among the S1 entities competing for the same record (ties share the rank)
    order = np.lexsort((-score, b))
    b_s, sc_s = b[order], score[order]
    n = len(order)
    new_group = np.r_[True, b_s[1:] != b_s[:-1]]
    new_block = new_group | np.r_[True, sc_s[1:] != sc_s[:-1]]
    pos = np.arange(n)
    group_start = np.maximum.accumulate(np.where(new_group, pos, 0))
    block_start = np.maximum.accumulate(np.where(new_block, pos, 0))
    b_rank = np.empty(n, np.float32)
    b_rank[order] = block_start - group_start + 1
    b_gap = np.empty(n, np.float32)
    b_gap[order] = sc_s[group_start] - sc_s
    del order, b_s, sc_s, new_group, new_block, pos, group_start, block_start
    core_counts = s1["name_core"].value_counts()
    norm_counts = s1["business_name_norm"].value_counts()
    cs = s1["name_core"] + "|" + s1["addr_state"]
    return {
        "b_rank": b_rank, "b_gap": b_gap,
        "n_cands": np.bincount(a, minlength=len(s1)).astype(np.float32),
        "b_deg": np.bincount(b, minlength=len(other)).astype(np.float32),
        "is_s3": other["entity_id"].str.startswith("S3").to_numpy(np.float32),
        "known": other["name_known_ratio"].to_numpy(np.float32),
        "domain": other["name_is_domain"].to_numpy(np.float32),
        "script": np.array([_SCRIPT_CODE.get(x, 2) for x in other["name_script"].to_numpy()], np.float32),
        "missing_a": s1["addr_is_missing"].to_numpy(np.float32),
        "a_core_dup": s1["name_core"].map(core_counts).to_numpy(np.float32),
        "a_core_state_dup": cs.map(cs.value_counts()).to_numpy(np.float32),
        "b_core_in_s1": other["name_core"].map(core_counts).fillna(0).to_numpy(np.float32),
        "a_norm_dup": s1["business_name_norm"].map(norm_counts).to_numpy(np.float32),
        "b_norm_in_s1": other["business_name_norm"].map(norm_counts).fillna(0).to_numpy(np.float32),
    }


def context_features(cands: pd.DataFrame, comp: dict, sel: np.ndarray | None = None) -> np.ndarray:
    """CONTEXT_FEATURES for rows ``sel`` of ``cands`` (all rows if None)."""
    if sel is None:
        sel = np.arange(len(cands))
    a_s, b_s = cands["a"].to_numpy()[sel], cands["b"].to_numpy()[sel]
    cols = [
        cands["score"].to_numpy()[sel], cands["n_keys"].to_numpy()[sel], cands["rank"].to_numpy()[sel],
        comp["n_cands"][a_s], comp["b_deg"][b_s], comp["b_rank"][sel], comp["b_gap"][sel],
        comp["is_s3"][b_s], comp["known"][b_s], comp["domain"][b_s], comp["script"][b_s],
        comp["missing_a"][a_s],
        cands["score_name"].to_numpy()[sel], cands["score_addr"].to_numpy()[sel],
        comp["a_core_dup"][a_s], comp["a_core_state_dup"][a_s], comp["b_core_in_s1"][b_s],
        comp["a_norm_dup"][a_s], comp["b_norm_in_s1"][b_s],
    ]
    out = np.empty((len(sel), len(cols)), np.float32)
    for j, c in enumerate(cols):
        out[:, j] = c
    return out


# ------------------------------------------------------------ word statistics
class WordStats:
    """House-number agreement rate per added word, dropped word and legal-form change, counted
    over candidate pairs that share a name word and both have a house number. Two versions:
    'loose' pairs share any address word; 'strict' (*_s) pairs are on the same street.
    Uses no labels."""

    KINDS = ("extra", "missing", "legal", "extra_s", "missing_s", "legal_s")

    def __init__(self) -> None:
        self.eq = {k: Counter() for k in self.KINDS}
        self.tot = {k: Counter() for k in self.KINDS}

    def _add(self, suffix: str, extra, missing, legal_key: str, same: bool) -> None:
        for t in extra:
            self.tot["extra" + suffix][t] += 1
            self.eq["extra" + suffix][t] += same
        for t in missing:
            self.tot["missing" + suffix][t] += 1
            self.eq["missing" + suffix][t] += same
        self.tot["legal" + suffix][legal_key] += 1
        self.eq["legal" + suffix][legal_key] += same

    def update(self, core_a, core_b, legal_a, legal_b, house_a, house_b, addr_a, addr_b,
               street_a, street_b) -> None:
        for ca, cb, la, lb, ha, hb, aa, ab, sa, sb in zip(core_a, core_b, legal_a, legal_b, house_a,
                                                          house_b, addr_a, addr_b, street_a, street_b):
            ha, hb = _lead_num(ha), _lead_num(hb)
            if not ha or not hb:
                continue
            A, B = set(ca.split()), set(cb.split())
            if not (A & B):
                continue
            wa = {w for w in aa.split() if w.isalpha() and len(w) > 2}
            wb = {w for w in ab.split() if w.isalpha() and len(w) > 2}
            if not (wa & wb):
                continue
            same = ha == hb
            extra, missing, key = B - A, A - B, _legal_key(la, lb)
            self._add("", extra, missing, key, same)
            ka = " ".join(sorted({w for w in sa.split() if w.isalpha() and len(w) > 2}))
            if ka and ka == " ".join(sorted({w for w in sb.split() if w.isalpha() and len(w) > 2})):
                self._add("_s", extra, missing, key, same)

    def table(self, alpha: float = 20.0, min_count: int = 20) -> dict:
        out = {}
        for kind in self.KINDS:
            n = sum(self.tot[kind].values())
            r0 = sum(self.eq[kind].values()) / max(n, 1)
            out[kind] = {t: round((self.eq[kind][t] + alpha * r0) / (c + alpha), 4)
                         for t, c in self.tot[kind].items() if c >= min_count}
        return out


WORD_STAT_COLUMNS = ["entity_id", "name_core", "name_legal", "addr_house_no", "business_address_norm",
                     "addr_street"]


def build_word_stats(splits=("train", "test"), max_pairs_per_country: int = 15_000_000) -> dict:
    """Word statistics over the candidate pairs of ``splits`` together (all countries).

    One table is shared by training and scoring: computed per split, the same-street rates
    shift between train and test (test has more decoy records), and a model trained on the
    train values then misreads the test values."""
    from ..blocking.candidate_generator import candidates_path, countries, load_country
    stats = WordStats()
    for split, country in ((sp, c) for sp in splits for c in countries(sp)):
        cands = pd.read_parquet(candidates_path(split, country), columns=["a", "b"])
        if len(cands) > max_pairs_per_country:
            cands = cands.sample(max_pairs_per_country, random_state=0)
        s1 = load_country(split, (1,), country, WORD_STAT_COLUMNS)
        other = load_country(split, (2, 3), country, WORD_STAT_COLUMNS)
        a, b = cands["a"].to_numpy(), cands["b"].to_numpy()
        stats.update(*(df[c].to_numpy()[i] for c in ("name_core", "name_legal", "addr_house_no",
                                                      "business_address_norm", "addr_street")
                       for df, i in ((s1, a), (other, b))))
    return stats.table()
