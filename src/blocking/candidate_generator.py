"""Multi-pass key blocking: Source 1 entity -> plausible Source 2/3 candidates.

Every record emits a set of blocking keys (all scoped to its country, which is
read from the data, never hard-coded):

    n   core-name token + state          p   phonetic token + state
    N   core-name token (country-wide)   b   any pair of core-name tokens
    c   whole core name                  h   any address number + state
    s   whole core name + state          w   whole core name + address word
    t   core-name token + address number W   address number + address word
    y   core-name token + city
    a   address word + state             A   adjacent address-word bigram
    m   core-name token, emitted on the S2/S3 side only by records WITHOUT a
        state (missing address); S1 emits it for every token, so address-less
        records are still reachable through a small, state-free block
    M / q  whole core name / any pair of core tokens, likewise emitted only by
        address-less S2/S3 records

Keys whose Source-2/3 block is larger than a per-kind cap are dropped (they
carry no signal). Each (S1, candidate) pair is scored by the summed IDF of the
keys it shares -- in total, over name keys only and over address keys only --
and a candidate is kept if it is in the entity's top ``k`` by total score, top
``k_addr`` by address score (brand-only / renamed copies) or top ``k_name`` by
name score (copies with a missing or rewritten address), plus the top
``k_noaddr`` address-less candidates by name score (their name is the only
evidence, so they rarely win a slot on total score).
Everything is vectorised with numpy (sorted key arrays + searchsorted), so
a whole country is processed in bounded memory, chunk by chunk.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..preprocessing.preprocessing import processed_path

KEY_COLUMNS = ["entity_id", "name_core", "name_phonetic", "addr_state", "addr_numbers",
               "addr_city", "business_address_norm", "country_norm"]
# States that the data uses interchangeably (Telangana was part of Andhra Pradesh until 2014).
STATE_EQUIV = {"tg": "ap"}
# per-kind cap on the Source-2/3 block size, and a weight multiplier
KINDS = {
    "n": (400, 1.0), "N": (150, 0.8), "p": (400, 0.6), "b": (400, 1.2),
    "c": (600, 1.5), "h": (300, 1.0), "a": (400, 0.7), "A": (400, 0.9),
    "m": (300, 1.0), "s": (300, 1.6), "w": (300, 1.6),
    "t": (300, 1.8), "W": (300, 1.3), "y": (300, 1.2),
    "M": (300, 1.5), "q": (300, 1.2),
}
# key kind -> score group (0 name, 1 address, 2 name x address compound)
KIND_GROUP = {"n": 0, "N": 0, "p": 0, "b": 0, "c": 0, "s": 0, "m": 0, "M": 0, "q": 0,
              "h": 1, "a": 1, "A": 1, "W": 1, "t": 2, "w": 2, "y": 2}
_STOP = {"and", "of", "the", "de", "du", "la", "le", "des", "et", "for", "in"}
# address words too generic to block on
_ADDR_STOP = {
    "street", "road", "drive", "avenue", "lane", "court", "circle", "place", "boulevard", "trail",
    "highway", "unit", "suite", "apartment", "apartments", "floor", "north", "south", "east", "west",
    "city", "county", "nagar", "colony", "near", "opposite", "sector", "plot", "flat", "main", "cross",
    "layout", "the", "and", "rue", "avenue", "chemin", "route", "des", "du", "de", "la", "le", "les",
    "building", "house", "road", "district", "village", "post", "office", "ground", "first", "second",
    "township", "cdp", "way", "parkway", "terrace", "square", "phase", "block", "stage", "extension",
}
_MAX_PAIR_TOKENS = 6
# candidates kept per S1 entity: top K by total score + top K_ADDR by address score + top K_NAME by name
K, K_ADDR, K_NAME, K_NOADDR = 30, 15, 5, 3


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_country(split: str, sources: tuple[int, ...], country: str, columns: list[str]) -> pd.DataFrame:
    """Processed records of ``sources`` for one country (row order = file order)."""
    cols = list(dict.fromkeys(columns + ["country_norm"]))
    parts = [pq.read_table(processed_path(split, s), columns=cols,
                           filters=[("country_norm", "=", country)]).to_pandas() for s in sources]
    return pd.concat(parts, ignore_index=True)


def countries(split: str) -> list[str]:
    """All country labels present in a split's Source 1 (open set)."""
    t = pq.read_table(processed_path(split, 1), columns=["country_norm"]).column(0)
    return sorted(set(t.to_pylist()))


# ------------------------------------------------------------------ keys
def _hash(strings) -> np.ndarray:
    return pd.util.hash_array(np.asarray(strings, dtype=object))


def make_keys(df: pd.DataFrame, offset: int = 0, side: str = "b") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (key_hash uint64, row index int32, kind code int8) for every key of every record.
    ``side`` is "a" for the querying Source-1 records, "b" for indexed Source-2/3 records."""
    df = df.reset_index(drop=True)
    idx = np.arange(offset, offset + len(df), dtype=np.int32)
    state = df["addr_state"].fillna("").replace(STATE_EQUIV)
    hashes, rows, kinds = [], [], []

    def add(kind: str, keys: pd.Series, row_idx: np.ndarray) -> None:
        if len(keys):
            hashes.append(_hash(kind + "|" + keys))
            rows.append(row_idx.astype(np.int32))
            kinds.append(np.full(len(keys), list(KINDS).index(kind), dtype=np.int8))

    toks = df["name_core"].str.split().explode().dropna()
    toks = toks[(toks.str.len() >= 2) & ~toks.isin(_STOP)]
    tok_rows = toks.index.to_numpy()
    add("n", toks + "|" + state.to_numpy()[tok_rows], idx[tok_rows])
    add("N", toks, idx[tok_rows])
    no_state = (state.to_numpy()[tok_rows] == "") if side == "b" else np.ones(len(toks), bool)
    add("m", toks[no_state], idx[tok_rows][no_state])

    ph = df["name_phonetic"].str.split().explode().dropna()
    ph = ph[ph.str.len() >= 3]
    add("p", ph + "|" + state.to_numpy()[ph.index.to_numpy()], idx[ph.index.to_numpy()])

    words = df["name_core"].str.split()
    bi = words.map(_token_pairs).explode().dropna()
    bi_rows = bi.index.to_numpy()
    add("b", bi, idx[bi_rows])
    no_state_rows = (state.to_numpy() == "") if side == "b" else np.ones(len(df), bool)
    add("q", bi[no_state_rows[bi_rows]], idx[bi_rows][no_state_rows[bi_rows]])

    core = df["name_core"]
    m = (core.str.len() >= 3).to_numpy()
    add("c", core[m], idx[m])
    add("M", core[m & no_state_rows], idx[m & no_state_rows])

    m = (core.str.len() >= 3).to_numpy()
    add("s", core[m] + "|" + state[m], idx[m])

    # every number in the address: noise often prepends a fake one ("DOOR NO #904 284")
    nums = df["addr_numbers"].fillna("").str.split().explode().dropna()
    nums = nums[nums != ""]
    nr = nums.index.to_numpy()
    add("h", nums + "|" + state.to_numpy()[nr], idx[nr])
    addr_words = df["business_address_norm"].fillna("").map(
        lambda a: [w for w in a.split() if len(w) >= 3 and w.isalpha() and w not in _ADDR_STOP])
    aw = addr_words.explode().dropna()
    ar = aw.index.to_numpy()
    add("a", aw + "|" + state.to_numpy()[ar], idx[ar])
    core_np = core.to_numpy()
    m4 = np.array([len(c) >= 3 for c in core_np[ar]], bool) if len(ar) else np.zeros(0, bool)
    add("w", core_np[ar][m4] + "|" + aw[m4], idx[ar][m4])
    # compound keys: robust to a typo in one name token, a renamed / brand-only name,
    # or a common house number ("jay|185", "202|surat", "infotech|siliguri")
    num_df = pd.DataFrame({"r": nr, "num": nums.to_numpy()})
    tok_df = pd.DataFrame({"r": tok_rows, "tok": toks.to_numpy()})
    word_df = pd.DataFrame({"r": ar, "word": aw.to_numpy()})
    tn = tok_df.merge(num_df, on="r")
    add("t", tn["tok"] + "|" + tn["num"], idx[tn["r"].to_numpy()])
    nw = num_df.merge(word_df, on="r")
    add("W", nw["num"] + "|" + nw["word"], idx[nw["r"].to_numpy()])
    city = df["addr_city"].fillna("").to_numpy()
    has_city = city[tok_rows] != ""
    add("y", toks[has_city] + "|" + city[tok_rows][has_city], idx[tok_rows][has_city])
    abi = addr_words.map(lambda w: [x + " " + y for x, y in zip(w, w[1:])]).explode().dropna()
    add("A", abi, idx[abi.index.to_numpy()])
    return np.concatenate(hashes), np.concatenate(rows), np.concatenate(kinds)


def _token_pairs(words) -> list[str]:
    """All sorted pairs of the first few core tokens: robust to reordering and one typo."""
    if words is None or len(words) < 2:
        return []
    w = [t for t in words[:_MAX_PAIR_TOKENS] if len(t) >= 2 and t not in _STOP]
    return [a + " " + b if a < b else b + " " + a for i, a in enumerate(w) for b in w[i + 1:]]


# ----------------------------------------------------------------- index
@dataclass
class BlockIndex:
    keys: np.ndarray      # sorted unique-able key hashes (uint64)
    rows: np.ndarray      # candidate row for each key entry
    uniq: np.ndarray      # unique kept keys
    start: np.ndarray     # start offset of each unique key in keys/rows
    size: np.ndarray      # block size
    weight: np.ndarray    # idf * kind weight per unique key
    group: np.ndarray     # score group of each unique key (see KIND_GROUP)
    n_records: int
    no_addr: np.ndarray   # per indexed record: no state (address missing)

    @classmethod
    def build(cls, df: pd.DataFrame, chunk: int = 1_000_000) -> "BlockIndex":
        hs, rs, ks = [], [], []
        for off in range(0, len(df), chunk):
            h, r, k = make_keys(df.iloc[off:off + chunk], offset=off)
            hs.append(h), rs.append(r), ks.append(k)
        h, r, k = np.concatenate(hs), np.concatenate(rs), np.concatenate(ks)
        del hs, rs, ks
        order = np.argsort(h, kind="stable")
        h, r, k = h[order], r[order], k[order]
        # drop repeated (key, row) entries (a token twice in one name)
        keep = np.ones(len(h), bool)
        keep[1:] = (h[1:] != h[:-1]) | (r[1:] != r[:-1])
        h, r, k = h[keep], r[keep], k[keep]
        uniq, start, size = np.unique(h, return_index=True, return_counts=True)
        kind_of = k[start]
        caps = np.array([c for c, _ in KINDS.values()])[kind_of]
        mult = np.array([w for _, w in KINDS.values()])[kind_of]
        ok = size <= caps
        weight = (np.log(len(df) / size) * mult).astype(np.float32)
        group = np.array([KIND_GROUP[k] for k in KINDS], np.int8)[kind_of]
        no_addr = (df["addr_state"].fillna("").to_numpy() == "")
        return cls(h, r, uniq[ok], start[ok], size[ok], weight[ok], group[ok], len(df), no_addr)

    def query(self, df: pd.DataFrame, k: int, k_addr: int = 0, k_name: int = 0,
              k_noaddr: int = 0) -> pd.DataFrame:
        """Candidates for each row of ``df`` -> (a, b, score, score_name, score_addr, n_keys, rank).
        Kept: top ``k`` by total score, plus top ``k_addr`` by address score, top ``k_name`` by name
        score and top ``k_noaddr`` address-less candidates by name score."""
        h, a, _ = make_keys(df, side="a")
        pos = np.searchsorted(self.uniq, h)
        pos[pos >= len(self.uniq)] = 0
        hit = self.uniq[pos] == h
        a, pos = a[hit], pos[hit]
        # a key repeated for the same S1 row counts once
        pair_key = pd.DataFrame({"a": a, "p": pos}).drop_duplicates()
        a, pos = pair_key["a"].to_numpy(), pair_key["p"].to_numpy()
        sizes = self.size[pos]
        total = int(sizes.sum())
        a_rep = np.repeat(a, sizes)
        starts = np.repeat(self.start[pos] - np.concatenate([[0], np.cumsum(sizes)[:-1]]), sizes)
        b = self.rows[starts + np.arange(total)]
        w = np.repeat(self.weight[pos], sizes)
        g = np.repeat(self.group[pos], sizes)
        pairs = pd.DataFrame({"a": a_rep, "b": b, "score": w,
                              "score_name": np.where(g == 0, w, 0).astype(np.float32),
                              "score_addr": np.where(g == 1, w, 0).astype(np.float32),
                              "n_keys": np.ones(len(w), np.int16)})
        agg = pairs.groupby(["a", "b"], sort=False).sum().reset_index()
        agg["rank"] = agg.groupby("a")["score"].rank(ascending=False, method="first").astype(np.int16) - 1
        keep = agg["rank"] < k
        if k_addr:
            ra = agg.groupby("a")["score_addr"].rank(ascending=False, method="first")
            keep |= (ra <= k_addr) & (agg["score_addr"] > 0)
        if k_name:
            rn = agg.groupby("a")["score_name"].rank(ascending=False, method="first")
            keep |= (rn <= k_name) & (agg["score_name"] > 0)
        if k_noaddr:
            na = self.no_addr[agg["b"].to_numpy()] & (agg["score_name"] > 0).to_numpy()
            r_na = agg["score_name"].where(na, -1.0).groupby(agg["a"]).rank(ascending=False, method="first")
            keep |= na & (r_na <= k_noaddr).to_numpy()
        return agg[keep].sort_values(["a", "rank"]).reset_index(drop=True)


def generate_candidates(s1: pd.DataFrame, index: BlockIndex, k: int = K, k_addr: int = K_ADDR,
                        k_name: int = K_NAME, k_noaddr: int = K_NOADDR, chunk: int = 20_000) -> pd.DataFrame:
    """Candidates for all rows of s1 (row positions) against the index's records."""
    out = []
    for off in range(0, len(s1), chunk):
        c = index.query(s1.iloc[off:off + chunk], k, k_addr, k_name, k_noaddr)
        c["a"] += off
        out.append(c)
    res = pd.concat(out, ignore_index=True)
    res["a"] = res["a"].astype(np.int32)
    res["b"] = res["b"].astype(np.int32)
    return res


# ------------------------------------------------------------------ driver
CANDIDATE_DIR = processed_path("train", 1).parents[1] / "candidates"


def candidates_path(split: str, country: str):
    return CANDIDATE_DIR / f"{split}_{country.replace(' ', '_')}.parquet"


def run_blocking(split: str, k: int = K, only: list[str] | None = None) -> None:
    """Candidates for every Source-1 entity of ``split``, one file per country."""
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    for country in countries(split):
        if only and country not in only:
            continue
        t0 = time.time()
        other = load_country(split, (2, 3), country, KEY_COLUMNS)
        s1 = load_country(split, (1,), country, KEY_COLUMNS)
        if len(other) == 0:
            cands = pd.DataFrame({"a": [], "b": [], "score": [], "score_name": [], "score_addr": [],
                                  "n_keys": [], "rank": []})
        else:
            index = BlockIndex.build(other)
            cands = generate_candidates(s1, index, k=k)
            del index
        cands["s1_id"] = s1["entity_id"].to_numpy()[cands["a"].to_numpy(dtype=np.int64)]
        cands["cand_id"] = other["entity_id"].to_numpy()[cands["b"].to_numpy(dtype=np.int64)]
        cands.to_parquet(candidates_path(split, country), compression="zstd", index=False)
        log(f"{split}/{country}: {len(s1):,} S1 x {len(other):,} S2/S3 -> {len(cands):,} candidates "
            f"({len(cands) / max(len(s1), 1):.1f}/entity, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate blocking candidates")
    ap.add_argument("--split", nargs="+", default=["train", "test"])
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--countries", nargs="*", help="only these countries (others keep their candidates)")
    args = ap.parse_args()
    for sp in args.split:
        run_blocking(sp, args.k, args.countries)
