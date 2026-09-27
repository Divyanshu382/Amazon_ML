"""Lazy loaders for the data-driven lookup tables in ``resources/``.

The tables are produced by ``python -m src.preprocessing.mine_resources`` from
the TRAINING data only. Every loader returns an empty table if its file is
missing, so the normalizers still work (just less well) before mining.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
INDIC_TOKEN_MAP = RESOURCE_DIR / "indic_token_map.tsv"      # native token -> latin token
NATIVE_STATE_MAP = RESOURCE_DIR / "native_state_map.tsv"    # native component -> country \t state
NAME_VOCAB = RESOURCE_DIR / "name_vocab.tsv"                # latin token -> count
ADDRESS_VOCAB = RESOURCE_DIR / "address_vocab.tsv"          # latin token -> count
BRAND_PIECES = RESOURCE_DIR / "brand_pieces.tsv"            # substring recurring in synthetic brands


def _read_tsv(path: Path, ncols: int) -> list[list[str]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        next(f, None)  # header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= ncols:
                rows.append(parts)
    return rows


@lru_cache(maxsize=1)
def indic_token_map() -> dict[str, str]:
    return {r[0]: r[1] for r in _read_tsv(INDIC_TOKEN_MAP, 2)}


@lru_cache(maxsize=1)
def native_state_map() -> dict[str, tuple[str, str]]:
    return {r[0]: (r[1], r[2]) for r in _read_tsv(NATIVE_STATE_MAP, 3)}


class Vocab:
    """Unigram model over clean Source-1 name tokens (segmentation + spell checks)."""

    def __init__(self, counts: dict[str, int]):
        self.counts = counts
        self.total = sum(counts.values()) or 1
        self.max_len = max((len(w) for w in counts), default=0)
        self._log_total = math.log(self.total)

    def __contains__(self, word: str) -> bool:
        return word in self.counts

    def __bool__(self) -> bool:
        return bool(self.counts)

    def logp(self, word: str) -> float:
        c = self.counts.get(word)
        if c:
            return math.log(c) - self._log_total
        # unseen: heavily penalised, more so for longer strings (Norvig-style)
        return math.log(10.0) - self._log_total - 2.3 * len(word)


@lru_cache(maxsize=1)
def name_vocab() -> Vocab:
    return Vocab({r[0]: int(r[1]) for r in _read_tsv(NAME_VOCAB, 2)})


@lru_cache(maxsize=1)
def address_vocab() -> dict[str, int]:
    """Token counts from Source-1 addresses (guards typo correction of real words)."""
    return {r[0]: int(r[1]) for r in _read_tsv(ADDRESS_VOCAB, 2)}


@lru_cache(maxsize=1)
def brand_pieces() -> frozenset[str]:
    """Substrings that recur across synthetic brand names ('kelo', 'faye', 'zeph')."""
    return frozenset(r[0] for r in _read_tsv(BRAND_PIECES, 2))
