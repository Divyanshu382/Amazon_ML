"""Business-name normalisation.

``normalize_name(name)`` returns one canonical string; ``parse_name(name)``
returns every derived field used downstream (see ``NameParts``).

Noise patterns handled (all observed in train Source 2/3 vs Source 1):

* Indic-script names, fully or partly ("Sky હેલ્થકેર Private Limited") ->
  mined token dictionary, else rule-based transliteration
* injected accents ("Gróup", "Límited") and look-alike digits / letters
  ("5tar", "Mem0rial", "RUSHlN")
* alias markers "<brand> dba|f/k/a|formerly|t/a|... <real name>"
* ID tails "#60562", "(ID: 93967)", "| www.site.com"; bracket wrappers
  "[LLC]", "(Corp)"; junk prefixes "-- ", ">> ", "***"
* glued domains / handles "tristateguild.com", "#rajkotbiotech", "@north24"
  -> segmented "tri state guild", "rajkot biotech", "north 24"
* legal-form variants: Private/Pvt, Limited/Ltd, L.L.C., Incorporated, ...
  (incl. French SARL/SAS/EURL/SCI, Cie, Ets)
* honorifics (Shri, Sri, Smt, M/s, The) and repeated tokens ("Royal Royal")
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from .resources import brand_pieces, indic_token_map, name_vocab
from .text_utils import (clean_unicode, detect_script, fold_accents, has_indic,
                         phonetic_key, transliterate_indic)

# ------------------------------------------------------------------ tables
# Multi-token phrases first; matched on whole tokens after punctuation removal.
LEGAL_PHRASES = [
    ("limited liability partnership", "llp"), ("limited liability company", "llc"),
    ("public limited company", "ltd"), ("private limited", "pvt ltd"), ("pvt limited", "pvt ltd"),
    ("private ltd", "pvt ltd"), ("pte ltd", "pvt ltd"),
    ("professional corporation", "pc"), ("professional limited liability company", "pllc"),
    ("one person company", "opc"), ("societe anonyme", "sa"), ("l l c", "llc"), ("l l p", "llp"),
    ("p l l c", "pllc"), ("p c", "pc"), ("l p", "lp"), ("s a s u", "sasu"), ("s a s", "sas"),
    ("s a r l", "sarl"), ("s c i", "sci"), ("e u r l", "eurl"),
]
LEGAL_WORDS = {
    "limited": "ltd", "ltd": "ltd", "private": "pvt", "pvt": "pvt", "prvt": "pvt", "pte": "pvt",
    "incorporated": "inc", "inc": "inc", "corporation": "corp", "corp": "corp", "company": "co",
    "co": "co", "llc": "llc", "llp": "llp", "lp": "lp", "pc": "pc", "pllc": "pllc", "plc": "plc",
    "opc": "opc", "gmbh": "gmbh", "sa": "sa", "sas": "sas", "sasu": "sasu", "sarl": "sarl",
    "eurl": "eurl", "sci": "sci", "snc": "snc", "scop": "scop", "compagnie": "cie", "cie": "cie",
    "etablissements": "ets", "etablissement": "ets", "ets": "ets", "cooperative": "coop",
    "coop": "coop",
}
LEGAL_FORMS = set(LEGAL_WORDS.values())
# Stripped from the "core" name only (kept in name_norm).
HONORIFICS = {"the", "shri", "sri", "shree", "smt", "ms", "mr", "mrs", "dr", "messrs"}
DESIGNATIONS = {"esq", "dds", "cpa"}
CORE_STOPWORDS = {"and"}

ALIAS_MARKER_RE = re.compile(
    r"\s+(?:d/b/a|d\.\s?b\.\s?a\.?|dba|doing business as|f/k/a|f\.\s?k\.\s?a\.?|fka|"
    r"formerly known as|formerly|a/k/a|a\.\s?k\.\s?a\.?|aka|also known as|t/a|trading as|n[e\u00e9]e)"
    r"\s*[:\-]?\s+",
    re.IGNORECASE)
_ID_TAIL_RE = re.compile(r"\s*(?:\(\s*id\s*[:#-]?\s*\d+\s*\)|#\s*\d{4,})\s*$", re.IGNORECASE)
_GLUED_RE = re.compile(
    r"^(?:[#@]+(?P<handle>[0-9a-z][0-9a-z_-]*)|(?:https?://)?(?:www\.)?"
    r"(?P<domain>[0-9a-z][0-9a-z-]*)\.(?:co\.in|co\.uk|com|net|org|in|co|biz|info|us|fr|io)/?)$")
_INNER_DIGIT_RE = re.compile(r"[a-z]+[013456789][a-z]+")
_PLUS_AND_RE = re.compile(r"(?<=\w)\s*\+\s*(?=\w)")
_DOTTED_ACRONYM_RE = re.compile(r"\b(?:[a-z]\.){2,}(?:[a-z]\b\.?)?")
_NON_ALNUM_RE = re.compile(r"[^0-9a-zऀ-ൿ]+")
_CAPS_WITH_L_RE = re.compile(r"\b[A-Z]*l[A-Z]*\b")
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b",
                       "9": "g"})
_ONE_DIGIT_WORD_RE = re.compile(r"^[a-z]*[013456789][a-z]*$")
_ORDINAL_RE = re.compile(r"^\d+(?:st|nd|rd|th)$")


@dataclass
class NameParts:
    name_norm: str = ""       # canonical full name (legal forms canonicalised)
    name_core: str = ""       # name_norm minus legal forms / honorifics / 'and'
    name_legal: str = ""      # canonical legal forms found, sorted ("llc", "ltd pvt")
    name_alt: str = ""        # the other side of a dba / f/k/a marker, normalised
    name_is_domain: bool = False  # contained a glued handle / domain ("#rajkotbiotech", "x.com")
    name_known_ratio: float = 1.0  # share of core words known from S1 names (0 = brand-only)
    name_script: str = "latin"  # script of the RAW name
    name_phonetic: str = ""   # consonant skeleton of name_core tokens (plural -s dropped)

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- helpers
def _fix_caps_l(text: str) -> str:
    """'RUSHlN' -> 'RUSHIN': a lone lower-case l inside an all-caps word is an I."""
    def rep(m):
        w = m.group(0)
        return w.replace("l", "I") if len(w) >= 3 and sum(c.isupper() for c in w) >= 2 else w
    return _CAPS_WITH_L_RE.sub(rep, text)


def _romanize_tokens(text: str) -> str:
    """Replace native-script tokens via the mined map, else rule transliteration."""
    if not has_indic(text):
        return text
    tmap = indic_token_map()
    out = []
    for tok in text.split():
        if has_indic(tok):
            core = tok.strip(".,;:()[]{}\"'")
            tok = tmap.get(core) or transliterate_indic(core)
        out.append(tok)
    return " ".join(out)


def segment(word: str) -> list[str]:
    """Split a glued string into vocabulary words: 'tristateguild' -> ['tri','state','guild']."""
    vocab = name_vocab()
    if not vocab or len(word) < 4 or word in vocab:
        return [word]
    n = len(word)
    max_len = min(vocab.max_len, 20)
    best = [0.0] + [float("-inf")] * n
    back = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(max(0, i - max_len), i):
            score = best[j] + vocab.logp(word[j:i])
            if score > best[i]:
                best[i], back[i] = score, j
    parts, i = [], n
    while i > 0:
        parts.append(word[back[i]:i])
        i = back[i]
    return parts[::-1]


def _fix_token(tok: str) -> str:
    """Undo digit-for-letter swaps ('5tar', 'mem0rial') when that yields a known word."""
    if tok.isalpha() or tok.isdigit() or _ORDINAL_RE.match(tok) or not _ONE_DIGIT_WORD_RE.match(tok):
        return tok
    if sum(c.isalpha() for c in tok) < 3:
        return tok
    fixed = tok.translate(_LEET)
    vocab = name_vocab()
    return fixed if (not vocab or fixed in vocab) and tok not in vocab else tok


def _canonical_legal(tokens: list[str]) -> list[str]:
    out, i, n = [], 0, len(tokens)
    while i < n:
        for phrase, canon in LEGAL_PHRASES:
            p = phrase.split()
            if tokens[i:i + len(p)] == p:
                out.extend(canon.split())
                i += len(p)
                break
        else:
            out.append(LEGAL_WORDS.get(tokens[i], tokens[i]))
            i += 1
    return out


def _dedupe_consecutive(tokens: list[str]) -> list[str]:
    return [t for k, t in enumerate(tokens) if k == 0 or t != tokens[k - 1]]


def _unglue(word: str) -> list[str]:
    """Split a glued handle / domain label: 'orbo1adome' -> ['orbola', 'dome'],
    '1329main' -> ['1329', 'main'], 'rajkotbiotech' -> ['rajkot', 'biotech']."""
    out: list[str] = []
    for w in re.split(r"[-_.]+", word):
        if _INNER_DIGIT_RE.fullmatch(w):          # look-alike digit inside a glued word
            w = w.translate(_LEET)
        for part in re.findall(r"\d+|[a-z]+", w):
            out.extend(segment(part) if part.isalpha() else [part])
    return out


def _split_long_oov(tok: str) -> list[str]:
    """'mapessuperiorresourcecom' -> ['mapes', 'superior', 'resource']: a long unknown token is
    split only if every piece is a well-attested Source-1 word and none is a syllable of the
    synthetic brands ('fayevantagequo' stays one token)."""
    vocab = name_vocab()
    if len(tok) < 10 or not tok.isalpha() or not vocab or vocab.counts.get(tok, 0) >= 3:
        return [tok]
    parts = segment(tok)
    if len(parts) > 2 and parts[-1] in ("com", "in", "net", "org"):
        parts = parts[:-1]
    brand = brand_pieces()
    if len(parts) >= 2 and all(len(p) >= 3 and vocab.counts.get(p, 0) >= 20 and p[:6] not in brand
                               for p in parts):
        return parts
    return [tok]


def _basic_tokens(text: str) -> tuple[list[str], bool]:
    """Raw (single, marker-free) name -> canonical tokens, has-glued-handle/domain flag."""
    text = _ID_TAIL_RE.sub("", text)
    if "|" in text:  # "complete optimal pro llc | www.completeo.com"
        text = next((p for p in text.split("|") if p.strip()), "")
    text = _ID_TAIL_RE.sub("", text).strip()
    text = _fix_caps_l(text)
    text = _romanize_tokens(text)
    text = fold_accents(text).lower()
    pieces, glued = [], False
    for tok in text.split():                      # "#rajkotbiotech", "@tulsacatering", "x.com"
        m = _GLUED_RE.match(tok.strip(".,;:!()[]{}\"'"))
        word = m and (m.group("handle") or m.group("domain"))
        if word and sum(c.isalpha() for c in word) >= 4:
            glued = True
            pieces.extend(_unglue(word))
        else:
            pieces.append(tok)
    text = " ".join(pieces)
    text = text.replace("&", " and ")
    text = _PLUS_AND_RE.sub(" and ", text)        # "Whipple + Williams"; a trailing "+" is junk
    text = re.sub(r"['`´]", "", text)             # possessives: twyla's -> twylas
    text = re.sub(r"\bm\s*/\s*s\b", " ms ", text)  # "M/s" honorific
    text = _DOTTED_ACRONYM_RE.sub(lambda m: m.group(0).replace(".", "") + " ", text)
    tokens = _NON_ALNUM_RE.sub(" ", text).split()
    tokens = [_fix_token(t) for t in tokens]
    split_tokens = []
    for t in tokens:
        parts = _split_long_oov(t)
        glued |= len(parts) > 1
        split_tokens.extend(parts)
    tokens = _canonical_legal(split_tokens)
    return _dedupe_consecutive(tokens), glued


def _known_ratio(core: list[str]) -> float:
    """Share of alphabetic core tokens seen (>= 3 times) in clean Source-1 names.
    0.0 flags brand-only / fully garbled names ('Drexvio', 'Korkelo')."""
    vocab = name_vocab()
    words = [t for t in core if t.isalpha() and len(t) > 1]
    if not words or not vocab:
        return 1.0
    return round(sum(vocab.counts.get(t, 0) >= 3 for t in words) / len(words), 3)


def _phonetic(tokens: list[str]) -> str:
    """Phonetic key per token, plural -s dropped ('services' ~ 'service')."""
    return " ".join(phonetic_key(t[:-1] if len(t) > 4 and t.endswith("s") else t)
                    for t in tokens if not t.isdigit())


def _core(tokens: list[str]) -> list[str]:
    drop = LEGAL_FORMS | HONORIFICS | DESIGNATIONS | CORE_STOPWORDS
    core = [t for t in tokens if t not in drop]
    return core or tokens


# -------------------------------------------------------------------- API
def split_alias(name: str) -> tuple[str, str]:
    """'Nexsolzeph DBA Blue Archimedes LLC' -> ('Blue Archimedes LLC', 'Nexsolzeph').

    In this data the real business name follows the marker and the part
    before it is a synthetic brand, so the part after is returned first.
    """
    parts = ALIAS_MARKER_RE.split(name, maxsplit=1)
    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
        return parts[1].strip(), parts[0].strip()
    return name, ""


def parse_name(name: str) -> NameParts:
    raw = clean_unicode(name or "")
    if not raw:
        return NameParts()
    main, alt = split_alias(raw)
    tokens, is_domain = _basic_tokens(main)
    alt_tokens = _basic_tokens(alt)[0] if alt else []
    core = _core(tokens)
    return NameParts(
        name_norm=" ".join(tokens),
        name_core=" ".join(core),
        name_legal=" ".join(sorted({t for t in tokens if t in LEGAL_FORMS})),
        name_alt=" ".join(alt_tokens),
        name_is_domain=is_domain,
        name_known_ratio=_known_ratio(core),
        name_script=detect_script(raw),
        name_phonetic=_phonetic(core),
    )


def normalize_name(name: str) -> str:
    """'ABC Technologies Pvt. Ltd.' -> 'abc technologies pvt ltd'."""
    return parse_name(name).name_norm


def basic_name_tokens(name: str) -> list[str]:
    """Light tokenizer (no resources): used to build the vocab and mine token maps."""
    text = fold_accents(clean_unicode(name or "")).lower().replace("&", " and ")
    text = re.sub(r"['`´]", "", text)
    return _NON_ALNUM_RE.sub(" ", text).split()
