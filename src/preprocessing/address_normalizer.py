"""Business-address normalisation.

``normalize_address(address, country)`` returns one canonical string;
``parse_address(address, country)`` returns every derived field
(see ``AddressParts``). ``country`` is an open-set label: unknown countries
get the generic rules; US / India / France additionally get their own
abbreviation and state tables.

Noise patterns handled (abbreviation pairs were mined from train Source 1 vs
its matched Source 2/3 records; see reports/eda_report.md):

* CAPS, injected accents, Indic-script state names ("ಕರ್ನಾಟಕ")
* street-type abbreviations (St/Rd/Dr/Ave/Blvd/..., French R./Bd/Av/Rte/...)
  and one-edit typos of them ("steet", "drve", "nagr", "foor")
* state code vs full name vs old name (TX/Texas, KA/Karnataka, Orissa/Odisha)
* house-number markers (#, No., H.no, Door No, N°) and zero padding ("00478/1")
* injected PMB boxes, repeated tokens ("Unit Unit"), placeholder
  components ("<NULL>", "N/A"), renamed cities (Bombay -> Mumbai)
* component re-ordering: state/city/street are also returned as separate fields
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from functools import lru_cache

from .gazetteer import AMBIGUOUS_STATE_CITY, CITY_ALIASES, STATE_MAPS, canonical_state, normalize_country
from .resources import address_vocab, native_state_map
from .text_utils import clean_unicode, fold_accents, has_indic, phonetic_key, transliterate_indic

# ------------------------------------------------------------------ tables
COMMON_ABBREV = {
    "rd": "road", "ave": "avenue", "av": "avenue", "avn": "avenue", "blvd": "boulevard",
    "dr": "drive", "drv": "drive", "ln": "lane", "ct": "court", "crt": "court", "cir": "circle",
    "pl": "place", "trl": "trail", "hwy": "highway", "ter": "terrace", "terr": "terrace",
    "pkwy": "parkway", "pky": "parkway", "cv": "cove", "sq": "square", "rdg": "ridge",
    "tpke": "turnpike", "expy": "expressway", "fwy": "freeway", "aly": "alley", "xing": "crossing",
    "cres": "crescent", "hts": "heights", "jct": "junction", "mtn": "mountain", "pt": "point",
    "ft": "fort", "mt": "mount", "apt": "apartment", "apts": "apartments", "bldg": "building",
    "fl": "floor", "flr": "floor", "rm": "room", "dept": "department", "twp": "township",
    "twnship": "township", "townhsip": "township", "cty": "city", "blk": "block",
    "nr": "near", "opp": "opposite", "stn": "station", "flt": "flat", "plt": "plot",
    "ngr": "nagar", "clny": "colony", "col": "colony", "sec": "sector", "sect": "sector",
    "ph": "phase", "grnd": "ground", "gf": "ground floor", "ff": "first floor",
    "mkt": "market", "extn": "extension", "ext": "extension", "indl": "industrial",
    "ind": "industrial", "estt": "estate", "comp": "complex", "cplx": "complex", "chs": "society",
    "soc": "society", "vill": "village", "vlg": "village", "po": "post office", "dist": "district",
    "distt": "district", "tq": "taluk", "tal": "taluk", "tk": "taluk",
    "n": "north", "s": "south", "e": "east", "w": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest",
}
# Direction letters are only expanded for the US (in India "138 N" is a house number suffix).
_DIRECTIONS = {"n", "s", "e", "w", "ne", "nw", "se", "sw"}
COUNTRY_ABBREV = {
    "us": {"ste": "suite", "st": "street"},
    "india": {"st": "street", "mg": "mahatma gandhi"},
    "france": {"r": "rue", "bd": "boulevard", "bld": "boulevard", "boul": "boulevard",
               "rte": "route", "chem": "chemin", "ch": "chemin", "imp": "impasse", "all": "allee",
               "crs": "cours", "fg": "faubourg", "fbg": "faubourg", "qu": "quai", "st": "saint",
               "ste": "sainte", "pl": "place", "av": "avenue", "sq": "square", "res": "residence",
               "resid": "residence", "lot": "lotissement", "za": "zone artisanale",
               "zi": "zone industrielle", "zac": "zone amenagement concerte", "ter": "ter", "bis": "bis",
               "q": "quai", "psg": "passage", "pass": "passage", "appt": "appartement", "app": "appartement",
               "apt": "appartement", "appart": "appartement", "apartment": "appartement"},
}
# French house-number suffixes written as one letter after the number ("10 B", "17 T").
_FR_NUMBER_SUFFIX = {"b": "bis", "t": "ter"}
# After "st", these mean the "st" was "street" ("Main St Apt 4", "Elm St N").
_AFTER_STREET = {"apt", "apartment", "suite", "ste", "unit", "fl", "floor", "rm", "room", "bldg",
                 "building", "north", "south", "east", "west", "n", "s", "e", "w", "ne", "nw",
                 "se", "sw", "extension", "ext"}
# House-number markers; the number itself is kept.
HOUSE_MARKERS = [("house", "no"), ("door", "no"), ("h", "no"), ("d", "no"), ("hno",),
                 ("dno",), ("no",), ("number",), ("num",)]
PLACEHOLDERS = {"", "null", "none", "nil", "na", "n a", "nan", "unknown", "not available", "0", "x"}
# Closed vocabulary of frequent address words for one-edit typo repair.
TYPO_VOCAB = {
    "street", "road", "drive", "avenue", "lane", "court", "circle", "place", "boulevard",
    "trail", "highway", "terrace", "parkway", "square", "ridge", "township", "city", "county",
    "north", "south", "east", "west", "suite", "floor", "building", "apartment", "nagar",
    "colony", "plot", "flat", "house", "sector", "phase", "layout", "cross", "main", "near",
    "opposite", "market", "complex", "estate", "industrial", "village", "district", "ground",
    "station", "society", "tower", "plaza", "center", "centre", "park", "garden", "gardens",
    "extension", "block", "chowk", "marg", "bazar", "bazaar", "puram", "palayam", "halli",
    "rue", "chemin", "impasse", "allee", "route", "cours", "quai", "faubourg", "residence",
}

_NUMERO_RE = re.compile(r"\bn\s*[°º]\s*", re.IGNORECASE)   # French "N°43"
_PMB_RE = re.compile(r"\b(?:pmb|p\.?\s?o\.?\s?box|po box|box)\s*#?\s*\d+\b", re.IGNORECASE)
_NUM_RE = re.compile(r"\d+")
_LEADING_ZERO_RE = re.compile(r"\b0+(?=\d)")
_HOUSE_NO_RE = re.compile(r"(?<![\w/-])([a-z]?[-/]?\d+[a-z]?(?:[-/]\d+[a-z]?)*)\b", re.IGNORECASE)
_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")
_POSTCODE_RES = {
    "india": re.compile(r"(?<![\d/-])([1-9]\d{5})\b"),
    "us": re.compile(r"\b(\d{5})(?:-\d{4})?\b"),
    "france": re.compile(r"\b((?:0[1-9]|[1-8]\d|9[0-5]|2a|2b)\d{3})\b", re.IGNORECASE),
}


@dataclass
class AddressParts:
    address_norm: str = ""   # full normalised address, state as canonical code
    addr_street: str = ""    # first component with a house number / street, normalised
    addr_city: str = ""      # last non-state, non-numeric component
    addr_state: str = ""     # canonical state / region code ('' if unknown)
    addr_house_no: str = ""  # first house-number-like token, zero padding removed
    addr_numbers: str = ""   # all numeric runs (sorted, unique, space-separated)
    addr_postcode: str = ""  # postcode if present (country-specific pattern)
    addr_is_missing: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- helpers
@lru_cache(maxsize=1)
def _typo_index() -> dict[str, str]:
    """Symmetric-delete index: every 1-char deletion of a vocab word -> word."""
    idx: dict[str, str] = {}
    for w in sorted(TYPO_VOCAB):
        idx.setdefault(w, w)
        for i in range(len(w)):
            idx.setdefault(w[:i] + w[i + 1:], w)
    return idx


@lru_cache(maxsize=200_000)
def _fix_typo(tok: str) -> str:
    """'steet' -> 'street', 'drve' -> 'drive', 'nagr' -> 'nagar' (one edit, closed vocab)."""
    if len(tok) < 4 or tok in TYPO_VOCAB or not tok.isalpha():
        return tok
    idx = _typo_index()
    cand = idx.get(tok)                   # tok is a deletion of a word
    if not cand and len(tok) >= 5:        # short real words ("lake", "mall") look like typos
        for i in range(len(tok)):         # word is a deletion of tok / substitution / swap
            c = idx.get(tok[:i] + tok[i + 1:])
            if c and abs(len(c) - len(tok)) <= 1:
                cand = c
                break
    if not cand:
        return tok
    # A real word ("worth" vs "north") is common in clean S1 addresses; a typo is
    # rare relative to the word it corrupts ("steet": 82 vs "street": 480k).
    vocab = address_vocab()
    return cand if vocab.get(tok, 0) < 0.005 * vocab.get(cand, 0) else tok


def _strip_house_markers(tokens: list[str]) -> list[str]:
    out, i = [], 0
    while i < len(tokens):
        for m in HOUSE_MARKERS:
            j = i + len(m)
            if tuple(tokens[i:j]) == m and j < len(tokens) and (
                    tokens[j][:1].isdigit()
                    or (len(tokens[j]) == 1 and j + 1 < len(tokens) and tokens[j + 1][:1].isdigit())):
                i += len(m)
                break
        else:
            out.append(tokens[i])
            i += 1
    return out


def _expand(tokens: list[str], country: str) -> list[str]:
    local = COUNTRY_ABBREV.get(country, {})
    out = []
    for k, t in enumerate(tokens):
        if country == "france" and t in _FR_NUMBER_SUFFIX and k and tokens[k - 1].isdigit():
            t = _FR_NUMBER_SUFFIX[t]                  # "10 B Rue ..." -> "10 bis rue ..."
        elif t == "st" and country != "france":
            # "St Louis", "123 St Marks Pl" -> saint; "Main St", "Main St Apt 4" -> street
            nxt = tokens[k + 1] if k + 1 < len(tokens) else ""
            t = "saint" if nxt and not nxt[0].isdigit() and nxt not in _AFTER_STREET else "street"
        elif t in local:
            t = local[t]
        elif t in _DIRECTIONS and country != "us":
            pass
        elif t in COMMON_ABBREV:
            t = COMMON_ABBREV[t]
        else:
            t = CITY_ALIASES.get(t) or _fix_typo(t)
        out.extend(t.split())
    return out


def _dedupe_consecutive(tokens: list[str]) -> list[str]:
    return [t for k, t in enumerate(tokens) if k == 0 or t != tokens[k - 1]]


def _component_tokens(comp: str, country: str) -> list[str]:
    comp = _LEADING_ZERO_RE.sub("", comp.replace("#", " # "))
    tokens = _NON_ALNUM_RE.sub(" ", comp).split()
    tokens = _strip_house_markers(tokens)
    return _dedupe_consecutive(_expand(tokens, country))


@lru_cache(maxsize=None)
def _state_phonetics(country: str) -> dict[str, str]:
    """phonetic key of every multi-letter state spelling -> code (for transliterations)."""
    out: dict[str, str] = {}
    for name, code in STATE_MAPS.get(country, {}).items():
        if len(name) > 3:
            out.setdefault(phonetic_key(name.replace(" ", "")), code)
    return out


def _native_state(comp_raw: str, romanized_key: str, country: str) -> str:
    """State code of an Indic-script component: mined spelling map, else phonetic match."""
    hit = native_state_map().get(comp_raw.strip())
    if hit and (hit[0] == country or country not in STATE_MAPS):
        return hit[1]
    return _state_phonetics(country).get(phonetic_key(romanized_key.replace(" ", "")), "")


# -------------------------------------------------------------------- API
def parse_address(address: str, country: str = "") -> AddressParts:
    country = normalize_country(country)
    raw = clean_unicode(address or "")
    raw = _NUMERO_RE.sub(" no ", _PMB_RE.sub(" ", raw))
    comps_raw = [c.strip() for c in raw.split(",")]
    comps: list[list[str]] = []
    state = ""
    state_idx = -1
    last = max((k for k, c in enumerate(comps_raw) if c), default=-1)
    fallback_state = ""
    for k, comp in enumerate(comps_raw):
        if not comp:
            continue
        indic = has_indic(comp)
        text = fold_accents(transliterate_indic(comp) if indic else comp).lower()
        key = " ".join(_NON_ALNUM_RE.sub(" ", text).split())
        if key in PLACEHOLDERS:
            continue
        code = _native_state(comp, key, country) if indic else canonical_state(key, country)
        if code and key in AMBIGUOUS_STATE_CITY and k != last:
            fallback_state = fallback_state or code  # used only if no real state is found
            code = ""                             # "New Delhi, West Delhi, Delhi": city, not state
        if code:
            state, state_idx = code, len(comps)   # last state-like component wins
            comps.append([code])
            continue
        toks = _component_tokens(text, country)
        if toks:
            comps.append(toks)
    if not comps:
        return AddressParts()
    state = state or fallback_state             # "Delhi, New Delhi, R-16 Model Town": still Delhi

    flat = [t for c in comps for t in c]
    norm = " ".join(_dedupe_consecutive(flat))
    non_state = [c for k, c in enumerate(comps) if k != state_idx]
    street = next((c for c in non_state if any(t[0].isdigit() for t in c)), [])
    city = next((c for c in reversed(non_state) if not any(t[0].isdigit() for t in c)), [])
    city = [t for t in city if t != "cdp"]               # "Prince Frederick CDP"

    folded = fold_accents(raw).lower()
    folded = _LEADING_ZERO_RE.sub("", folded)
    house = _HOUSE_NO_RE.search(folded.replace("#", " "))
    postcode = ""
    pc_re = _POSTCODE_RES.get(country)
    if pc_re:
        m = None
        for m in pc_re.finditer(folded):
            pass  # last match: postcodes sit at the end
        if m and (country != "us" or m.start() > len(folded) // 2):
            postcode = "".join(g for g in m.groups() if g)
    numbers = sorted(set(_NUM_RE.findall(norm)), key=lambda x: (len(x), x))
    return AddressParts(
        address_norm=norm,
        addr_street=" ".join(street),
        addr_city=" ".join(city),
        addr_state=state,
        addr_house_no=house.group(1).strip("-/") if house else "",
        addr_numbers=" ".join(numbers),
        addr_postcode=postcode,
        addr_is_missing=False,
    )


def normalize_address(address: str, country: str = "") -> str:
    """'2238 WILLIAMSBURG RD, GLEN DALE, West Virginia' -> '2238 williamsburg road glen dale wv'."""
    return parse_address(address, country).address_norm
