"""Script-agnostic text utilities: Unicode cleanup, accent folding, script
detection, and rule-based transliteration of Indic (Brahmic) scripts.

All Brahmic scripts in Unicode (Devanagari, Bengali, Gurmukhi, Gujarati, Odia,
Tamil, Telugu, Kannada, Malayalam) follow the same ISCII-derived layout: each
block is 0x80 wide and a given offset encodes the same phonetic unit in every
script. One offset table therefore transliterates all of them.
"""
from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------
# Scripts
# --------------------------------------------------------------------------
INDIC_BLOCKS = {
    0x0900: "devanagari", 0x0980: "bengali", 0x0A00: "gurmukhi", 0x0A80: "gujarati",
    0x0B00: "odia", 0x0B80: "tamil", 0x0C00: "telugu", 0x0C80: "kannada", 0x0D00: "malayalam",
}
# Scripts where a word-final inherent vowel is normally silent (Hindi "राम" = "ram").
_SCHWA_DELETING = {"devanagari", "gujarati", "gurmukhi", "bengali"}

_INDIC_RE = re.compile(r"[ऀ-ൿ]")


def indic_script(ch: str) -> str | None:
    cp = ord(ch)
    if 0x0900 <= cp <= 0x0D7F:
        return INDIC_BLOCKS[cp & ~0x7F]
    return None


def has_indic(text: str) -> bool:
    return bool(_INDIC_RE.search(text))


def detect_script(text: str) -> str:
    """'latin', 'indic:<script>', 'mixed', or 'other' for the letters in ``text``."""
    scripts = set()
    for ch in text:
        if not ch.isalpha():
            continue
        s = indic_script(ch)
        if s:
            scripts.add("indic:" + s)
        elif ch.isascii() or unicodedata.name(ch, "").startswith("LATIN"):
            scripts.add("latin")
        else:
            scripts.add("other")
    if not scripts:
        return "latin"
    return scripts.pop() if len(scripts) == 1 else "mixed"


# --------------------------------------------------------------------------
# Unicode cleanup and accent folding
# --------------------------------------------------------------------------
_ZERO_WIDTH_RE = re.compile(r"[​-‏⁠﻿­]")
_SPECIAL_FOLD = str.maketrans({
    "ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ß": "ss",
    "đ": "d", "Đ": "D", "ł": "l", "Ł": "L", "ı": "i", "þ": "th", "ð": "d",
    "‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "–": "-", "—": "-", "−": "-",
    "°": "o",  # French "N°" -> "No"
})
_WS_RE = re.compile(r"\s+")


def clean_unicode(text: str) -> str:
    """NFKC-normalise, drop zero-width/soft-hyphen chars, collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    return _WS_RE.sub(" ", text).strip()


def fold_accents(text: str) -> str:
    """'Léarning Ésq Gúild' -> 'Learning Esq Guild'. Indic scripts are left intact."""
    if text.isascii():
        return text
    text = text.translate(_SPECIAL_FOLD)
    out = []
    for ch in unicodedata.normalize("NFD", text):
        if unicodedata.combining(ch) and not (0x0900 <= ord(ch) <= 0x0D7F):
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))


def squash_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


# --------------------------------------------------------------------------
# Indic transliteration (offset-based, shared by all Brahmic scripts)
# --------------------------------------------------------------------------
_VOWELS = {  # independent vowels
    0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ii", 0x09: "u", 0x0A: "uu", 0x0B: "ri",
    0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o",
    0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "li",
}
_MATRAS = {  # dependent vowel signs
    0x3E: "aa", 0x3F: "i", 0x40: "ii", 0x41: "u", 0x42: "uu", 0x43: "ri", 0x44: "ri",
    0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o",
    0x4C: "au", 0x62: "li", 0x63: "li", 0x55: "", 0x56: "ai", 0x57: "au",
}
_CONSONANTS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng", 0x1A: "ch", 0x1B: "chh",
    0x1C: "j", 0x1D: "jh", 0x1E: "ny", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh",
    0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r",
    0x31: "r", 0x32: "l", 0x33: "l", 0x34: "zh", 0x35: "v", 0x36: "sh", 0x37: "sh",
    0x38: "s", 0x39: "h",
    0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y",
}
_NUKTA_FORMS = {0x15: "q", 0x16: "kh", 0x17: "g", 0x1C: "z", 0x21: "r", 0x22: "rh", 0x2B: "f", 0x2F: "y"}
_MALAYALAM_CHILLU = {0x7A: "n", 0x7B: "n", 0x7C: "r", 0x7D: "l", 0x7E: "l", 0x7F: "k"}
_LABIALS = ("p", "b", "m")
_LONG_VOWEL_RE = re.compile(r"([aiu])\1")
_VIRAMA, _NUKTA, _ANUSVARA, _CANDRABINDU, _VISARGA = 0x4D, 0x3C, 0x02, 0x01, 0x03


def _transliterate_word(word: str, script: str | None) -> str:
    out: list[str] = []
    pending_a = False  # consonant emitted, inherent 'a' not yet resolved
    n = len(word)
    i = 0
    while i < n:
        ch = word[i]
        cp = ord(ch)
        if not (0x0900 <= cp <= 0x0D7F):
            if pending_a:
                out.append("a")
                pending_a = False
            out.append(ch)
            i += 1
            continue
        block = cp & ~0x7F
        off = cp - block
        if block == 0x0D00 and off in _MALAYALAM_CHILLU:
            if pending_a:
                out.append("a")
            out.append(_MALAYALAM_CHILLU[off])
            pending_a = False
        elif block == 0x0A00 and off == 0x70:      # Gurmukhi tippi = nasal
            if pending_a:
                out.append("a")
            out.append("n")
            pending_a = False
        elif block == 0x0A00 and off == 0x71:      # Gurmukhi addak = gemination
            pass
        elif block == 0x0D00 and off == 0x31 and word[i + 1:i + 3] == "്റ":
            if pending_a:                          # Malayalam "റ്റ" is /tt/, not /rr/
                out.append("a")
            out.append("tt")
            pending_a = True
            i += 2
        elif block == 0x0D00 and off == 0x31 and word[i - 2:i] == "ന്":
            out.append("t")                        # Malayalam "ന്റ" is /nt/
            pending_a = True
        elif block == 0x0B80 and off == _VISARGA and word[i + 1:i + 2] == "ப":
            if pending_a:                          # Tamil aytham + pa = /f/
                out.append("a")
            out.append("f")
            pending_a = True
            i += 1
        elif block == 0x0B00 and off == 0x71:      # Odia wa
            if pending_a:
                out.append("a")
            out.append("w")
            pending_a = True
        elif off in _CONSONANTS:
            if pending_a:
                out.append("a")
            if i + 1 < n and ord(word[i + 1]) == block + _NUKTA and off in _NUKTA_FORMS:
                out.append(_NUKTA_FORMS[off])
                i += 1
            else:
                out.append(_CONSONANTS[off])
            pending_a = True
        elif off in _MATRAS:
            out.append(_MATRAS[off])
            pending_a = False
        elif off == _VIRAMA:
            pending_a = False
        elif off in _VOWELS:
            if pending_a:
                out.append("a")
            out.append(_VOWELS[off])
            pending_a = False
        elif off in (_ANUSVARA, _CANDRABINDU):
            if pending_a:
                out.append("a")
            nxt = word[i + 1] if i + 1 < n else ""
            nxt_c = _CONSONANTS.get(ord(nxt) - block, "") if nxt else ""
            final_m = not nxt and script in ("malayalam", "tamil")  # "കേരളം" = keralam
            out.append("m" if final_m or nxt_c.startswith(_LABIALS) else "n")
            pending_a = False
        elif off == _VISARGA:
            if pending_a:
                out.append("a")
            out.append("h")
            pending_a = False
        elif 0x66 <= off <= 0x6F:                  # native digits
            if pending_a:
                out.append("a")
            out.append(str(off - 0x66))
            pending_a = False
        # nukta on its own, avagraha, other marks: dropped
        i += 1
    if pending_a and script not in _SCHWA_DELETING:
        out.append("a")
    return _LONG_VOWEL_RE.sub(r"\1", "".join(out))  # 'raam' -> 'ram' (closer to English spelling)


_WORD_RE = re.compile(r"[ऀ-ൿ]+")


def transliterate_indic(text: str) -> str:
    """Rule-based romanisation of every Brahmic-script run in ``text``.

    'राम मार्केटिंग' -> 'ram maarketing', 'సాయి' -> 'saayi'. Latin text passes
    through unchanged. The output is phonetic, not English spelling, so
    downstream code should prefer the mined dictionary in
    ``resources/indic_token_map.tsv`` and fall back to this.
    """
    if not has_indic(text):
        return text
    return _WORD_RE.sub(lambda m: _transliterate_word(m.group(0), indic_script(m.group(0)[0])), text)


# --------------------------------------------------------------------------
# Phonetic skeleton (bridges transliteration and English spelling)
# --------------------------------------------------------------------------
_PHON_SUBS = [
    (re.compile(r"ph"), "f"), (re.compile(r"[wv]"), "v"), (re.compile(r"(?<=[kgcjtdpbsz])h"), ""),
    (re.compile(r"ck|q"), "k"), (re.compile(r"c(?=[eiy])"), "s"), (re.compile(r"c"), "k"),
    (re.compile(r"z"), "s"), (re.compile(r"x"), "ks"), (re.compile(r"[aeiouy]"), ""),
    (re.compile(r"(.)\1+"), r"\1"),
]


def phonetic_key(token: str) -> str:
    """Vowel-less consonant skeleton: 'software' and 'saphtaveyar' -> 'sftvr'.

    Keeps the first letter even if it is a vowel so 'apex' != 'pax'.
    """
    token = token.lower()
    if not token:
        return ""
    key = token
    for pat, rep in _PHON_SUBS:
        key = pat.sub(rep, key)
    if token[0] in "aeiou":
        key = token[0] + key
    return key
