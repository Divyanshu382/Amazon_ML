"""Static, offline tables of country / state / city spellings.

These are generic naming conventions (postal abbreviations, official region
names, renamed cities) typed in by hand -- not business-identity data and not
an online lookup. Country is treated as an OPEN set: unknown labels pass
through ``normalize_country`` unchanged (lower-cased), and every function
here degrades gracefully for a country it has no table for.

State tables are kept per country because codes collide across countries
(TN = Tennessee / Tamil Nadu, GA = Georgia / Goa, CT = Connecticut /
Chhattisgarh, ...).
"""
from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------- countries
_COUNTRY_ALIASES = {
    "us": "us", "usa": "us", "u s": "us", "u s a": "us", "united states": "us",
    "united states of america": "us", "america": "us",
    "in": "india", "ind": "india", "india": "india", "bharat": "india",
    "fr": "france", "fra": "france", "france": "france", "republique francaise": "france",
}


def _key(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def normalize_country(country: str) -> str:
    """Map known aliases to a canonical label; anything else -> its lower-cased key."""
    k = _key(country or "")
    return _COUNTRY_ALIASES.get(k, k)


# ------------------------------------------------------------------- states
US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "dc": "district of columbia",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho", "il": "illinois",
    "in": "indiana", "ia": "iowa", "ks": "kansas", "ky": "kentucky", "la": "louisiana",
    "me": "maine", "md": "maryland", "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
    "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon",
    "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina", "sd": "south dakota",
    "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia",
    "wa": "washington", "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming",
    "pr": "puerto rico", "gu": "guam", "vi": "virgin islands",
}
_US_EXTRA = {"washington dc": "dc", "d c": "dc", "wash": "wa", "calif": "ca", "tex": "tx",
             "penn": "pa", "mass": "ma", "fla": "fl"}

INDIA_STATES = {  # canonical code -> (canonical name, other spellings / codes)
    "an": ("andaman and nicobar islands", ["andaman nicobar", "andaman and nicobar"]),
    "ap": ("andhra pradesh", []),
    "ar": ("arunachal pradesh", []),
    "as": ("assam", []),
    "br": ("bihar", ["bh"]),
    "ch": ("chandigarh", []),
    "cg": ("chhattisgarh", ["ct", "chattisgarh", "chhatisgarh"]),
    "dn": ("dadra and nagar haveli and daman and diu",
           ["dadra and nagar haveli", "daman and diu", "dd", "dh"]),
    "dl": ("delhi", ["new delhi", "nct of delhi", "nct delhi"]),
    "ga": ("goa", []),
    "gj": ("gujarat", ["gujrat"]),
    "hr": ("haryana", []),
    "hp": ("himachal pradesh", []),
    "jk": ("jammu and kashmir", ["jammu kashmir", "j and k", "j k"]),
    "jh": ("jharkhand", ["jd"]),
    "ka": ("karnataka", ["kar"]),
    "kl": ("kerala", ["keralam", "ke"]),
    "la": ("ladakh", []),
    "ld": ("lakshadweep", []),
    "mp": ("madhya pradesh", []),
    "mh": ("maharashtra", ["maharastra"]),
    "mn": ("manipur", []),
    "ml": ("meghalaya", []),
    "mz": ("mizoram", []),
    "nl": ("nagaland", []),
    "od": ("odisha", ["orissa", "or"]),
    "py": ("puducherry", ["pondicherry", "pondy"]),
    "pb": ("punjab", []),
    "rj": ("rajasthan", []),
    "sk": ("sikkim", []),
    "tn": ("tamil nadu", ["tamilnadu"]),
    "tg": ("telangana", ["ts", "tl"]),
    "tr": ("tripura", []),
    "up": ("uttar pradesh", []),
    "uk": ("uttarakhand", ["uttaranchal", "ut"]),
    "wb": ("west bengal", ["bengal"]),
}

# France: metropolitan regions, plus departments mapped to their region so that
# "Tourcoing, Hauts-de-France" and "Tourcoing, Nord" agree on the state field.
FRANCE_REGIONS = {
    "ara": "auvergne rhone alpes", "bfc": "bourgogne franche comte", "bre": "bretagne",
    "cvl": "centre val de loire", "cor": "corse", "ges": "grand est", "hdf": "hauts de france",
    "idf": "ile de france", "nor": "normandie", "naq": "nouvelle aquitaine", "occ": "occitanie",
    "pdl": "pays de la loire", "pac": "provence alpes cote d azur",
}
FRANCE_DEPARTMENTS = {
    "hdf": ["aisne", "nord", "oise", "pas de calais", "somme"],
    "naq": ["charente", "charente maritime", "correze", "creuse", "dordogne", "gironde", "landes",
            "lot et garonne", "pyrenees atlantiques", "deux sevres", "vienne", "haute vienne"],
    "pdl": ["loire atlantique", "maine et loire", "mayenne", "sarthe", "vendee"],
    "idf": ["paris", "seine et marne", "yvelines", "essonne", "hauts de seine", "seine saint denis",
            "val de marne", "val d oise"],
    "bre": ["cotes d armor", "finistere", "ille et vilaine", "morbihan"],
    "nor": ["calvados", "eure", "manche", "orne", "seine maritime"],
    "occ": ["haute garonne", "herault", "gard", "aude", "pyrenees orientales", "tarn"],
    "pac": ["bouches du rhone", "alpes maritimes", "var", "vaucluse"],
    "ara": ["rhone", "isere", "haute savoie", "savoie", "loire", "ain", "puy de dome"],
    "ges": ["bas rhin", "haut rhin", "moselle", "marne", "meurthe et moselle"],
}
_FR_EXTRA = {"provence alpes cote dazur": "pac", "paca": "pac", "rhone alpes": "ara",
             "aquitaine": "naq", "nord pas de calais": "hdf", "picardie": "hdf",
             "region parisienne": "idf"}


def _build_state_maps() -> dict[str, dict[str, str]]:
    us = {}
    for code, name in US_STATES.items():
        us[code] = code
        us[name] = code
    us.update(_US_EXTRA)
    india = {}
    for code, (name, alts) in INDIA_STATES.items():
        india[code] = code
        india[name] = code
        for a in alts:
            india[a] = code
    fr = {}
    for code, name in FRANCE_REGIONS.items():
        fr[name] = code
    for code, depts in FRANCE_DEPARTMENTS.items():
        for d in depts:
            fr[d] = code
    fr.update(_FR_EXTRA)
    return {"us": us, "india": india, "france": fr}


STATE_MAPS = _build_state_maps()
# Full names only (no 2-letter codes): safe to use when the country is unknown.
_ANY_COUNTRY_STATES = {k: f"{c}:{v}" for c, m in STATE_MAPS.items() for k, v in m.items() if len(k) > 3}

# "delhi" is both a state and a city: keep it as a state only when it is the
# last address component (handled by the address parser).
AMBIGUOUS_STATE_CITY = {"delhi", "new delhi", "paris", "var", "manche", "loire", "rhone"}


def canonical_state(component: str, country: str) -> str:
    """Return the canonical state code for an address component ('' if not a state).

    ``component`` must already be accent-folded, lower-cased and punctuation
    free. For a country without a table only full names are matched, and the
    result is prefixed with the matching country ("india:ka").
    """
    table = STATE_MAPS.get(country)
    if table is not None:
        return table.get(component, "")
    return _ANY_COUNTRY_STATES.get(component, "")


# ------------------------------------------------------------ city aliases
# Renamed / historically-spelled cities. Applied to address tokens.
CITY_ALIASES = {
    "bombay": "mumbai", "calcutta": "kolkata", "madras": "chennai", "bengaluru": "bangalore",
    "gurugram": "gurgaon", "poona": "pune", "mysuru": "mysore", "trivandrum": "thiruvananthapuram",
    "cochin": "kochi", "baroda": "vadodara", "vizag": "visakhapatnam", "benares": "varanasi",
    "banaras": "varanasi", "allahabad": "prayagraj", "cawnpore": "kanpur", "pondicherry": "puducherry",
    "mangaluru": "mangalore", "belagavi": "belgaum", "kalaburagi": "gulbarga", "hubballi": "hubli",
    "shivamogga": "shimoga", "tumakuru": "tumkur", "calicut": "kozhikode",
    "trichur": "thrissur", "quilon": "kollam", "alleppey": "alappuzha", "ooty": "udhagamandalam",
    "tuticorin": "thoothukudi", "trichy": "tiruchirappalli", "tanjore": "thanjavur",
    "atl": "atlanta", "nyc": "new york",
}
