"""Unit tests for the preprocessing layer. Examples are real noise patterns
taken from the training data (Source 1 record vs its matched S2/S3 records).

Run: .venv/bin/python -m pytest tests -q
"""
import pytest

from src.preprocessing.address_normalizer import normalize_address, parse_address
from src.preprocessing.gazetteer import canonical_state, normalize_country
from src.preprocessing.name_normalizer import normalize_name, parse_name, split_alias
from src.preprocessing.preprocessing import OUTPUT_COLUMNS, preprocess_record
from src.preprocessing.resources import indic_token_map
from src.preprocessing.text_utils import (detect_script, fold_accents, phonetic_key,
                                          transliterate_indic)

needs_resources = pytest.mark.skipif(not indic_token_map(), reason="run mine_resources first")


# ------------------------------------------------------------------ names
@pytest.mark.parametrize("raw, expected", [
    ("ABC Technologies Pvt. Ltd.", "abc technologies pvt ltd"),
    ("Esquire Marketing Private Limited", "esquire marketing pvt ltd"),
    ("Esquire Marketing Private Ltd", "esquire marketing pvt ltd"),
    ("Rushing, Sherrill & Hanson L.L.C.", "rushing sherrill and hanson llc"),
    ("WHIPPLE + WILLIAMS ACQUISITION,-LLC #60562", "whipple and williams acquisition llc"),
    ("Tri-State Gúild (ID: 93967)", "tri state guild"),
    ("Eye Gróup of Atlanta Incorporated", "eye group of atlanta inc"),
    ("Star Fx [LLC]", "star fx llc"),
    ("royal royal surya [mánagement]", "royal surya management"),
    ("complete optimal pro llc | www.completeo.com", "complete optimal pro llc"),
    ("Fractales Amis Groupe S.A.S", "fractales amis groupe sas"),
    ("Établissements Durand Compagnie", "ets durand cie"),
    ("", ""),
])
def test_normalize_name(raw, expected):
    assert normalize_name(raw) == expected


def test_alias_marker_keeps_real_name_after_marker():
    assert split_alias("Nexsolzeph DBA Blue Archimedes LLC") == ("Blue Archimedes LLC", "Nexsolzeph")
    p = parse_name("Yumaflux Formerly Tri-State Guild")
    assert p.name_norm == "tri state guild" and p.name_alt == "yumaflux"
    assert split_alias("Aka Construction Pvt Ltd") == ("Aka Construction Pvt Ltd", "")
    for raw in ("Veoxylomirax D.B.A. North Logan Dental", "Vioxylo DBA: North Logan Dental",
                "Fayebelo formerly: North Logan Dental", "Zetafaye n\u00e9e North Logan Dental",
                "Jaxavi f/k/a North Logan Dental"):
        assert split_alias(raw)[0] == "North Logan Dental", raw


def test_core_and_legal_fields():
    p = parse_name("Smt PRETTY CONSTRUCTIONS (CORP)")
    assert p.name_core == "pretty constructions"
    assert p.name_legal == "corp"
    assert parse_name("LLC").name_core == "llc"        # never empty


def test_caps_l_is_i():
    assert normalize_name("RUSHlN, JOHNSON & NISHIOKA ASSOCIATES").startswith("rushin johnson")


@needs_resources
@pytest.mark.parametrize("raw, expected", [
    ("5tar Flxe LLC", "star flxe llc"),
    ("Celestial Mem0rial Trust Corp", "celestial memorial trust corp"),
    ("6lobal Steel Pvt Limited", "global steel pvt ltd"),
    ("tristateguild.com", "tri state guild"),
    ("celestialmemorialtrust.com", "celestial memorial trust"),
    ("सन सॉफ्टवेयर लिमिटेड", "sun software ltd"),
    ("रॉयल सूर्य मैनेजमेंट", "royal surya management"),
    ("Sky હેલ્થકેર Private Limited", "sky healthcare pvt ltd"),
    ("ଅରିହନ୍ତ ପ୍ରୋପର୍ଟିଜ୍", "arihant properties"),
    ("ವಿಜಯ್ ಯುನಿವರ್ಸಲ್ ಎಕ್ಸ್‌ಪೋರ್ಟ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್", "vijay universal exports pvt ltd"),
])
def test_names_with_mined_resources(raw, expected):
    assert normalize_name(raw) == expected


def test_domain_flag():
    assert parse_name("tristateguild.com").name_is_domain
    assert not parse_name("Tri-State Guild").name_is_domain
    assert not parse_name("#4 Presidio").name_is_domain


@needs_resources
@pytest.mark.parametrize("raw, expected", [
    ("#Rájkotbiotech", "rajkot biotech"),
    ("@tulsacatering", "tulsa catering"),
    ("#1329main", "1329 main"),
    ("M/s shubhamup.com", "ms shubham up"),
    ("orbo1adome.com", "orbola dome"),
    ("havelis0cial.com | www.havelisoci.com", "haveli social"),
    ("Drexaria+", "drexaria"),
    ("mapessuperiorresourcecom", "mapes superior resource"),
    ("sunsoftwarelimited", "sun software ltd"),
    ("Vantagezeph", "vantagezeph"),       # brand words are not split into fragments
    ("Miranovivera", "miranovivera"),
    ("Fayevantagequo", "fayevantagequo"),
    ("kalicharanduttaroad", "kali charan dutta road"),
])
def test_glued_handles_and_domains(raw, expected):
    assert normalize_name(raw) == expected


@needs_resources
def test_known_ratio_flags_brand_only_names():
    assert parse_name("Drexvio").name_known_ratio == 0.0
    assert parse_name("Rizairidrex Co").name_known_ratio == 0.0
    assert parse_name("Eye Group of Atlanta Inc.").name_known_ratio == 1.0


# -------------------------------------------------------------- addresses
@pytest.mark.parametrize("raw, country, expected", [
    ("2238 WILLIAMSBURG RD, GLEN DALE, WV", "US", "2238 williamsburg road glen dale wv"),
    ("2238 Williamsburg Road, Glen Dale, West Virginia", "US", "2238 williamsburg road glen dale wv"),
    ("0701 OLD GLORY RD, MARYVILLE, TN", "US", "701 old glory road maryville tn"),
    ("915 Dares Beach Rd, PMB 6098, Prince Frederick CDP, Maryland", "US",
     "915 dares beach road prince frederick cdp md"),
    ("FT WORTH, 2859 PARK MEADOW COURT, TX, null", "US", "fort worth 2859 park meadow court tx"),
    ("2238 WILLIAMSBURG STEET, GLEN DALE, WV", "US", "2238 williamsburg street glen dale wv"),
    ("12 St Louis Ave, St Louis, MO", "US", "12 saint louis avenue saint louis mo"),
    ("DOOR NO 35 , BRENTWOOD APARTMENTS, BANGALORE, KA", "India", "35 brentwood apartments bangalore ka"),
    ("Plot No.569/1400, Kuakhai Nagar, Khordha, Orissa", "India", "plot 569 1400 kuakhai nagar khordha od"),
    ("Flat No. A/#610, Bombay, महाराष्ट्र", "India", "flat a 610 mumbai mh"),
    ("15 R. DU PROGRES, ROUBAIX, Nord", "France", "15 rue du progres roubaix hdf"),
    ("DUNKERQUE, N°43 R D'ÉPINAL", "France", "dunkerque 43 rue d epinal"),
    ("19 BIS RUE DU MEOLE, Pessac, Nouvelle-Aquitaine", "France", "19 bis rue du meole pessac naq"),
    ("<NULL>", "India", ""),
    ("", "US", ""),
])
def test_normalize_address(raw, country, expected):
    assert normalize_address(raw, country) == expected


def test_component_order_does_not_change_fields():
    a = parse_address("TX, Cypress, 20126 Crossvine Trail Lane", "US")
    b = parse_address("20126 Crossvine Trail Ln, Cypress, Texas", "US")
    for f in ("addr_street", "addr_city", "addr_state", "addr_house_no"):
        assert getattr(a, f) == getattr(b, f)


def test_address_numbers_and_state():
    p = parse_address("00478/1, MSHC LAYOUT, HEBBAL, BANGALORE, ಕರ್ನಾಟಕ", "India")
    assert p.addr_house_no == "478/1" and p.addr_state == "ka" and p.addr_numbers == "1 478"
    p = parse_address("KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi", "India")
    assert p.addr_state == "dl" and "new delhi" in p.address_norm
    assert parse_address("Delhi, New Delhi, Promoter, R-16 Model Town-Iii", "India").addr_state == "dl"
    assert parse_address("12 MG Road, Bangalore, 560001, KA", "India").addr_postcode == "560001"
    assert parse_address("", "US").addr_is_missing


def test_unknown_country_is_open_set():
    assert normalize_country("France") == "france"
    assert normalize_country("Deutschland") == "deutschland"
    p = parse_address("Hauptstr 5, Berlin", "Germany")
    assert p.address_norm == "hauptstr 5 berlin" and not p.addr_is_missing
    assert canonical_state("tn", "us") == "tn" and canonical_state("tamil nadu", "india") == "tn"
    assert canonical_state("karnataka", "germany") == "india:ka"   # full names only
    assert canonical_state("ka", "germany") == ""


# ----------------------------------------------------------- text utils
def test_text_utils():
    assert fold_accents("Léarning Ésq Gúild") == "Learning Esq Guild"
    assert transliterate_indic("राम मार्केटिंग") == "ram marketing"
    assert transliterate_indic("കേരളം") == "keralam"
    assert phonetic_key("software") == phonetic_key(transliterate_indic("सॉफ्टवेयर"))
    assert detect_script("Sky હેલ્થકેર") == "mixed"
    assert detect_script("राम") == "indic:devanagari"


def test_preprocess_record_interface():
    r = preprocess_record("S2-1", "Star Flex  LLC", "2238 WILLIAMSBURG RD, GLEN DALE, WV", "US")
    assert list(r) == OUTPUT_COLUMNS
    assert r["source"] == "S2" and r["country_norm"] == "us"
    assert r["business_name_norm"] == "star flex llc"
    assert r["business_name_raw"] == "Star Flex  LLC"
