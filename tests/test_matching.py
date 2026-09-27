"""Tests for the metric, the decision rule and the pairwise features."""
import pandas as pd
import pytest

from src.evaluation.metrics import per_entity_scores
from src.matching.feature_engineering import STRING_FEATURES, _pair_features
from src.matching.threshold_tuning import apply_rule, best_prefix, decide


def test_metric_matches_readme_example():
    truth = pd.DataFrame({"s1_id": ["S1-1", "S1-1"], "cand_id": ["S2-47", "S3-812"]})
    pred = pd.DataFrame({"s1_id": ["S1-1"] * 3, "cand_id": ["S2-47", "S2-193", "S3-812"]})
    s = per_entity_scores(pred, truth, ["S1-1"])
    assert s.loc["S1-1", "f05"] == pytest.approx(0.714, abs=1e-3)


def test_metric_singletons():
    truth = pd.DataFrame({"s1_id": [], "cand_id": []})
    pred = pd.DataFrame({"s1_id": ["S1-2"], "cand_id": ["S2-9"]})
    s = per_entity_scores(pred, truth, ["S1-1", "S1-2"])
    assert s.loc["S1-1", "f05"] == 1.0      # empty truth, empty prediction
    assert s.loc["S1-2", "f05"] == 0.0      # false merge on a singleton


def test_decide_one_to_one_and_threshold():
    scored = pd.DataFrame({"s1_id": ["a", "b", "a", "c"], "cand_id": ["x", "x", "y", "z"],
                           "p": [0.9, 0.8, 0.5, 0.95]})
    kept = decide(scored, t=0.6)
    assert set(zip(kept["s1_id"], kept["cand_id"])) == {("a", "x"), ("c", "z")}
    kept = decide(scored, t=0.6, one_to_one=False)
    assert ("b", "x") in set(zip(kept["s1_id"], kept["cand_id"]))


def _rec(name, core, addr, state, house, nums, missing=False, alt=""):
    return ("id", name, core, "", alt, "", False, 1.0, "latin", addr, addr, "", state, house, nums, "",
            missing, "us")


def test_pair_features_shape_and_missing_address():
    a = _rec("star flex llc", "star flex", "2238 williamsburg road glen dale wv", "wv", "2238", "2238")
    b = _rec("star flex llc", "star flex", "2238 williamsburg road glen dale wv", "wv", "2238", "2238")
    f = dict(zip(STRING_FEATURES, _pair_features(a, b)))
    assert len(f) == len(STRING_FEATURES)
    assert f["name_core_exact"] == 1 and f["state_eq"] == 1 and f["house_eq"] == 1
    c = _rec("star flex llc", "star flex", "", "", "", "", missing=True)
    f = dict(zip(STRING_FEATURES, _pair_features(a, c)))
    assert f["addr_missing_b"] == 1 and f["state_eq"] == -1    # unknown, not a mismatch


def test_expected_f_rule():
    import numpy as np
    assert best_prefix(np.array([0.99, 0.98, 0.01])) == 2      # two sure matches
    assert best_prefix(np.array([0.2, 0.1])) == 0              # likely a singleton: predict empty
    scored = pd.DataFrame({"s1_id": ["a", "a", "b"], "cand_id": ["x", "y", "z"], "p": [0.99, 0.02, 0.3]})
    kept = apply_rule(scored, {"rule": "expected_f", "cal": 1.0})
    assert set(zip(kept["s1_id"], kept["cand_id"])) == {("a", "x")}


def test_house_relations():
    from src.matching.feature_engineering import _REL, _house_rel
    assert _house_rel("2915", "915")[0] == _REL["b_suffix_of_a"]      # leading digit dropped
    assert _house_rel("19753", "1975")[0] == _REL["b_prefix_of_a"]    # trailing digit dropped
    assert _house_rel("3", "12") == (_REL["plus_small"], 9.0)          # decoy sibling a few doors up
    assert _house_rel("2412", "2411")[0] == _REL["last_digit"]
    assert _house_rel("", "12")[0] == _REL["missing"]


def test_french_acronyms():
    from src.matching.feature_engineering import _acronym
    from src.preprocessing.name_normalizer import parse_name

    def acr(a, b):
        pa, pb = parse_name(a), parse_name(b)
        return _acronym(pa.name_core.split(), pa.name_norm, pb.name_core)
    assert acr("Blue Crest Partners LLC", "BCP")
    assert acr("Ets Lavant SARL", "EL")                # "Ets" kept, legal form skipped
    assert acr("Comité du Very", "CV")                 # article skipped
    assert not acr("Blue Crest Partners LLC", "BC")


def test_region_fill_city_key():
    from src.preprocessing.preprocessing import _city_key
    assert _city_key("st herblain") == _city_key("saint herblain")
    assert _city_key("ste foy") == "sainte foy"
