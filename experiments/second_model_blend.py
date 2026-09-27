"""Second LightGBM on the saved v6 features; blend with v6 on the thr group."""
import json, time, numpy as np, pandas as pd, lightgbm as lgb
from src.matching.train_model import _NpySeq, load_group, LGB_PARAMS, FEATURE_DIR, MODEL_DIR, CATEGORICAL
from src.matching.feature_engineering import FEATURES
from src.blocking.candidate_generator import countries
from src.matching.threshold_tuning import assign_one_to_one
from src.preprocessing.io import ground_truth_pairs, read_ground_truth
t0 = time.time(); log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
P2 = dict(LGB_PARAMS, num_leaves=511, min_child_samples=50, feature_fraction=0.5, bagging_fraction=0.7, seed=11)
fit = [_NpySeq(FEATURE_DIR / f"X_fit_{c}.npy") for c in countries("train")]
y = np.concatenate([pd.read_parquet(FEATURE_DIR / f"meta_fit_{c}.parquet")["label"].to_numpy() for c in countries("train")])
Xes, mes = load_group("es")
dfit = lgb.Dataset(fit, y, feature_name=FEATURES, categorical_feature=CATEGORICAL, free_raw_data=True)
des = lgb.Dataset(Xes, mes["label"], reference=dfit)
b2 = lgb.train(P2, dfit, num_boost_round=5000, valid_sets=[des], callbacks=[lgb.early_stopping(150), lgb.log_evaluation(250)])
b2.save_model(str(MODEL_DIR / "lgbm_matcher_m2.txt"))
log(f"m2 trained: {b2.best_iteration} trees")
del dfit, des, Xes
Xthr, mthr = load_group("thr")
p2 = b2.predict(Xthr, num_threads=7)
sc = pd.read_parquet(FEATURE_DIR / "thr_scored.parquet")
assert (sc.s1_id.to_numpy() == mthr.s1_id.to_numpy()).all() and (sc.cand_id.to_numpy() == mthr.cand_id.to_numpy()).all()
sc["p2"] = p2; sc.to_parquet(FEATURE_DIR / "thr_scored_m2.parquet")
g = pd.read_parquet(FEATURE_DIR / "train_entity_groups.parquet"); E = g[g.group == "thr"].reset_index(drop=True)
code = pd.Series(np.arange(len(E)), index=E.s1_id)
truth = ground_truth_pairs(read_ground_truth()).rename(columns={"source1_entity_id": "s1_id", "entity_id": "cand_id"})
truth = truth[truth.s1_id.isin(code.index)]
n_true = np.bincount(code[truth.s1_id].to_numpy(), minlength=len(E))
rng = np.random.default_rng(1); half = rng.random(len(E)) < 0.5
def f05(d):
    ec = code[d.s1_id].to_numpy(); lab = d.label.to_numpy().astype(bool)
    npred = np.bincount(ec, minlength=len(E)); tp = np.bincount(ec[lab], minlength=len(E))
    pr = np.where(npred > 0, tp / np.maximum(npred, 1), 0); rc = np.where(n_true > 0, tp / np.maximum(n_true, 1), 0)
    den = .25 * pr + rc; f = np.where(den > 0, 1.25 * pr * rc / np.where(den > 0, den, 1), 0)
    return np.where((n_true == 0) & (npred == 0), 1, f)
for w in (0.0, 0.3, 0.5, 0.7, 1.0):
    d = assign_one_to_one(sc.assign(p=(1 - w) * sc.p + w * sc.p2))
    best = max(((t, f05(d[d.p >= t])) for t in np.round(np.arange(0.5, 0.9, 0.025), 3)), key=lambda x: x[1].mean())
    print(f"w_m2={w}: t={best[0]} macro={best[1].mean():.5f} half1={best[1][half].mean():.5f} half2={best[1][~half].mean():.5f}", flush=True)
log(f"done ({time.time() - t0:.0f}s)")
