import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.mixture import GaussianMixture

# ----------------------------
# CONFIG (edit paths if needed)
# ----------------------------
IN_PATH_RG  = Path("agg_region_grade_subject_domain.csv")   # REQUIRED
IN_PATH_GND = Path("agg_gender_domain.csv")                  # OPTIONAL
OUT_PATH    = Path("region_grade_clusters_hybrid.csv")

RANDOM_STATE     = 42
N_CLUSTERS       = 3
IMBALANCE_RATIO  = 0.20       # if min_count/max_count < 0.20 OR <3 clusters -> collapse
LOW_Q, HIGH_Q    = 0.33, 0.67 # fallback cutoffs

# Fix common encoding/alias issues in region names
REGION_MAP = {
    "Ø¨Ø¹Ù„Ø¨Ùƒ ÙˆØ§Ù„Ù‡Ø±Ù…Ù„": "Baalbek-Hermel",
    "بعلبك والهرمل": "Baalbek-Hermel",
}

# ----------------------------
# HELPERS
# ----------------------------
def zscore(x: pd.Series) -> pd.Series:
    s = x.std(ddof=0)
    if s == 0 or np.isnan(s):
        return x - x.mean()
    return (x - x.mean()) / (s + 1e-9)

def build_region_features(dfr: pd.DataFrame) -> pd.DataFrame:
    req = {"region","subject","avg_score","pct_below"}
    miss = req - set(dfr.columns)
    if miss:
        raise ValueError(f"Missing columns in region file: {miss}")

    dfr = dfr.copy()
    dfr.columns = [c.strip().lower() for c in dfr.columns]
    if "grade" not in dfr.columns: dfr["grade"] = "All"
    if "n_students" not in dfr.columns: dfr["n_students"] = np.nan

    dfr["region_canonical"] = dfr["region"].replace(REGION_MAP)

    for c in ["avg_score","pct_below","n_students"]:
        if c in dfr.columns:
            dfr[c] = pd.to_numeric(dfr[c], errors="coerce")

    # per-(grade,subject) normalization
    dfr["z_avg"]  = dfr.groupby(["grade","subject"])["avg_score"].transform(zscore)
    dfr["pct_ok"] = 100 - dfr["pct_below"]          # higher is better
    dfr["z_ok"]   = dfr.groupby(["grade","subject"])["pct_ok"].transform(zscore)
    dfr["z_subject_strength"] = 0.5*dfr["z_avg"] + 0.5*dfr["z_ok"]

    # size signal
    if dfr["n_students"].notna().any():
        dfr["log_n_students"] = np.log1p(dfr["n_students"].clip(lower=0))
        dfr["z_size"] = dfr.groupby(["grade","subject"])["log_n_students"].transform(zscore)
    else:
        dfr["z_size"] = 0.0

    # aggregate to Region × Grade
    subj_strength = dfr.pivot_table(
        index=["region_canonical","grade"],
        columns="subject",
        values="z_subject_strength",
        aggfunc="mean"
    )
    subj_strength.columns = [f"strength_{s}" for s in subj_strength.columns]
    subj_strength = subj_strength.reset_index()

    agg = dfr.groupby(["region_canonical","grade"]).agg(
        mean_z_avg=("z_avg","mean"),
        mean_z_ok=("z_ok","mean"),
        mean_z_size=("z_size","mean"),
        composite_overall=("z_subject_strength","mean"),
        subjects_covered=("subject","nunique"),
        entries=("subject","size"),
    ).reset_index()

    feats = pd.merge(agg, subj_strength, on=["region_canonical","grade"], how="left")

    # fill missing subject strengths with 0 (neutral)
    for c in [c for c in feats.columns if c.startswith("strength_")]:
        feats[c] = feats[c].fillna(0.0)

    # percentiles (for output/explainability)
    comp = feats["composite_overall"].fillna(feats["composite_overall"].mean())
    feats["performance_percentile_global"] = (comp.rank(pct=True) * 100).round(1)
    feats["performance_percentile_within_grade"] = feats.groupby("grade")["composite_overall"].rank(pct=True) * 100

    return feats

def build_gender_features(dfg: pd.DataFrame | None) -> pd.DataFrame | None:
    # Optional enrichment for clustering (not included in final columns)
    if dfg is None: return None
    dfg = dfg.copy(); dfg.columns = [c.strip().lower() for c in dfg.columns]
    need = {"region","gender","avg_score"}
    if not need.issubset(dfg.columns): return None

    if "grade" not in dfg.columns: dfg["grade"] = "All"
    if "subject" not in dfg.columns: dfg["subject"] = "All"

    dfg["region_canonical"] = dfg["region"].replace(REGION_MAP)
    dfg["avg_score"] = pd.to_numeric(dfg["avg_score"], errors="coerce")
    if "pct_below" in dfg.columns:
        dfg["pct_below"] = pd.to_numeric(dfg["pct_below"], errors="coerce")
    else:
        dfg["pct_below"] = np.nan

    pv = dfg.pivot_table(index=["region_canonical","grade","subject"],
                         columns="gender",
                         values=["avg_score","pct_below"], aggfunc="mean")
    pv.columns = [f"{a}_{b}".lower() for a,b in pv.columns.to_flat_index()]
    pv = pv.reset_index()

    def safe_diff(a,b):
        return (pv[a] - pv[b]) if (a in pv.columns and b in pv.columns) else np.nan

    pv["gender_gap_avg"] = safe_diff("avg_score_female","avg_score_male")
    pv["gender_gap_ok"]  = ((100 - pv["pct_below_female"]) - (100 - pv["pct_below_male"])) \
                            if ("pct_below_female" in pv and "pct_below_male" in pv) else np.nan

    pv["z_gender_gap_avg"] = pv.groupby(["grade","subject"])["gender_gap_avg"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0)+1e-9))
    pv["z_gender_gap_ok"]  = pv.groupby(["grade","subject"])["gender_gap_ok"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0)+1e-9))

    gfeat = pv.groupby(["region_canonical","grade"]).agg(
        mean_gender_gap_avg=("z_gender_gap_avg","mean"),
        mean_gender_gap_ok=("z_gender_gap_ok","mean")
    ).reset_index()
    return gfeat

def label_with_gmm_or_percentiles(feats: pd.DataFrame) -> pd.DataFrame:
    # Build feature matrix (include gender gaps if present, but not required)
    sub_cols = [c for c in feats.columns if c.startswith("strength_")]
    opt_gender = [c for c in ["mean_gender_gap_avg","mean_gender_gap_ok"] if c in feats.columns]
    X_cols = ["mean_z_avg","mean_z_ok","mean_z_size","composite_overall",
              "subjects_covered","entries"] + sub_cols + opt_gender

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(feats[X_cols].fillna(0.0))

    # Try GMM
    gmm = GaussianMixture(n_components=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    labels = gmm.fit_predict(X_scaled)
    vc = pd.Series(labels).value_counts()
    imbalanced = (len(vc) < N_CLUSTERS) or ((vc.min() / vc.max()) < IMBALANCE_RATIO)

    if not imbalanced:
        feats["__cid"] = labels
        # Rank clusters by mean composite to name them
        means = feats.groupby("__cid")["composite_overall"].mean().sort_values(ascending=False)
        rank_map = {cid: lab for cid, lab in zip(means.index, ["High","Low","Medium"])}  # order will be remapped later anyway
        feats["cluster_label"] = feats["__cid"].map(rank_map)
        feats.drop(columns=["__cid"], inplace=True)
        return feats

    # Fallback: global percentile thresholds (balanced)
    q_low, q_high = feats["composite_overall"].quantile([LOW_Q, HIGH_Q])
    def to_label(v):
        if v <= q_low: return "Low"
        if v >= q_high: return "High"
        return "Medium"
    feats["cluster_label"] = feats["composite_overall"].apply(to_label)
    return feats

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    # 1) Build base features from region-grade-subject file
    dfr = pd.read_csv(IN_PATH_RG)
    feats = build_region_features(dfr)

    # 2) Optionally enrich with gender-gap features (only to help clustering; not in final columns)
    try:
        if IN_PATH_GND.exists():
            dfg = pd.read_csv(IN_PATH_GND)
            gfeat = build_gender_features(dfg)
            if gfeat is not None:
                feats = feats.merge(gfeat, on=["region_canonical","grade"], how="left")
    except Exception:
        pass  # if gender file problematic, proceed without it

    # 3) Assign labels via GMM or percentile fallback
    feats = label_with_gmm_or_percentiles(feats)

    # 4) Produce final CSV with EXACT columns & sorting you asked for
    # Custom cluster order per grade: High, Low, Medium
    order_map = {"High": 0, "Low": 1, "Medium": 2}
    feats["__order"] = feats["cluster_label"].map(order_map)

    final_cols = [
        "region_canonical","grade","cluster_label",
        "performance_percentile_global","performance_percentile_within_grade",
        "composite_overall","mean_z_avg","mean_z_ok","mean_z_size",
        "subjects_covered","entries","strength_Arabic","strength_English","strength_French"
    ]

    # If some subjects have different names/case in your data, normalize here:
    for needed in ["strength_Arabic","strength_English","strength_French"]:
        if needed not in feats.columns:
            # try to find a matching column ignoring case
            maybe = [c for c in feats.columns if c.lower() == needed.lower()]
            if maybe:
                feats.rename(columns={maybe[0]: needed}, inplace=True)
            else:
                feats[needed] = 0.0  # if totally missing, fill neutral

    out = (feats[final_cols + ["__order"]]
           .sort_values(["grade","__order","region_canonical"])
           .drop(columns="__order")
           .reset_index(drop=True))

    out.to_csv(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH.resolve()}")



