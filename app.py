import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Football ML Predictor", layout="wide")

############################################################
# COLUMN DEFINITIONS
############################################################

REQUIRED_COLUMNS = [
    "country", "season", "round",
    "home", "away",
    "home_goals", "away_goals",
    "home_win", "draw", "away_win",
    "over2.5", "under2.5",
    "gg", "ng",
    "suth", "suta",
    "corh", "cora",
    "yellowh", "yellowa",
]

OPTIONAL_COLUMNS = [
    "ballph", "ballpa",
    "foulsh", "foulsa",
    "sutht", "sutat",
]

COLUMN_DESCRIPTIONS = {
    "country": "League identifier (e.g., ENG1, GER1, SRB1).",
    "season": "Season number or year (e.g., 2023).",
    "round": "Match round index (integer).",
    "home": "Home team name.",
    "away": "Away team name.",
    "home_goals": "Full-time goals scored by home team.",
    "away_goals": "Full-time goals scored by away team.",
    "home_win": "Closing odds for home win (1).",
    "draw": "Closing odds for draw (X).",
    "away_win": "Closing odds for away win (2).",
    "over2.5": "Odds for over 2.5 total goals.",
    "under2.5": "Odds for under 2.5 total goals.",
    "gg": "Both teams to score YES (BTTS).",
    "ng": "Both teams to score NO.",
    "suth": "Home shots (total).",
    "suta": "Away shots (total).",
    "sutht": "Home shots on target.",
    "sutat": "Away shots on target.",
    "corh": "Home corners.",
    "cora": "Away corners.",
    "yellowh": "Home yellow cards.",
    "yellowa": "Away yellow cards.",
    "ballph": "Home ball possession %.",
    "ballpa": "Away ball possession %.",
    "foulsh": "Home fouls committed.",
    "foulsa": "Away fouls committed.",
}

TARGET_COLS = {
    "shots_home": "suth",
    "shots_away": "suta",
    "corners_home": "corh",
    "corners_away": "cora",
    "cards_home": "yellowh",
    "cards_away": "yellowa",
}

############################################################
# HEADER + COLUMN HELP
############################################################

st.title("⚽ Football ML Predictor – Shots / Corners / Cards")

with st.expander("📘 Column requirements and template", expanded=False):
    st.markdown("### Required columns")
    for col in REQUIRED_COLUMNS:
        st.markdown(f"- **{col}** — {COLUMN_DESCRIPTIONS.get(col, '')}")

    st.markdown("### Optional columns (improve model accuracy)")
    for col in OPTIONAL_COLUMNS:
        st.markdown(f"- **{col}** — {COLUMN_DESCRIPTIONS.get(col, '')}")

    st.markdown("---")
    st.markdown("### Example row (values are illustrative)")
    example = {
        "country": "ENG1",
        "season": 2023,
        "round": 12,
        "home": "Arsenal",
        "away": "Chelsea",
        "home_goals": 3,
        "away_goals": 1,
        "home_win": 1.85,
        "draw": 3.60,
        "away_win": 4.20,
        "over2.5": 1.70,
        "under2.5": 2.20,
        "gg": 1.72,
        "ng": 2.05,
        "suth": 14,
        "suta": 7,
        "corh": 6,
        "cora": 3,
        "yellowh": 2,
        "yellowa": 3,
        "ballph": 58,
        "ballpa": 42,
        "foulsh": 9,
        "foulsa": 14,
        "sutht": 6,
        "sutat": 3,
    }
    st.dataframe(pd.DataFrame([example]))


############################################################
# CSV UPLOAD
############################################################

uploaded = st.file_uploader("📤 Upload your match dataset (CSV)", type=["csv"])

if uploaded is None:
    st.info("Upload your CSV to continue.")
    st.stop()


############################################################
# LOAD CSV (AUTO-DETECT FORMAT)
############################################################

@st.cache_data
def load_csv(file) -> pd.DataFrame | None:
    # Try standard CSV (comma separated, header row)
    try:
        df_try = pd.read_csv(file)
        if df_try is not None and df_try.shape[1] > 5:
            return df_try
    except Exception:
        pass

    # Fallback: single-column file with semicolon-separated lines
    try:
        file.seek(0)
        raw = pd.read_csv(file, header=None)
        if raw.shape[1] == 1:
            df = raw[0].str.split(";", expand=True)
            header = df.iloc[0].tolist()
            df = df.drop(index=0).reset_index(drop=True)
            df.columns = header
            return df
    except Exception:
        pass

    return None


df = load_csv(uploaded)

if df is None:
    st.error(
        "Could not parse the CSV.\n\n"
        "- Either upload a normal multi-column CSV with a header row, **or**\n"
        "- A single-column CSV where each row is semicolon-separated."
    )
    st.stop()

# Safe numeric conversion
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    except Exception:
        st.warning(f"Column '{col}' could not be converted to numeric. Kept as text.")


############################################################
# VALIDATE REQUIRED COLUMNS
############################################################

missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

if missing:
    st.error("Your CSV is missing required columns:")
    for m in missing:
        st.markdown(f"- **{m}** — {COLUMN_DESCRIPTIONS.get(m, '')}")
    st.stop()


############################################################
# ODDS NORMALIZATION + EXPECTED METRICS
############################################################

def normalize_odds(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    odds_cols = ["home_win", "draw", "away_win", "over2.5", "under2.5", "gg", "ng"]
    for c in odds_cols:
        if c in df.columns:
            df[f"p_{c}"] = 1 / df[c]

    groups = {
        "1x2": ["p_home_win", "p_draw", "p_away_win"],
        "total": ["p_over2.5", "p_under2.5"],
        "btts": ["p_gg", "p_ng"],
    }
    for _, cols in groups.items():
        valid = [c for c in cols if c in df.columns]
        if len(valid) >= 2:
            s = df[valid].sum(axis=1)
            for v in valid:
                df[v.replace("p_", "pn_")] = df[v] / s
    return df


def add_expected_metrics(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    # xG total and split by 1x2
    df["xG_total"] = 2.5 + (df["pn_over2.5"] - 0.5) * 2.8

    df["att_home"] = df["pn_home_win"] + 0.5 * df["pn_draw"]
    df["att_away"] = df["pn_away_win"] + 0.5 * df["pn_draw"]

    df["xG_home"] = df["xG_total"] * (df["att_home"] / (df["att_home"] + df["att_away"]))
    df["xG_away"] = df["xG_total"] - df["xG_home"]

    # Dataset-calibrated factors
    beta_home = 8.02
    beta_away = 8.73
    gamma = 0.364
    delta_home = 0.404
    delta_away = 0.406

    df["xShots_home"] = df["xG_home"] * beta_home
    df["xShots_away"] = df["xG_away"] * beta_away

    df["xSOT_home"] = df["xShots_home"] * gamma
    df["xSOT_away"] = df["xShots_away"] * gamma

    df["xCorners_home"] = df["xShots_home"] * delta_home
    df["xCorners_away"] = df["xShots_away"] * delta_away

    # Possession defaults if missing
    df["pos_home"] = df.get("ballph", pd.Series(50, index=df.index)) / 100
    df["pos_away"] = df.get("ballpa", pd.Series(50, index=df.index)) / 100

    # Foul rate uses available columns or defaults to 20 fouls/match
    total_fouls = df.get("foulsh", 0).sum() + df.get("foulsa", 0).sum()
    foul_rate = total_fouls / max(len(df), 1) if total_fouls > 0 else 20.0

    df["xFouls_home"] = (1 - df["pos_home"]) * foul_rate * 0.5
    df["xFouls_away"] = (1 - df["pos_away"]) * foul_rate * 0.5

    theta_home = 0.176
    theta_away = 0.193

    df["tempo"] = 0.5 + df["pn_over2.5"]
    df["xCards_home"] = df["xFouls_home"] * theta_home * df["tempo"]
    df["xCards_away"] = df["xFouls_away"] * theta_away * df["tempo"]

    return df


df = normalize_odds(df)
df = add_expected_metrics(df)


############################################################
# BUILD ML DATASET
############################################################

# Only numeric features, exclude target columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in TARGET_COLS.values()]

# Drop rows where targets or features are missing
ml_df = df.dropna(subset=list(TARGET_COLS.values()) + feature_cols)

if ml_df.empty:
    st.error("After cleaning, there are no rows with complete targets and features. "
             "Check that your dataset has shots, corners, cards, and odds filled.")
    st.stop()


############################################################
# TRAIN MODELS (CACHED)
############################################################

@st.cache_resource
def train_models(ml: pd.DataFrame, feature_names: list[str]) -> dict:
    models = {}
    X = ml[feature_names]

    for name, tgt in TARGET_COLS.items():
        y = ml[tgt]
        model = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
        )
        model.fit(X, y)
        models[name] = model
    return models


models = train_models(ml_df, feature_cols)


############################################################
# TABS
############################################################

tab_predict, tab_leagues, tab_shap = st.tabs(
    ["Predict future match", "League predictability", "SHAP explanations"]
)

############################################################
# TAB 1 – PREDICT FUTURE MATCH
############################################################

with tab_predict:
    st.subheader("🔮 Predict match stats from odds")

    leagues = sorted(df["country"].unique())
    league = st.selectbox("League", leagues)

    teams = sorted(df[df["country"] == league]["home"].unique())
    c1, c2 = st.columns(2)
    home_team = c1.selectbox("Home team", teams)
    away_team = c2.selectbox("Away team", teams)

    c3, c4, c5 = st.columns(3)
    home_odds = c3.number_input("Home win (1)", value=2.00)
    draw_odds = c4.number_input("Draw (X)", value=3.30)
    away_odds = c5.number_input("Away win (2)", value=3.40)

    c6, c7, c8 = st.columns(3)
    over25 = c6.number_input("Over 2.5", value=2.00)
    under25 = c7.number_input("Under 2.5", value=1.85)
    gg = c8.number_input("BTTS Yes", value=1.80)
    ng = st.number_input("BTTS No", value=1.90)

    if st.button("Predict stats", type="primary"):
        # Build single-row feature frame
        row = pd.DataFrame([{
            **{c: 0 for c in feature_cols},  # initialize numeric features
            "home_win": home_odds,
            "draw": draw_odds,
            "away_win": away_odds,
            "over2.5": over25,
            "under2.5": under25,
            "gg": gg,
            "ng": ng,
            "country": None,  # not used as numeric feature
            "season": df["season"].max(),
            "round": df["round"].max() + 1 if "round" in df.columns else 1,
            "home_goals": 0,
            "away_goals": 0,
            "ballph": 50,
            "ballpa": 50,
            "foulsh": 10,
            "foulsa": 10,
        }])

        # Normalize + expected metrics
        row = normalize_odds(row)
        row = add_expected_metrics(row)

        # Ensure all numeric feature columns exist
        for c in feature_cols:
            if c not in row.columns:
                row[c] = 0

        X_row = row[feature_cols]

        preds = {name: float(models[name].predict(X_row)[0]) for name in models}

        st.markdown("### 📊 Predicted statistics")
        s1, s2 = st.columns(2)
        s1.metric("Home shots", f"{preds['shots_home']:.1f}")
        s2.metric("Away shots", f"{preds['shots_away']:.1f}")

        c1, c2 = st.columns(2)
        c1.metric("Home corners", f"{preds['corners_home']:.1f}")
        c2.metric("Away corners", f"{preds['corners_away']:.1f}")

        k1, k2 = st.columns(2)
        k1.metric("Home cards", f"{preds['cards_home']:.2f}")
        k2.metric("Away cards", f"{preds['cards_away']:.2f}")


############################################################
# TAB 2 – LEAGUE PREDICTABILITY
############################################################

with tab_leagues:
    st.subheader("📈 League predictability index")

    rows = []
    for lg, subset in ml_df.groupby("country"):
        if len(subset) < 150:
            continue
        X_lg = subset[feature_cols]
        for name, model in models.items():
            y_true = subset[TARGET_COLS[name]]
            y_pred = model.predict(X_lg)
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            std = y_true.std()
            if std > 0:
                pred_idx = 1 - rmse / std
                rows.append({
                    "league": lg,
                    "target": name,
                    "rmse": rmse,
                    "std": std,
                    "predictability": pred_idx,
                    "n_matches": len(subset),
                })

    if rows:
        pred_df = pd.DataFrame(rows)
        st.dataframe(
            pred_df.sort_values(["target", "predictability"], ascending=[True, False]),
            use_container_width=True,
            height=500,
        )
    else:
        st.info("Not enough data per league to compute predictability (need ≥150 matches).")


############################################################
# TAB 3 – SHAP EXPLANATIONS (ON DEMAND)
############################################################

with tab_shap:
    st.subheader("🧠 SHAP feature importance (on demand)")

    shap_target = st.selectbox("Select model", list(models.keys()))
    run_shap = st.button("Compute SHAP explanations for this model")

    if run_shap:
        st.write("Sampling data and computing SHAP values (first run may take a bit)...")
        X_sample = ml_df[feature_cols].sample(
            min(800, len(ml_df)), random_state=42
        )

        explainer = shap.TreeExplainer(models[shap_target])
        shap_values = explainer.shap_values(X_sample)

        shap.summary_plot(shap_values, X_sample, show=False)
        st.pyplot(plt.gcf())

st.success("App initialized successfully.")
