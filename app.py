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
    "season": "Season number (e.g., 2023).",
    "round": "Match round (integer).",
    "home": "Home team.",
    "away": "Away team.",
    "home_goals": "Full-time home goals.",
    "away_goals": "Full-time away goals.",
    "home_win": "Odds for home win.",
    "draw": "Odds for draw.",
    "away_win": "Odds for away win.",
    "over2.5": "Over 2.5 goals odds.",
    "under2.5": "Under 2.5 goals odds.",
    "gg": "BTTS Yes.",
    "ng": "BTTS No.",
    "suth": "Home shots.",
    "suta": "Away shots.",
    "sutht": "Home SOT.",
    "sutat": "Away SOT.",
    "corh": "Home corners.",
    "cora": "Away corners.",
    "yellowh": "Home yellow cards.",
    "yellowa": "Away yellow cards.",
    "ballph": "Home possession (%).",
    "ballpa": "Away possession (%).",
    "foulsh": "Home fouls.",
    "foulsa": "Away fouls.",
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
# HEADER + HELP
############################################################

st.title("⚽ Football ML Predictor – Shots / Corners / Cards")

with st.expander("📘 Column requirements"):
    st.markdown("### Required columns:")
    for col in REQUIRED_COLUMNS:
        st.markdown(f"- **{col}** — {COLUMN_DESCRIPTIONS.get(col, '')}")

    st.markdown("### Optional columns:")
    for col in OPTIONAL_COLUMNS:
        st.markdown(f"- **{col}** — {COLUMN_DESCRIPTIONS.get(col, '')}")

############################################################
# CSV UPLOAD
############################################################

uploaded = st.file_uploader("📤 Upload your dataset (CSV)", type=["csv"])

if not uploaded:
    st.info("Upload CSV to begin.")
    st.stop()

############################################################
# LOAD CSV WITH AUTO-DETECT
############################################################

@st.cache_data
def load_csv(file):
    # Try normal CSV
    try:
        df = pd.read_csv(file)
        if df.shape[1] > 5:
            return df
    except:
        pass

    # Try single-column semicolon dataset
    try:
        file.seek(0)
        raw = pd.read_csv(file, header=None)
        if raw.shape[1] == 1:
            df = raw[0].str.split(";", expand=True)
            header = df.iloc[0].tolist()
            df = df.drop(index=0).reset_index(drop=True)
            df.columns = header
            return df
    except:
        pass

    return None

df = load_csv(uploaded)

if df is None:
    st.error(
        "❌ Could not parse CSV.\n"
        "Upload a standard CSV or a semicolon-separated single-column file."
    )
    st.stop()

# Safe numeric conversion
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    except:
        pass

############################################################
# VALIDATE COLUMNS
############################################################

missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

if missing:
    st.error("❌ Missing required columns:")
    for m in missing:
        st.write(f"- {m}")
    st.stop()

############################################################
# ODDS NORMALIZATION + EXPECTED METRICS
############################################################

def normalize_odds(df_in):
    df = df_in.copy()
    oc = ["home_win", "draw", "away_win", "over2.5", "under2.5", "gg", "ng"]
    for c in oc:
        df[f"p_{c}"] = 1 / df[c]

    groups = {
        "1x2": ["p_home_win", "p_draw", "p_away_win"],
        "total": ["p_over2.5", "p_under2.5"],
        "btts": ["p_gg", "p_ng"],
    }

    for _, cols in groups.items():
        total = df[cols].sum(axis=1)
        for c in cols:
            df[c.replace("p_", "pn_")] = df[c] / total

    return df

def add_expected(df_in):
    df = df_in.copy()

    df["xG_total"] = 2.5 + (df["pn_over2.5"] - 0.5) * 2.8

    df["att_home"] = df["pn_home_win"] + 0.5 * df["pn_draw"]
    df["att_away"] = df["pn_away_win"] + 0.5 * df["pn_draw"]

    df["xG_home"] = df["xG_total"] * df["att_home"] / (df["att_home"] + df["att_away"])
    df["xG_away"] = df["xG_total"] - df["xG_home"]

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

    df["pos_home"] = df.get("ballph", pd.Series(50, index=df.index)) / 100
    df["pos_away"] = df.get("ballpa", pd.Series(50, index=df.index)) / 100

    total_fouls = df.get("foulsh", 0).sum() + df.get("foulsa", 0).sum()
    foul_rate = total_fouls / len(df) if total_fouls > 0 else 20

    df["xFouls_home"] = (1 - df["pos_home"]) * foul_rate * 0.5
    df["xFouls_away"] = (1 - df["pos_away"]) * foul_rate * 0.5

    theta_home = 0.176
    theta_away = 0.193

    df["tempo"] = 0.5 + df["pn_over2.5"]
    df["xCards_home"] = df["xFouls_home"] * theta_home * df["tempo"]
    df["xCards_away"] = df["xFouls_away"] * theta_away * df["tempo"]

    return df

df = normalize_odds(df)
df = add_expected(df)

############################################################
# ML DATA
############################################################

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in TARGET_COLS.values()]

ml_df = df.dropna(subset=list(TARGET_COLS.values()) + feature_cols)

if ml_df.empty:
    st.error("No complete rows for training (missing shots/corners/cards/odds).")
    st.stop()

############################################################
# TRAIN MODELS
############################################################

@st.cache_resource
def train_models(df_ml, feature_cols):
    models = {}
    X = df_ml[feature_cols]

    for name, target in TARGET_COLS.items():
        y = df_ml[target]

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
# UI (TABS)
############################################################

tab_pred, tab_pred_idx, tab_shap = st.tabs(
    ["Predict match", "League predictability", "SHAP"]
)

############################################################
# PREDICT MATCH
############################################################

with tab_pred:
    st.subheader("🔮 Predict match stats")

    leagues = sorted(df["country"].unique())
    league = st.selectbox("Select league", leagues)

    teams = sorted(df[df["country"] == league]["home"].unique())
    c1, c2 = st.columns(2)
    home_t = c1.selectbox("Home team", teams)
    away_t = c2.selectbox("Away team", teams)

    c3, c4, c5 = st.columns(3)
    home_odds = c3.number_input("Home win", value=2.0)
    draw_odds = c4.number_input("Draw", value=3.3)
    away_odds = c5.number_input("Away win", value=3.4)

    c6, c7, c8 = st.columns(3)
    over25 = c6.number_input("Over 2.5", value=2.0)
    under25 = c7.number_input("Under 2.5", value=1.85)
    gg = c8.number_input("BTTS Yes", value=1.8)
    ng = st.number_input("BTTS No", value=1.9)

    if st.button("Predict"):
        row = pd.DataFrame([{
            **{c: 0 for c in feature_cols},
            "home_win": home_odds,
            "draw": draw_odds,
            "away_win": away_odds,
            "over2.5": over25,
            "under2.5": under25,
            "gg": gg,
            "ng": ng,
        }])

        row = normalize_odds(row)
        row = add_expected(row)

        for c in feature_cols:
            if c not in row.columns:
                row[c] = 0

        X_row = row[feature_cols]

        preds = {name: float(models[name].predict(X_row)[0]) for name in models}

        st.metric("Home shots", f"{preds['shots_home']:.1f}")
        st.metric("Away shots", f"{preds['shots_away']:.1f}")
        st.metric("Home corners", f"{preds['corners_home']:.1f}")
        st.metric("Away corners", f"{preds['corners_away']:.1f}")
        st.metric("Home cards", f"{preds['cards_home']:.2f}")
        st.metric("Away cards", f"{preds['cards_away']:.2f}")

############################################################
# PREDICTABILITY INDEX (RMSE FIX)
############################################################

with tab_pred_idx:
    st.subheader("📈 League predictability index (RMSE-based)")

    rows = []

    for lg, subset in ml_df.groupby("country"):
        if len(subset) < 150:
            continue

        X_lg = subset[feature_cols]

        for name, model in models.items():
            y_true = subset[TARGET_COLS[name]]
            y_pred = model.predict(X_lg)

            mse = mean_squared_error(y_true, y_pred)
            rmse = float(np.sqrt(mse))
            std = float(y_true.std())

            pred_idx = 1 - rmse / std if std > 0 else 0

            rows.append({
                "league": lg,
                "target": name,
                "rmse": rmse,
                "std": std,
                "predictability": pred_idx,
                "n_matches": len(subset),
            })

    if rows:
        table = pd.DataFrame(rows)
        st.dataframe(
            table.sort_values(["target", "predictability"], ascending=[True, False]),
            use_container_width=True
        )
    else:
        st.info("Not enough data (need 150+ matches per league).")

############################################################
# SHAP
############################################################

with tab_shap:
    st.subheader("🧠 SHAP Feature Importance")

    shap_target = st.selectbox("Select model:", list(models.keys()))

    if st.button("Run SHAP"):
        st.write("Computing SHAP values...")

        # Sample safely
        X_sample = ml_df[feature_cols].sample(
            min(600, len(ml_df)), random_state=42
        )

        explainer = shap.TreeExplainer(models[shap_target])
        shap_values = explainer.shap_values(X_sample)

        shap.summary_plot(shap_values, X_sample, show=False)
        st.pyplot(plt.gcf())

st.success("App loaded successfully.")
