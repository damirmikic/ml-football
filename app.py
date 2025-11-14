###############################################
# Optimized app.py — Deployable on Streamlit Cloud
###############################################

import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import shap
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Football Match Predictor", layout="wide")


############################################################
# 0) DEFINITIONS
############################################################

TARGET_COLS = {
    "shots_home": "suth",
    "shots_away": "suta",
    "corners_home": "corh",
    "corners_away": "cora",
    "cards_home": "yellowh",
    "cards_away": "yellowa",
}


############################################################
# 1) LOAD CSV  (FAST + CACHED)
############################################################

@st.cache_data(show_spinner=True)
def load_data():
    raw = pd.read_csv("mldatafootball.csv", header=None)
    df = raw[0].str.split(";", expand=True)
    header = df.iloc[0].tolist()
    df = df.drop(index=0).reset_index(drop=True)
    df.columns = header

    df = df.replace({"": np.nan, "None": np.nan})
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df

df = load_data()


############################################################
# 2) BUILD FEATURE ENGINEERING (CACHED)
############################################################

@st.cache_data(show_spinner=True)
def preprocess(df):
    df = df.copy()

    # Odds normalization
    odds_cols = ["home_win","draw","away_win","over2.5","under2.5","gg","ng"]
    for col in odds_cols:
        if col in df.columns:
            df[f"p_{col}"] = 1 / df[col]

    groups = {
        "1x2": ["p_home_win","p_draw","p_away_win"],
        "total": ["p_over2.5","p_under2.5"],
        "btts": ["p_gg","p_ng"],
    }

    for name, cols in groups.items():
        valid = [c for c in cols if c in df.columns]
        if len(valid) >= 2:
            s = df[valid].sum(axis=1)
            for c in valid:
                df[c.replace("p_","pn_")] = df[c] / s

    # Expected metrics
    df["xG_total"] = 2.5 + (df["pn_over2.5"] - 0.5) * 2.8
    df["att_share_home"] = df["pn_home_win"] + 0.5 * df["pn_draw"]
    df["att_share_away"] = df["pn_away_win"] + 0.5 * df["pn_draw"]

    df["xG_home"] = df["xG_total"] * (
        df["att_share_home"] / (df["att_share_home"] + df["att_share_away"])
    )
    df["xG_away"] = df["xG_total"] - df["xG_home"]

    # Dataset-derived calibrated factors
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

    df["pos_home"] = df["ballph"] / 100
    df["pos_away"] = df["ballpa"] / 100

    foul_rate = (df["foulsh"].sum() + df["foulsa"].sum()) / len(df)

    df["xFouls_home"] = (1 - df["pos_home"]) * foul_rate * 0.5
    df["xFouls_away"] = (1 - df["pos_away"]) * foul_rate * 0.5

    df["tempo_factor"] = 0.5 + df["pn_over2.5"]

    theta_home = 0.176
    theta_away = 0.193
    df["xCards_home"] = df["xFouls_home"] * theta_home * df["tempo_factor"]
    df["xCards_away"] = df["xFouls_away"] * theta_away * df["tempo_factor"]

    return df

df = preprocess(df)


############################################################
# 3) BUILD ML DATASET (CACHED)
############################################################

@st.cache_data(show_spinner=True)
def build_ml(df):
    ml = df.dropna(subset=TARGET_COLS.values())
    feature_cols = [c for c in df.columns if c not in TARGET_COLS.values()]
    return ml, feature_cols

ml_df, feature_cols = build_ml(df)


############################################################
# 4) TRAIN MODELS (CACHED)
############################################################

@st.cache_resource(show_spinner=True)
def train_models(ml_df, feature_cols):
    models = {}
    for name, target in TARGET_COLS.items():
        X = ml_df[feature_cols]
        y = ml_df[target]

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
# 5) PREDICTION FUNCTION
############################################################

def predict_match(row):
    preds = {}
    for name, model in models.items():
        preds[name] = float(model.predict(row[feature_cols])[0])
    return preds


############################################################
# STREAMLIT UI
############################################################

st.title("⚽ Football Match Predictor (Fast + Cloud-Optimized)")

tab1, tab2, tab3 = st.tabs(["Predict Match", "League Predictability", "Explainability"])


############################################################
# TAB 1 – PREDICT MATCH
############################################################

with tab1:
    st.subheader("Select match and enter odds")

    leagues = sorted(df["country"].unique())
    league = st.selectbox("League", leagues)

    home_teams = sorted(df[df["country"] == league]["home"].unique())
    away_teams = home_teams

    col1, col2 = st.columns(2)
    home = col1.selectbox("Home Team", home_teams)
    away = col2.selectbox("Away Team", away_teams)

    colA, colB, colC = st.columns(3)
    home_odds = colA.number_input("Home Win", value=2.00)
    draw_odds = colB.number_input("Draw", value=3.20)
    away_odds = colC.number_input("Away Win", value=3.50)

    colD, colE, colF = st.columns(3)
    over25 = colD.number_input("Over 2.5", value=2.00)
    under25 = colE.number_input("Under 2.5", value=1.85)
    gg = colF.number_input("BTTS Yes", value=1.80)
    ng = st.number_input("BTTS No", value=1.90)

    if st.button("Predict"):
        # Build input row
        row = pd.DataFrame([{
            **{c:0 for c in feature_cols},
            "home_win": home_odds,
            "draw": draw_odds,
            "away_win": away_odds,
            "over2.5": over25,
            "under2.5": under25,
            "gg": gg,
            "ng": ng,
            "country": league,
            "season": df["season"].max(),
            "round": 1,
            "home": home,
            "away": away,
        }])

        # normalize + expected metrics
        row = preprocess(normalize_odds(row))

        preds = predict_match(row)

        st.subheader("Predicted Stats")
        c1, c2 = st.columns(2)
        c1.metric("Home Shots", f"{preds['shots_home']:.1f}")
        c2.metric("Away Shots", f"{preds['shots_away']:.1f}")

        c3, c4 = st.columns(2)
        c3.metric("Home Corners", f"{preds['corners_home']:.1f}")
        c4.metric("Away Corners", f"{preds['corners_away']:.1f}")

        c5, c6 = st.columns(2)
        c5.metric("Home Cards", f"{preds['cards_home']:.2f}")
        c6.metric("Away Cards", f"{preds['cards_away']:.2f}")


############################################################
# TAB 2 – LEAGUE PREDICTABILITY
############################################################

with tab2:
    st.subheader("Predictability Index by League")

    rows = []
    for lg, subset in ml_df.groupby("country"):
        if len(subset) < 200:
            continue
        for name, model in models.items():
            y = subset[TARGET_COLS[name]]
            pred = model.predict(subset[feature_cols])
            rmse = mean_squared_error(y, pred, squared=False)
            std = y.std()
            rows.append({"league": lg, "target": name, "predictability": 1 - rmse/std})

    pred_index = pd.DataFrame(rows)
    st.dataframe(pred_index.sort_values(["target", "predictability"], ascending=False))


############################################################
# TAB 3 – SHAP (LAZY LOADED)
############################################################

with tab3:
    st.subheader("Model Explainability (SHAP)")

    shap_target = st.selectbox("Select model", list(models.keys()))
    st.write("Computing SHAP… (first time may take ~10s)")

    X_sample = ml_df[feature_cols].sample(min(1500, len(ml_df)), random_state=42)

    explainer = shap.TreeExplainer(models[shap_target])
    shap_values = explainer.shap_values(X_sample)

    fig = shap.summary_plot(shap_values, X_sample, show=False)
    st.pyplot(fig)

st.success("App Ready!")
