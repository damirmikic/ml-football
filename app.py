###############################################
# app.py — Full Sports Prediction Web App
###############################################

import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import shap
import base64
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Football Match Predictor", layout="wide")

###############################################
# 1) Load and preprocess dataset
###############################################

@st.cache_data
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

###############################################
# 2) Odds normalization
###############################################

def normalize_odds(df):
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

    return df

df = normalize_odds(df)

###############################################
# 3) Expected metrics (xG, xShots, xCorners...)
###############################################

def add_expected_metrics(df):
    df["xG_total"] = 2.5 + (df["pn_over2.5"] - 0.5) * 2.8

    df["att_share_home"] = df["pn_home_win"] + 0.5 * df["pn_draw"]
    df["att_share_away"] = df["pn_away_win"] + 0.5 * df["pn_draw"]

    df["xG_home"] = df["xG_total"] * (
        df["att_share_home"] / (df["att_share_home"] + df["att_share_away"])
    )
    df["xG_away"] = df["xG_total"] - df["xG_home"]

    # Calibrated factors (derived earlier)
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

    theta_home = 0.176
    theta_away = 0.193

    df["tempo_factor"] = 0.5 + df["pn_over2.5"]
    df["xCards_home"] = df["xFouls_home"] * theta_home * df["tempo_factor"]
    df["xCards_away"] = df["xFouls_away"] * theta_away * df["tempo_factor"]

    return df

df = add_expected_metrics(df)

###############################################
# 4) Rolling form features
###############################################

@st.cache_data
def add_rolling_form(df):
    df = df.sort_values(["country","season","round"]).reset_index(drop=True)

    def side_form(df, side, team_col, prefix):
        g = df.groupby(["country","season", team_col])

        if side == "home":
            gf, ga = "home_goals", "away_goals"
            sh, sa = "suth", "suta"
            sot_f, sot_a = "sutht", "sutat"
            cor_f, cor_a = "corh", "cora"
            yc_f, yc_a = "yellowh", "yellowa"
            xg_f, xg_a = "xG_home", "xG_away"
            xp_f, xp_a = "xPts_home", "xPts_away"
        else:
            gf, ga = "away_goals", "home_goals"
            sh, sa = "suta", "suth"
            sot_f, sot_a = "sutat", "sutht"
            cor_f, cor_a = "cora", "corh"
            yc_f, yc_a = "yellowa", "yellowh"
            xg_f, xg_a = "xG_away", "xG_home"
            xp_f, xp_a = "xPts_away", "xPts_home"

        df[f"{prefix}_GF5"] = g[gf].shift(1).rolling(5,min_periods=1).mean()
        df[f"{prefix}_GA5"] = g[ga].shift(1).rolling(5,min_periods=1).mean()
        df[f"{prefix}_Sh5"] = g[sh].shift(1).rolling(5,min_periods=1).mean()
        df[f"{prefix}_ShA5"] = g[sa].shift(1).rolling(5,min_periods=1).mean()
        df[f"{prefix}_SOT5"] = g[sot_f].shift(1).rolling(5,min_periods=1).mean()
        df[f"{prefix}_SOTA5"] = g[sot_a].shift(1).rolling(5,min_periods=1).mean()
        df[f"{prefix}_Cor5"] = g[cor_f].shift(1).rolling(5,min_periods=1).mean()
        df[f"{prefix}_CorA5"] = g[cor_a].shift(1).rolling(5,min_periods=1).mean()
        df[f"{prefix}_YC5"] = g[yc_f].shift(1).rolling(5,min_periods=1).mean()
        df[f"{prefix}_YCA5"] = g[yc_a].shift(1).rolling(5,min_periods=1).mean()
        df[f"{prefix}_xG5"] = g[xg_f].shift(1).rolling(5,min_periods=1).mean()
        df[f"{prefix}_xGA5"] = g[xg_a].shift(1).rolling(5,min_periods=1).mean()
        df[f"{prefix}_xPts5"] = g[xp_f].shift(1).rolling(5,min_periods=1).mean()
        df[f"{prefix}_xPtsA5"] = g[xp_a].shift(1).rolling(5,min_periods=1).mean()

        return df

    df = side_form(df,"home","home","H")
    df = side_form(df,"away","away","A")
    return df

df = add_rolling_form(df)

###############################################
# 5) ML dataset creation
###############################################

target_cols = {
    "shots_home": "suth",
    "shots_away": "suta",
    "corners_home": "corh",
    "corners_away": "cora",
    "cards_home": "yellowh",
    "cards_away": "yellowa",
}

feature_cols = [c for c in df.columns if c not in target_cols.values()]

ml_df = df.dropna(subset=target_cols.values())

###############################################
# 6) Train models
###############################################

models = {}
for name, target in target_cols.items():
    X = ml_df[feature_cols]
    y = ml_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)
    models[name] = model

###############################################
# 7) Predictability index per league
###############################################

def league_predictability(ml_df):
    rows = []
    for lg, subset in ml_df.groupby("country"):
        if len(subset) < 200:
            continue

        for name, model in models.items():
            y = subset[target_cols[name]]
            pred = model.predict(subset[feature_cols])
            rmse = mean_squared_error(y, pred, squared=False)
            std = y.std()
            if std == 0:
                continue
            pred_idx = 1 - rmse/std
            rows.append({"league": lg, "target": name, "predictability": pred_idx})

    return pd.DataFrame(rows)

predict_df = league_predictability(ml_df)

###############################################
# 8) Web UI
###############################################

st.title("⚽ Football Match Predictor — Shots, Corners, Cards")

tab_predict, tab_leagues, tab_shap = st.tabs([
    "Predict Future Match",
    "League Predictability",
    "SHAP Feature Explanations"
])

###############################################
# PREDICTION UI
###############################################

with tab_predict:
    st.header("Predict a Future Match")

    leagues = sorted(df["country"].unique())
    league_sel = st.selectbox("League", leagues)

    teams = sorted(df[df["country"] == league_sel]["home"].unique())
    col1, col2 = st.columns(2)
    home = col1.selectbox("Home Team", teams)
    away = col2.selectbox("Away Team", teams)

    round_sel = st.number_input("Match Round", min_value=1, max_value=60, value=1)

    st.subheader("Enter Market Odds")
    oc1, oc2, oc3 = st.columns(3)
    home_win = oc1.number_input("Home Win Odds", value=2.00)
    draw = oc2.number_input("Draw Odds", value=3.30)
    away_win = oc3.number_input("Away Win Odds", value=3.40)

    oc4, oc5, oc6 = st.columns(3)
    over25 = oc4.number_input("Over 2.5", value=2.00)
    under25 = oc5.number_input("Under 2.5", value=1.85)
    gg = oc6.number_input("BTTS Yes", value=1.80)
    ng = st.number_input("BTTS No", value=1.90)

    if st.button("Predict Match Stats"):

        # Build input row
        row = pd.DataFrame([{
            **{c:0 for c in feature_cols},  # initialize
            "home_win": home_win,
            "draw": draw,
            "away_win": away_win,
            "over2.5": over25,
            "under2.5": under25,
            "gg": gg,
            "ng": ng,
            "country": league_sel,
            "season": df["season"].max(),
            "round": round_sel,
            "home": home,
            "away": away,
        }])

        # Normalize odds + expected metrics again
        row = normalize_odds(row)
        row = add_expected_metrics(row)

        # Add rolling form (fallback if missing)
        for col in feature_cols:
            if col not in row:
                row[col] = 0

        st.subheader("Predicted Match Statistics")

        predictions = {}
        for name, model in models.items():
            predictions[name] = float(model.predict(row[feature_cols])[0])

        colA, colB = st.columns(2)
        colA.metric("Home Shots", f"{predictions['shots_home']:.1f}")
        colB.metric("Away Shots", f"{predictions['shots_away']:.1f}")

        colC, colD = st.columns(2)
        colC.metric("Home Corners", f"{predictions['corners_home']:.1f}")
        colD.metric("Away Corners", f"{predictions['corners_away']:.1f}")

        colE, colF = st.columns(2)
        colE.metric("Home Cards", f"{predictions['cards_home']:.2f}")
        colF.metric("Away Cards", f"{predictions['cards_away']:.2f}")


###############################################
# LEAGUE PREDICTABILITY
###############################################

with tab_leagues:
    st.header("League Predictability Index")
    st.write("Higher = more predictable")

    st.dataframe(
        predict_df.sort_values(["target","predictability"], ascending=False),
        height=600
    )


###############################################
# SHAP TAB
###############################################

with tab_shap:
    st.header("SHAP Feature Explanations")

    shap_target = st.selectbox(
        "Select model for SHAP analysis",
        list(models.keys())
    )

    X_sample = ml_df[feature_cols].sample(1500, random_state=42)
    explainer = shap.TreeExplainer(models[shap_target])
    shap_values = explainer.shap_values(X_sample)

    st.subheader("Feature Importance Summary")
    fig = shap.summary_plot(shap_values, X_sample, show=False)
    st.pyplot(fig)

st.success("App loaded successfully!")
