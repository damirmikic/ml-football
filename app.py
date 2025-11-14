import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import shap
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Football ML Predictor", layout="wide")

st.title("⚽ Football ML Predictor (Upload CSV Version)")


############################################################
# 1) UPLOAD CSV
############################################################

uploaded = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded is None:
    st.info("Please upload your mldatafootball.csv to begin.")
    st.stop()

@st.cache_data
def load_csv(uploaded):
    raw = pd.read_csv(uploaded, header=None)
    df = raw[0].str.split(";", expand=True)
    header = df.iloc[0].tolist()
    df = df.drop(index=0).reset_index(drop=True)
    df.columns = header

    df = df.replace({"": np.nan, "None": np.nan})
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df

df = load_csv(uploaded)


############################################################
# 2) ODDS NORMALIZATION + EXPECTED METRICS
############################################################

def normalize_odds(df):
    df = df.copy()
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


def add_expected_metrics(df):
    df = df.copy()

    df["xG_total"] = 2.5 + (df["pn_over2.5"] - 0.5) * 2.8
    df["att_share_home"] = df["pn_home_win"] + 0.5 * df["pn_draw"]
    df["att_share_away"] = df["pn_away_win"] + 0.5 * df["pn_draw"]

    df["xG_home"] = df["xG_total"] * (
        df["att_share_home"] / (df["att_share_home"] + df["att_share_away"])
    )
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


df = normalize_odds(df)
df = add_expected_metrics(df)


############################################################
# 3) BUILD ML DATASET
############################################################

TARGET_COLS = {
    "shots_home": "suth",
    "shots_away": "suta",
    "corners_home": "corh",
    "corners_away": "cora",
    "cards_home": "yellowh",
    "cards_away": "yellowa",
}

ml_df = df.dropna(subset=TARGET_COLS.values())
feature_cols = [c for c in df.columns if c not in TARGET_COLS.values()]


############################################################
# 4) TRAIN MODELS (CACHED)
############################################################

@st.cache_resource
def train_models():
    models = {}
    for name, target in TARGET_COLS.items():
        X = ml_df[feature_cols]
        y = ml_df[target]

        model = XGBRegressor(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
        )
        model.fit(X, y)
        models[name] = model
    return models

models = train_models()


############################################################
# 5) PREDICT MATCH UI
############################################################

st.header("Predict Future Match")

leagues = sorted(df["country"].unique())
league = st.selectbox("League", leagues)

teams = sorted(df[df["country"] == league]["home"].unique())
col1, col2 = st.columns(2)
home = col1.selectbox("Home Team", teams)
away = col2.selectbox("Away Team", teams)

st.subheader("Enter Odds")
c1, c2, c3 = st.columns(3)
home_win = c1.number_input("Home Win", value=2.00)
draw = c2.number_input("Draw", value=3.30)
away_win = c3.number_input("Away Win", value=3.40)

c4, c5, c6 = st.columns(3)
over25 = c4.number_input("Over 2.5", value=2.00)
under25 = c5.number_input("Under 2.5", value=1.85)
gg = c6.number_input("BTTS Yes", value=1.80)
ng = st.number_input("BTTS No", value=1.90)

if st.button("Predict Stats"):
    row = pd.DataFrame([{
        **{c:0 for c in feature_cols},
        "home_win": home_win,
        "draw": draw,
        "away_win": away_win,
        "over2.5": over25,
        "under2.5": under25,
        "gg": gg,
        "ng": ng,
        "country": league,
        "home": home,
        "away": away,
        "season": df["season"].max(),
        "round": 1
    }])

    row = normalize_odds(row)
    row = add_expected_metrics(row)

    preds = {name: models[name].predict(row[feature_cols])[0]
             for name in models}

    st.subheader("Predicted Statistics")

    ca, cb = st.columns(2)
    ca.metric("Home Shots", f"{preds['shots_home']:.1f}")
    cb.metric("Away Shots", f"{preds['shots_away']:.1f}")

    cc, cd = st.columns(2)
    cc.metric("Home Corners", f"{preds['corners_home']:.1f}")
    cd.metric("Away Corners", f"{preds['corners_away']:.1f}")

    ce, cf = st.columns(2)
    ce.metric("Home Cards", f"{preds['cards_home']:.2f}")
    cf.metric("Away Cards", f"{preds['cards_away']:.2f}")


############################################################
# 6) LEAGUE PREDICTABILITY
############################################################

st.header("League Predictability")

rows = []
for lg, subset in ml_df.groupby("country"):
    if len(subset) < 200:
        continue
    for name in models:
        y = subset[TARGET_COLS[name]]
        pred = models[name].predict(subset[feature_cols])
        rmse = mean_squared_error(y, pred, squared=False)
        std = y.std()
        rows.append({"league": lg, "target": name, "predictability": 1 - rmse/std})

pred_table = pd.DataFrame(rows)
st.dataframe(pred_table.sort_values(["target","predictability"], ascending=False))


############################################################
# 7) SHAP INTERPRETATION
############################################################

st.header("SHAP Explainability")

model_name = st.selectbox("Select target", list(models.keys()))

st.write("Computing SHAP (first run may take 10 seconds)...")

X_sample = ml_df[feature_cols].sample(min(800, len(ml_df)), random_state=42)
explainer = shap.TreeExplainer(models[model_name])
shap_vals = explainer.shap_values(X_sample)

fig = shap.summary_plot(shap_vals, X_sample, show=False)
st.pyplot(fig)
