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


############################################################
# 🟩 COLUMN DEFINITION SYSTEM
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
    "country": "League identifier (e.g., ENG1, GER1, SRB1)",
    "season": "Season number or year (e.g., 2023)",
    "round": "Match round index",
    "home": "Home team name",
    "away": "Away team name",

    "home_goals": "Full-time goals scored by home team",
    "away_goals": "Full-time goals scored by away team",

    "home_win": "Closing odds for home win (1)",
    "draw": "Closing odds for draw (X)",
    "away_win": "Closing odds for away win (2)",

    "over2.5": "Odds for over 2.5 goals",
    "under2.5": "Odds for under 2.5 goals",
    "gg": "Both teams to score YES",
    "ng": "Both teams to score NO",

    "suth": "Home shots",
    "suta": "Away shots",
    "sutht": "Home shots on target",
    "sutat": "Away shots on target",

    "corh": "Home corners",
    "cora": "Away corners",

    "yellowh": "Home yellow cards",
    "yellowa": "Away yellow cards",

    "ballph": "Home possession %",
    "ballpa": "Away possession %",

    "foulsh": "Home fouls",
    "foulsa": "Away fouls",
}


############################################################
# 🟦 Column Helper Panel
############################################################

with st.expander("📘 Column Requirements & Template"):
    st.markdown("### Required Columns for ML Model")
    st.write("Your CSV **must** contain these columns:")

    for col in REQUIRED_COLUMNS:
        st.markdown(f"- **{col}** — {COLUMN_DESCRIPTIONS.get(col, '')}")

    st.markdown("---")
    st.markdown("### Optional Columns (improve accuracy)")
    for col in OPTIONAL_COLUMNS:
        st.markdown(f"- **{col}** — {COLUMN_DESCRIPTIONS.get(col, '')}")

    st.markdown("---")
    st.markdown("### Example Template Row")

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
# 🟦 CSV Upload
############################################################

uploaded = st.file_uploader("📤 Upload your match dataset (CSV)", type=["csv"])

if uploaded is None:
    st.info("Please upload your CSV file to continue.")
    st.stop()


############################################################
# 🟦 Load CSV (Cached)
############################################################

@st.cache_data
def load_csv(file):
    # Try reading as normal CSV first
    try:
        df = pd.read_csv(file)
        # Check if this looks valid (has necessary columns)
        if len(df.columns) > 5:
            return df
    except:
        pass

    # Fallback: single-column semicolon-separated CSV (your old format)
    raw = pd.read_csv(file, header=None)
    if raw.shape[1] == 1:
        df = raw[0].str.split(";", expand=True)
        header = df.iloc[0].tolist()
        df = df.drop(index=0).reset_index(drop=True)
        df.columns = header
        return df

    # If no method works
    raise ValueError("Unrecognized CSV format. Upload a normal CSV or semicolon-delimited single-column CSV.")

############################################################
# 🟥 Validate Columns
############################################################

missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

if missing:
    st.error("Your CSV is missing required columns:")
    for m in missing:
        st.write(f"- **{m}** — {COLUMN_DESCRIPTIONS.get(m, '')}")
    st.stop()


############################################################
# 🟦 Convert Types
############################################################

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")


############################################################
# 🟩 Odds Normalization + Expected Stats
############################################################

def normalize_odds(df):
    df = df.copy()
    cols = ["home_win","draw","away_win","over2.5","under2.5","gg","ng"]
    for c in cols:
        if c in df.columns:
            df[f"p_{c}"] = 1 / df[c]

    groups = {
        "1x2": ["p_home_win","p_draw","p_away_win"],
        "total": ["p_over2.5","p_under2.5"],
        "btts": ["p_gg","p_ng"],
    }
    for name, cols in groups.items():
        valid = [c for c in cols if c in df.columns]
        if len(valid) >= 2:
            s = df[valid].sum(axis=1)
            for v in valid:
                df[v.replace("p_","pn_")] = df[v] / s
    return df


def add_expected(df):
    df = df.copy()
    df["xG_total"] = 2.5 + (df["pn_over2.5"] - 0.5) * 2.8

    df["att_home"] = df["pn_home_win"] + 0.5 * df["pn_draw"]
    df["att_away"] = df["pn_away_win"] + 0.5 * df["pn_draw"]

    df["xG_home"] = df["xG_total"] * (df["att_home"] /
                         (df["att_home"] + df["att_away"]))
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

    foul_rate = (df.get("foulsh",0).sum() + df.get("foulsa",0).sum()) / max(1,len(df))

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
# 🟦 ML Dataset
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
# 🟦 TRAIN MODELS (cached)
############################################################

@st.cache_resource
def train_models():
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

models = train_models()


############################################################
# 🟦 PREDICTION UI
############################################################

st.header("🔮 Predict Future Match Stats")

league = st.selectbox("League", sorted(df["country"].unique()))
teams = sorted(df[df["country"] == league]["home"].unique())

col1, col2 = st.columns(2)
home = col1.selectbox("Home", teams)
away = col2.selectbox("Away", teams)

col3, col4, col5 = st.columns(3)
home_odds = col3.number_input("Home Win", value=2.00)
draw_odds = col4.number_input("Draw", value=3.30)
away_odds = col5.number_input("Away Win", value=3.40)

col6, col7, col8 = st.columns(3)
over25 = col6.number_input("Over 2.5", value=2.00)
under25 = col7.number_input("Under 2.5", value=1.85)
gg = col8.number_input("BTTS Yes", value=1.80)
ng = st.number_input("BTTS No", value=1.90)

if st.button("Predict Stats"):
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
        "home": home,
        "away": away,
        "season": df["season"].max(),
        "round": 1
    }])

    row = normalize_odds(row)
    row = add_expected(row)

    preds = {name: models[name].predict(row[feature_cols])[0]
             for name in models}

    st.subheader("📊 Predicted Stats")
    a, b = st.columns(2)
    a.metric("Home Shots", f"{preds['shots_home']:.1f}")
    b.metric("Away Shots", f"{preds['shots_away']:.1f}")

    c, d = st.columns(2)
    c.metric("Home Corners", f"{preds['corners_home']:.1f}")
    d.metric("Away Corners", f"{preds['corners_away']:.1f}")

    e, f = st.columns(2)
    e.metric("Home Cards", f"{preds['cards_home']:.2f}")
    f.metric("Away Cards", f"{preds['cards_away']:.2f}")


############################################################
# 🟦 League Predictability
############################################################

st.header("📈 League Predictability Index")

rows = []
for lg, subset in ml_df.groupby("country"):
    if len(subset) < 150:
        continue
    for model_name in models:
        y = subset[TARGET_COLS[model_name]]
        pred = models[model_name].predict(subset[feature_cols])
        rmse = mean_squared_error(y, pred, squared=False)
        std = y.std()
        rows.append({"league": lg, "target": model_name, "predictability": 1 - rmse/std})

st.dataframe(pd.DataFrame(rows))


############################################################
# 🟦 SHAP
############################################################

st.header("🧠 SHAP Explanations")

target = st.selectbox("Model to explain", list(models.keys()))

X_sample = ml_df[feature_cols].sample(min(800, len(ml_df)), random_state=42)
explainer = shap.TreeExplainer(models[target])
shap_vals = explainer.shap_values(X_sample)

fig = shap.summary_plot(shap_vals, X_sample, show=False)
st.pyplot(fig)

