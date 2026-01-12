# ⚽ Expected Points (xPTS) Calculator

An automated web application for calculating expected points from betting odds in football/soccer matches with advanced league analytics.

## 🎯 What It Does

This application:
1. **Cleans missing data** - Removes fixtures with invalid or missing odds
2. **Calculates true probabilities** - Removes bookmaker's margin from odds
3. **Computes xPTS** - Calculates expected points for home and away teams
4. **League standings** - Creates comprehensive standings tables with performance metrics
5. **Color-coded analysis** - Visual gradients highlight over/underperformers
6. **Multi-season support** - Analyze single or multiple seasons together

## ✨ New Features

### Form Analysis & Trends 🔥
Advanced performance tracking and momentum indicators:

- **Rolling xPTS Analysis**
  - 5-game, 10-game, and 15-game rolling averages
  - Actual points vs expected points trends
  - Performance vs expectation over time

- **Interactive Form Charts**
  - Time-series visualization of team performance
  - Rolling averages with individual match results
  - Momentum charts showing over/underperformance

- **Momentum Indicators**
  - 🔥 Hot Streak (consistently outperforming)
  - 📈 Good Form (performing above expectations)
  - ➡️ Average (meeting expectations)
  - 📉 Poor Form (underperforming)
  - ❄️ Cold Streak (consistently underperforming)

- **Recent Form Tables**
  - Last 5, 10, 15 matches with color-coded results
  - Win/Draw/Loss streaks
  - Points and xPTS comparison
  - Match-by-match performance breakdown

### League Standings Tables
When you select a specific league, the app automatically generates:

- **Overall Performance Table**
  - Actual points vs xPTS with color-coded differences
  - Points per game (PPG) with gradient highlighting
  - Overperformance metrics (actual - expected)

- **Home Performance Table**
  - Home-specific statistics
  - Home PPG with color gradients
  - Home over/underperformance

- **Away Performance Table**
  - Away-specific statistics
  - Away PPG with color gradients
  - Away over/underperformance

### Multi-Season Selection
- Choose "All Seasons" for comprehensive analysis
- Select specific season(s) for targeted insights
- Multi-select allows combining multiple seasons

### Color Gradients
- 🟢 **Green** = Overperformers (earning more points than expected)
- 🟡 **Yellow** = Performing as expected
- 🔴 **Red** = Underperformers (earning fewer points than expected)

## 📊 How xPTS is Calculated

### Step 1: Implied Probabilities
From decimal odds, calculate implied probabilities:
- P(Home) = 1 / Home Odds
- P(Draw) = 1 / Draw Odds  
- P(Away) = 1 / Away Odds

### Step 2: Remove Bookmaker's Margin
Bookmakers build in a profit margin (overround):
- Total = P(Home) + P(Draw) + P(Away) > 1.0
- Margin = Total - 1.0
- True Probability = Implied Probability / Total

This normalizes probabilities to sum to exactly 1.0

### Step 3: Calculate xPTS
Expected points based on true probabilities:
- **xPTS Home** = 3 × P(Home Win) + 1 × P(Draw) + 0 × P(Loss)
- **xPTS Away** = 3 × P(Away Win) + 1 × P(Draw) + 0 × P(Loss)

## 📁 Data Format

The Excel file should contain these columns:

| Column | Description |
|--------|-------------|
| country | League/country code |
| sezonul | Season |
| etapa | Round/matchday number |
| txtechipa1 | Home team name |
| txtechipa2 | Away team name |
| scor1 | Home team final score |
| scor2 | Away team final score |
| cotaa | Home win odds (decimal) |
| cotae | Draw odds (decimal) |
| cotad | Away win odds (decimal) |

## 🚀 Running the App

### Option 1: Using Streamlit Cloud
```bash
streamlit run xpts_calculator.py
```

### Option 2: Local Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run xpts_calculator.py
```

Then open your browser to `http://localhost:8501`

## 🔧 Features

### Data Processing
- Automatic removal of invalid odds (< 1.01)
- Missing data handling
- Margin calculation and removal
- True probability computation

### Analytics
- Filter by country, season, or team
- Team-specific statistics (total xPTS, actual points, overperformance)
- Summary statistics (average margin, fixtures count)
- Visual data tables with formatted probabilities

### Export
- Download processed data as CSV
- Includes all calculated fields (true probabilities, xPTS, actual points)

## 📈 Use Cases

1. **Form-Based Betting** - Identify teams on hot/cold streaks for informed betting
2. **Performance Trends** - Track team momentum across different time windows
3. **League Performance Analysis** - Identify teams over/underperforming their expected points
4. **Team Comparison** - Compare home vs away performance across seasons
5. **Value Detection** - Find matches where bookmaker odds diverge from form-adjusted expectations
6. **Multi-Season Trends** - Track team performance evolution and consistency
7. **Regression Analysis** - Identify teams likely to regress to the mean
8. **Streak Identification** - Find teams in hot/cold streaks for strategic betting

## 📊 Sample League Standings Output

```
English Premier League (Seasons 23-24)

TOP 5 TEAMS:
Pos  Team           M    Pts   xPTS   Diff    PPG   xPPG
1    Man City      76   180   174.4   +5.6   2.37  2.29
2    Arsenal       76   173   155.2  +17.8   2.28  2.04  🟢 Overperformer
3    Liverpool     76   149   153.1   -4.1   1.96  2.01
4    Man Utd       76   135   123.3  +11.7   1.78  1.62  🟢 Overperformer
5    Newcastle     76   131   123.7   +7.3   1.72  1.63

BIGGEST OVERPERFORMERS:
- Aston Villa: +21.9 points above xPTS
- Arsenal: +17.8 points above xPTS

BIGGEST UNDERPERFORMERS:
- Chelsea: -21.0 points below xPTS
- Leeds: -11.9 points below xPTS
```

## 📈 Sample Form Analysis Output

```
Arsenal - Form Analysis

CURRENT MOMENTUM: 🔥 Hot Streak
Last 5 Games: W W W W W
Performance vs xPTS: +0.64 PPG

ROLLING AVERAGES:
Last 5:  15 pts (3.00 PPG) vs 11.8 xPTS (2.36 PPG) → +3.2 overperformance
Last 10: 25 pts (2.50 PPG) vs 22.4 xPTS (2.24 PPG) → +2.6 overperformance
Last 15: 40 pts (2.67 PPG) vs 34.3 xPTS (2.29 PPG) → +5.7 overperformance

SEASON COMPARISON:
Season 23: 84 pts (2.21 PPG) vs 73.6 xPTS → +10.4 overperformance
Season 24: 89 pts (2.34 PPG) vs 81.5 xPTS → +7.5 overperformance
```

## 🧪 Validation Results

Based on 45,168 fixtures:
- ✅ All probabilities sum to exactly 1.0 after margin removal
- ✅ Average bookmaker margin: 3.95%
- ✅ Average home xPTS: 1.565 points
- ✅ Average away xPTS: 1.173 points
- ✅ Combined average: 2.738 points (close to theoretical 3.0)

## 📊 Sample Output

```
Match: G. Chivas vs Juarez
Odds: Home=1.85, Draw=3.35, Away=5.28
Margin: 2.84%

True Probabilities:
- Home Win: 52.6%
- Draw: 29.0%
- Away Win: 18.4%

Expected Points:
- Home: 1.87 xPTS
- Away: 0.84 xPTS
```

## 🛠️ Technologies Used

- **Streamlit** - Web application framework
- **Pandas** - Data processing and analysis
- **NumPy** - Numerical computations
- **Plotly** - Interactive charts and visualizations
- **Matplotlib** - Color gradients and styling
- **OpenPyXL** - Excel file handling

## 🎨 Features Overview

### Visual Analytics
- Color-coded performance tables
- Green gradients for overperformers
- Red gradients for underperformers
- PPG (Points Per Game) highlighting
- Interactive Plotly charts
- Time-series trend visualization
- Performance momentum indicators

### Form Analysis Tools
- Rolling averages (5, 10, 15 games)
- Match-by-match performance tracking
- Streak identification (W/D/L patterns)
- Momentum classification
- Season-by-season comparison
- Downloadable match history

### Flexible Filtering
- League/country selection
- Multi-season selection
- Individual team analysis
- All seasons or specific season(s)

## 📝 Notes

- The proportional method is used for margin removal (most common approach)
- Invalid odds (< 1.01) are automatically filtered out
- Some extreme margins may indicate data quality issues
- xPTS represents long-term expected value, not single match predictions

## 🤝 Contributing

This is a specialized tool for betting analytics. Suggestions for improvements:
- Additional margin removal methods (additive, power, etc.)
- More detailed team statistics
- Historical trend visualization
- Comparison with alternative probability models

## ⚠️ Disclaimer

This tool is for analytical and educational purposes only. Expected points are probabilistic estimates and do not guarantee outcomes. Always bet responsibly and within your means.

---

**Built for professional betting analytics** 📊⚽
