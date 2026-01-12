# ⚽ Expected Points (xPTS) Calculator

An automated web application for calculating expected points from betting odds in football/soccer matches.

## 🎯 What It Does

This application:
1. **Cleans missing data** - Removes fixtures with invalid or missing odds
2. **Calculates true probabilities** - Removes bookmaker's margin from odds
3. **Computes xPTS** - Calculates expected points for home and away teams
4. **Provides analytics** - Team statistics, filtering, and data export

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

1. **Team Performance Analysis** - Compare expected vs actual points to identify overperforming/underperforming teams
2. **Betting Value** - Find matches where bookmaker odds diverge from true probabilities
3. **League Analysis** - Compare bookmaker margins across different leagues
4. **Historical Trends** - Analyze team xPTS trends over multiple seasons

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
- **Pandas** - Data processing
- **NumPy** - Numerical computations
- **OpenPyXL** - Excel file handling

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
