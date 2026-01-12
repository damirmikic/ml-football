import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="xPTS Calculator", page_icon="⚽", layout="wide")

st.title("⚽ Expected Points (xPTS) Calculator")
st.markdown("Calculate expected points from betting odds with automatic margin removal")

# Helper functions
def calculate_implied_probabilities(home_odds, draw_odds, away_odds):
    """Calculate implied probabilities from decimal odds"""
    prob_home = 1 / home_odds
    prob_draw = 1 / draw_odds
    prob_away = 1 / away_odds
    return prob_home, prob_draw, prob_away

def remove_margin(prob_home, prob_draw, prob_away):
    """Remove bookmaker's margin using proportional method"""
    total = prob_home + prob_draw + prob_away
    margin = total - 1
    
    # Normalize to get true probabilities
    true_prob_home = prob_home / total
    true_prob_draw = prob_draw / total
    true_prob_away = prob_away / total
    
    return true_prob_home, true_prob_draw, true_prob_away, margin

def calculate_xpts(prob_win, prob_draw):
    """Calculate expected points: 3 * P(win) + 1 * P(draw) + 0 * P(loss)"""
    return 3 * prob_win + prob_draw

def create_league_standings(df, league, seasons):
    """Create comprehensive league standings with xPTS and performance metrics"""
    # Filter data
    df_league = df[df['country'] == league].copy()
    if seasons:
        df_league = df_league[df_league['sezonul'].isin(seasons)]
    
    # Collect all teams
    all_teams = set(df_league['txtechipa1'].unique()) | set(df_league['txtechipa2'].unique())
    
    standings_data = []
    
    for team in all_teams:
        # Home games
        home_games = df_league[df_league['txtechipa1'] == team]
        # Away games
        away_games = df_league[df_league['txtechipa2'] == team]
        
        if len(home_games) == 0 and len(away_games) == 0:
            continue
        
        # Home statistics
        home_xpts = home_games['xPTS_home'].sum()
        home_actual = home_games['actual_pts_home'].sum()
        home_matches = len(home_games)
        
        # Away statistics
        away_xpts = away_games['xPTS_away'].sum()
        away_actual = away_games['actual_pts_away'].sum()
        away_matches = len(away_games)
        
        # Overall statistics
        total_xpts = home_xpts + away_xpts
        total_actual = home_actual + away_actual
        total_matches = home_matches + away_matches
        overperformance = total_actual - total_xpts
        
        standings_data.append({
            'Team': team,
            'Matches': total_matches,
            'Home_Matches': home_matches,
            'Away_Matches': away_matches,
            'Actual_Pts': total_actual,
            'xPTS': total_xpts,
            'Diff': overperformance,
            'Home_xPTS': home_xpts,
            'Home_Actual': home_actual,
            'Home_Diff': home_actual - home_xpts,
            'Away_xPTS': away_xpts,
            'Away_Actual': away_actual,
            'Away_Diff': away_actual - away_xpts,
            'Avg_xPTS': total_xpts / total_matches if total_matches > 0 else 0,
            'Avg_Actual': total_actual / total_matches if total_matches > 0 else 0,
        })
    
    standings_df = pd.DataFrame(standings_data)
    
    if len(standings_df) > 0:
        standings_df = standings_df.sort_values('Actual_Pts', ascending=False).reset_index(drop=True)
        standings_df.index = standings_df.index + 1
    
    return standings_df

def calculate_team_form(df, team, window_sizes=[5, 10, 15]):
    """Calculate rolling form for a team across different window sizes"""
    # Get all games for the team (sorted by season and round)
    home_games = df[df['txtechipa1'] == team].copy()
    away_games = df[df['txtechipa2'] == team].copy()
    
    # Add venue column
    home_games['venue'] = 'Home'
    home_games['team_xpts'] = home_games['xPTS_home']
    home_games['team_actual'] = home_games['actual_pts_home']
    home_games['opponent'] = home_games['txtechipa2']
    
    away_games['venue'] = 'Away'
    away_games['team_xpts'] = away_games['xPTS_away']
    away_games['team_actual'] = away_games['actual_pts_away']
    away_games['opponent'] = away_games['txtechipa1']
    
    # Combine and sort by season and round
    all_games = pd.concat([home_games, away_games], ignore_index=True)
    all_games = all_games.sort_values(['sezonul', 'etapa']).reset_index(drop=True)
    
    # Calculate rolling averages
    form_data = []
    for window in window_sizes:
        rolling_xpts = all_games['team_xpts'].rolling(window=window, min_periods=1).mean()
        rolling_actual = all_games['team_actual'].rolling(window=window, min_periods=1).mean()
        
        form_data.append({
            f'rolling_xpts_{window}': rolling_xpts,
            f'rolling_actual_{window}': rolling_actual,
            f'rolling_diff_{window}': rolling_actual - rolling_xpts
        })
    
    # Add form data to all_games
    for data in form_data:
        for key, values in data.items():
            all_games[key] = values
    
    return all_games

def calculate_momentum(recent_games, window=5):
    """Calculate momentum indicator based on recent performance"""
    if len(recent_games) < window:
        return "Insufficient Data"
    
    recent = recent_games.tail(window)
    diff = (recent['team_actual'] - recent['team_xpts']).mean()
    
    if diff > 0.5:
        return "🔥 Hot Streak"
    elif diff > 0.2:
        return "📈 Good Form"
    elif diff > -0.2:
        return "➡️ Average"
    elif diff > -0.5:
        return "📉 Poor Form"
    else:
        return "❄️ Cold Streak"

def get_form_string(recent_games, n=5):
    """Get last N results as a string (W/D/L)"""
    if len(recent_games) == 0:
        return "N/A"
    
    results = []
    for _, game in recent_games.tail(n).iterrows():
        if game['team_actual'] == 3:
            results.append('W')
        elif game['team_actual'] == 1:
            results.append('D')
        else:
            results.append('L')
    
    return ' '.join(results)

def create_form_chart(team_data, team_name):
    """Create interactive form chart showing xPTS vs actual points over time"""
    fig = go.Figure()
    
    # Add actual points line
    fig.add_trace(go.Scatter(
        x=list(range(1, len(team_data) + 1)),
        y=team_data['rolling_actual_10'],
        mode='lines+markers',
        name='Actual PPG (10-game avg)',
        line=dict(color='#2E86AB', width=3),
        marker=dict(size=6)
    ))
    
    # Add xPTS line
    fig.add_trace(go.Scatter(
        x=list(range(1, len(team_data) + 1)),
        y=team_data['rolling_xpts_10'],
        mode='lines+markers',
        name='xPTS PPG (10-game avg)',
        line=dict(color='#A23B72', width=3, dash='dash'),
        marker=dict(size=6)
    ))
    
    # Add individual match points as background
    fig.add_trace(go.Scatter(
        x=list(range(1, len(team_data) + 1)),
        y=team_data['team_actual'],
        mode='markers',
        name='Match Result',
        marker=dict(
            size=8,
            color=team_data['team_actual'].map({3: 'green', 1: 'orange', 0: 'red'}),
            opacity=0.3
        ),
        showlegend=True
    ))
    
    fig.update_layout(
        title=f"{team_name} - Form Analysis (Rolling 10-Game Average)",
        xaxis_title="Match Number",
        yaxis_title="Points Per Game",
        hovermode='x unified',
        height=500,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_momentum_chart(team_data, team_name):
    """Create chart showing performance vs expectation over time"""
    fig = go.Figure()
    
    # Calculate difference for each match
    team_data['diff'] = team_data['team_actual'] - team_data['team_xpts']
    
    # Create bar chart
    colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in team_data['diff']]
    
    fig.add_trace(go.Bar(
        x=list(range(1, len(team_data) + 1)),
        y=team_data['diff'],
        marker_color=colors,
        name='Performance vs Expectation',
        hovertemplate='Match %{x}<br>Diff: %{y:.2f}<extra></extra>'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3)
    
    # Add rolling average of difference
    fig.add_trace(go.Scatter(
        x=list(range(1, len(team_data) + 1)),
        y=team_data['rolling_diff_10'],
        mode='lines',
        name='10-Game Avg Difference',
        line=dict(color='blue', width=3)
    ))
    
    fig.update_layout(
        title=f"{team_name} - Performance vs Expectation",
        xaxis_title="Match Number",
        yaxis_title="Points Above/Below xPTS",
        hovermode='x unified',
        height=400,
        template='plotly_white',
        showlegend=True
    )
    
    return fig

def style_dataframe_with_gradient(df, column, cmap='RdYlGn', vmin=None, vmax=None):
    """Apply color gradient to a dataframe column"""
    if vmin is None:
        vmin = df[column].min()
    if vmax is None:
        vmax = df[column].max()
    
    # Normalize values
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.get_cmap(cmap)
    
    def color_cell(val):
        if pd.isna(val):
            return ''
        color = mcolors.rgb2hex(cmap(norm(val)))
        return f'background-color: {color}'
    
    return color_cell

def process_data(df):
    """Process the dataset and calculate xPTS"""
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    # Remove rows with missing odds
    initial_rows = len(df_processed)
    df_processed = df_processed.dropna(subset=['cotaa', 'cotae', 'cotad'])
    rows_removed = initial_rows - len(df_processed)
    
    # Remove rows with invalid odds (odds must be >= 1.01)
    df_processed = df_processed[
        (df_processed['cotaa'] >= 1.01) & 
        (df_processed['cotae'] >= 1.01) & 
        (df_processed['cotad'] >= 1.01)
    ]
    
    # Calculate implied probabilities
    df_processed['implied_prob_home'] = 1 / df_processed['cotaa']
    df_processed['implied_prob_draw'] = 1 / df_processed['cotae']
    df_processed['implied_prob_away'] = 1 / df_processed['cotad']
    
    # Calculate total (margin)
    df_processed['total_implied'] = (
        df_processed['implied_prob_home'] + 
        df_processed['implied_prob_draw'] + 
        df_processed['implied_prob_away']
    )
    df_processed['margin'] = df_processed['total_implied'] - 1
    
    # Calculate true probabilities (remove margin)
    df_processed['true_prob_home'] = df_processed['implied_prob_home'] / df_processed['total_implied']
    df_processed['true_prob_draw'] = df_processed['implied_prob_draw'] / df_processed['total_implied']
    df_processed['true_prob_away'] = df_processed['implied_prob_away'] / df_processed['total_implied']
    
    # Calculate xPTS
    df_processed['xPTS_home'] = calculate_xpts(
        df_processed['true_prob_home'], 
        df_processed['true_prob_draw']
    )
    df_processed['xPTS_away'] = calculate_xpts(
        df_processed['true_prob_away'], 
        df_processed['true_prob_draw']
    )
    
    # Calculate actual points
    df_processed['actual_pts_home'] = df_processed.apply(
        lambda row: 3 if row['scor1'] > row['scor2'] 
        else (1 if row['scor1'] == row['scor2'] else 0), 
        axis=1
    )
    df_processed['actual_pts_away'] = df_processed.apply(
        lambda row: 3 if row['scor2'] > row['scor1'] 
        else (1 if row['scor1'] == row['scor2'] else 0), 
        axis=1
    )
    
    return df_processed, rows_removed

# File upload
uploaded_file = st.file_uploader("Upload Excel file with betting data", type=['xlsx', 'xls'])

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_excel(uploaded_file)
        
        st.success(f"✅ File loaded successfully! {len(df):,} rows found")
        
        # Show original data structure
        with st.expander("📊 View Original Data Structure"):
            st.write(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
            st.write("**Columns:**", df.columns.tolist())
            st.dataframe(df.head(10))
        
        # Process data
        with st.spinner("Processing data and calculating xPTS..."):
            df_processed, rows_removed = process_data(df)
        
        # Display summary statistics
        st.header("📈 Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Fixtures", f"{len(df_processed):,}")
        with col2:
            st.metric("Rows Cleaned", f"{rows_removed:,}")
        with col3:
            st.metric("Avg Margin", f"{df_processed['margin'].mean():.2%}")
        with col4:
            st.metric("Countries", df_processed['country'].nunique())
        
        # Display processed data
        st.header("🎯 Calculated xPTS Data")
        
        # Select columns to display
        display_columns = [
            'country', 'sezonul', 'etapa', 'txtechipa1', 'txtechipa2',
            'cotaa', 'cotae', 'cotad',
            'true_prob_home', 'true_prob_draw', 'true_prob_away',
            'xPTS_home', 'xPTS_away',
            'scor1', 'scor2', 'actual_pts_home', 'actual_pts_away'
        ]
        
        df_display = df_processed[display_columns].copy()
        
        # Format probabilities and xPTS
        prob_cols = ['true_prob_home', 'true_prob_draw', 'true_prob_away']
        for col in prob_cols:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.1%}")
        
        xpts_cols = ['xPTS_home', 'xPTS_away']
        for col in xpts_cols:
            df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(df_display, use_container_width=True, height=400)
        
        # Filters
        st.header("🔍 Filter Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            countries = ['All'] + sorted(df_processed['country'].unique().tolist())
            selected_country = st.selectbox("League/Country", countries, key='country_filter')
        
        with col2:
            all_seasons = sorted(df_processed['sezonul'].unique().tolist())
            if selected_country != 'All':
                available_seasons = sorted(df_processed[df_processed['country'] == selected_country]['sezonul'].unique().tolist())
            else:
                available_seasons = all_seasons
            
            season_option = st.radio("Season Selection", ["All Seasons", "Specific Season(s)"], horizontal=True)
            
            if season_option == "All Seasons":
                selected_seasons = available_seasons
            else:
                selected_seasons = st.multiselect(
                    "Select Season(s)", 
                    available_seasons,
                    default=[available_seasons[-1]] if available_seasons else []
                )
        
        with col3:
            teams = ['All'] + sorted(
                set(df_processed['txtechipa1'].unique().tolist() + 
                    df_processed['txtechipa2'].unique().tolist())
            )
            selected_team = st.selectbox("Team", teams, key='team_filter')
        
        # Apply filters
        df_filtered = df_processed.copy()
        
        if selected_country != 'All':
            df_filtered = df_filtered[df_filtered['country'] == selected_country]
        
        if season_option == "Specific Season(s)" and selected_seasons:
            df_filtered = df_filtered[df_filtered['sezonul'].isin(selected_seasons)]
        
        if selected_team != 'All':
            df_filtered = df_filtered[
                (df_filtered['txtechipa1'] == selected_team) | 
                (df_filtered['txtechipa2'] == selected_team)
            ]
        
        if len(df_filtered) > 0:
            st.write(f"**Filtered Results:** {len(df_filtered):,} fixtures")
            
            # League Standings Tables (only show when a specific league is selected)
            if selected_country != 'All':
                st.header(f"📊 {selected_country} League Standings")
                
                season_display = "All Seasons" if season_option == "All Seasons" else f"Season(s): {', '.join(map(str, selected_seasons))}"
                st.subheader(season_display)
                
                # Create standings
                standings = create_league_standings(df_filtered, selected_country, selected_seasons if season_option == "Specific Season(s)" else None)
                
                if len(standings) > 0:
                    # Create tabs for different views
                    tab1, tab2, tab3 = st.tabs(["📈 Overall Performance", "🏠 Home Performance", "✈️ Away Performance"])
                    
                    with tab1:
                        st.markdown("### Overall Standings")
                        
                        # Prepare display dataframe
                        overall_display = standings[['Team', 'Matches', 'Actual_Pts', 'xPTS', 'Diff', 'Avg_Actual', 'Avg_xPTS']].copy()
                        overall_display.columns = ['Team', 'M', 'Pts', 'xPTS', 'Diff', 'PPG', 'xPPG']
                        
                        # Format numbers
                        overall_display['xPTS'] = overall_display['xPTS'].round(1)
                        overall_display['Diff'] = overall_display['Diff'].round(1)
                        overall_display['PPG'] = overall_display['PPG'].round(2)
                        overall_display['xPPG'] = overall_display['xPPG'].round(2)
                        
                        # Apply styling
                        styled_overall = overall_display.style.background_gradient(
                            subset=['Diff'],
                            cmap='RdYlGn',
                            vmin=-15,
                            vmax=15
                        ).background_gradient(
                            subset=['PPG'],
                            cmap='YlGn',
                            vmin=0,
                            vmax=3
                        ).format({
                            'xPTS': '{:.1f}',
                            'Diff': '{:+.1f}',
                            'PPG': '{:.2f}',
                            'xPPG': '{:.2f}'
                        })
                        
                        st.dataframe(styled_overall, use_container_width=True, height=600)
                        
                        # Statistics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Avg Points", f"{overall_display['Pts'].mean():.1f}")
                        with col2:
                            st.metric("Avg xPTS", f"{overall_display['xPTS'].mean():.1f}")
                        with col3:
                            best_overperformer = overall_display.loc[overall_display['Diff'].idxmax()]
                            st.metric("Best Overperformer", best_overperformer['Team'], f"+{best_overperformer['Diff']:.1f}")
                        with col4:
                            worst_underperformer = overall_display.loc[overall_display['Diff'].idxmin()]
                            st.metric("Biggest Underperformer", worst_underperformer['Team'], f"{worst_underperformer['Diff']:.1f}")
                    
                    with tab2:
                        st.markdown("### Home Performance")
                        
                        # Prepare home display dataframe
                        home_display = standings[['Team', 'Home_Matches', 'Home_Actual', 'Home_xPTS', 'Home_Diff']].copy()
                        home_display.columns = ['Team', 'M', 'Pts', 'xPTS', 'Diff']
                        home_display = home_display.sort_values('Pts', ascending=False).reset_index(drop=True)
                        home_display.index = home_display.index + 1
                        
                        # Format numbers
                        home_display['xPTS'] = home_display['xPTS'].round(1)
                        home_display['Diff'] = home_display['Diff'].round(1)
                        
                        # Add averages
                        home_display['PPG'] = (home_display['Pts'] / home_display['M']).round(2)
                        home_display['xPPG'] = (home_display['xPTS'] / home_display['M']).round(2)
                        
                        # Apply styling
                        styled_home = home_display.style.background_gradient(
                            subset=['Diff'],
                            cmap='RdYlGn',
                            vmin=-10,
                            vmax=10
                        ).background_gradient(
                            subset=['PPG'],
                            cmap='YlGn',
                            vmin=0,
                            vmax=3
                        ).format({
                            'xPTS': '{:.1f}',
                            'Diff': '{:+.1f}',
                            'PPG': '{:.2f}',
                            'xPPG': '{:.2f}'
                        })
                        
                        st.dataframe(styled_home, use_container_width=True, height=600)
                        
                        # Home statistics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Avg Home PPG", f"{home_display['PPG'].mean():.2f}")
                        with col2:
                            st.metric("Avg Home xPPG", f"{home_display['xPPG'].mean():.2f}")
                    
                    with tab3:
                        st.markdown("### Away Performance")
                        
                        # Prepare away display dataframe
                        away_display = standings[['Team', 'Away_Matches', 'Away_Actual', 'Away_xPTS', 'Away_Diff']].copy()
                        away_display.columns = ['Team', 'M', 'Pts', 'xPTS', 'Diff']
                        away_display = away_display.sort_values('Pts', ascending=False).reset_index(drop=True)
                        away_display.index = away_display.index + 1
                        
                        # Format numbers
                        away_display['xPTS'] = away_display['xPTS'].round(1)
                        away_display['Diff'] = away_display['Diff'].round(1)
                        
                        # Add averages
                        away_display['PPG'] = (away_display['Pts'] / away_display['M']).round(2)
                        away_display['xPPG'] = (away_display['xPTS'] / away_display['M']).round(2)
                        
                        # Apply styling
                        styled_away = away_display.style.background_gradient(
                            subset=['Diff'],
                            cmap='RdYlGn',
                            vmin=-10,
                            vmax=10
                        ).background_gradient(
                            subset=['PPG'],
                            cmap='YlGn',
                            vmin=0,
                            vmax=3
                        ).format({
                            'xPTS': '{:.1f}',
                            'Diff': '{:+.1f}',
                            'PPG': '{:.2f}',
                            'xPPG': '{:.2f}'
                        })
                        
                        st.dataframe(styled_away, use_container_width=True, height=600)
                        
                        # Away statistics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Avg Away PPG", f"{away_display['PPG'].mean():.2f}")
                        with col2:
                            st.metric("Avg Away xPPG", f"{away_display['xPPG'].mean():.2f}")
                
                st.markdown("---")
            
            # Calculate team-specific stats if a team is selected
            if selected_team != 'All':
                st.subheader(f"📊 {selected_team} Statistics")
                
                # Home games
                home_games = df_filtered[df_filtered['txtechipa1'] == selected_team]
                # Away games
                away_games = df_filtered[df_filtered['txtechipa2'] == selected_team]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_xpts = home_games['xPTS_home'].sum() + away_games['xPTS_away'].sum()
                    st.metric("Total xPTS", f"{total_xpts:.1f}")
                
                with col2:
                    total_actual = home_games['actual_pts_home'].sum() + away_games['actual_pts_away'].sum()
                    st.metric("Total Actual Points", f"{total_actual}")
                
                with col3:
                    avg_xpts = (home_games['xPTS_home'].mean() + away_games['xPTS_away'].mean()) / 2
                    st.metric("Avg xPTS per Game", f"{avg_xpts:.2f}")
                
                with col4:
                    difference = total_actual - total_xpts
                    st.metric("Overperformance", f"{difference:+.1f}")
                
                # FORM ANALYSIS SECTION
                st.markdown("---")
                st.header(f"📈 {selected_team} - Form Analysis & Trends")
                
                # Calculate form data
                team_form = calculate_team_form(df_filtered, selected_team)
                
                if len(team_form) >= 5:
                    # Create tabs for different form analyses
                    form_tab1, form_tab2, form_tab3, form_tab4 = st.tabs([
                        "📊 Form Charts", 
                        "🔥 Recent Form", 
                        "📉 Rolling Averages",
                        "📋 Match History"
                    ])
                    
                    with form_tab1:
                        st.markdown("### Performance Trends")
                        
                        # Main form chart
                        form_fig = create_form_chart(team_form, selected_team)
                        st.plotly_chart(form_fig, use_container_width=True)
                        
                        # Momentum chart
                        momentum_fig = create_momentum_chart(team_form, selected_team)
                        st.plotly_chart(momentum_fig, use_container_width=True)
                    
                    with form_tab2:
                        st.markdown("### Recent Form Analysis")
                        
                        # Get last 15 games
                        recent_15 = team_form.tail(15)
                        
                        # Form indicators
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            form_5 = get_form_string(team_form, 5)
                            st.metric("Last 5 Games", form_5)
                        
                        with col2:
                            momentum = calculate_momentum(team_form, 5)
                            st.metric("Momentum (Last 5)", momentum)
                        
                        with col3:
                            recent_5_actual = recent_15.tail(5)['team_actual'].sum()
                            st.metric("Points (Last 5)", f"{recent_5_actual}")
                        
                        with col4:
                            recent_5_xpts = recent_15.tail(5)['team_xpts'].sum()
                            recent_5_diff = recent_5_actual - recent_5_xpts
                            st.metric("vs xPTS (Last 5)", f"{recent_5_diff:+.1f}")
                        
                        # Recent matches table
                        st.markdown("#### Last 15 Matches")
                        
                        recent_display = recent_15[[
                            'sezonul', 'etapa', 'venue', 'opponent', 
                            'team_actual', 'team_xpts'
                        ]].copy()
                        
                        recent_display.columns = ['Season', 'Round', 'Venue', 'Opponent', 'Pts', 'xPTS']
                        recent_display['Diff'] = recent_display['Pts'] - recent_display['xPTS']
                        recent_display['Result'] = recent_display['Pts'].map({3: '✅ Win', 1: '🟡 Draw', 0: '❌ Loss'})
                        
                        # Format numbers
                        recent_display['xPTS'] = recent_display['xPTS'].round(2)
                        recent_display['Diff'] = recent_display['Diff'].round(2)
                        
                        # Reverse order to show most recent first
                        recent_display = recent_display.iloc[::-1].reset_index(drop=True)
                        
                        # Apply styling
                        styled_recent = recent_display.style.background_gradient(
                            subset=['Diff'],
                            cmap='RdYlGn',
                            vmin=-3,
                            vmax=3
                        ).applymap(
                            lambda x: 'background-color: #d4edda' if x == '✅ Win' 
                            else ('background-color: #fff3cd' if x == '🟡 Draw' 
                            else 'background-color: #f8d7da'),
                            subset=['Result']
                        )
                        
                        st.dataframe(styled_recent, use_container_width=True, height=400)
                    
                    with form_tab3:
                        st.markdown("### Rolling Performance Metrics")
                        
                        # Display rolling averages for different windows
                        windows = [5, 10, 15]
                        
                        for window in windows:
                            if len(team_form) >= window:
                                st.markdown(f"#### Last {window} Games Average")
                                
                                last_n = team_form.tail(window)
                                
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    avg_actual = last_n['team_actual'].mean()
                                    st.metric(f"Avg Points", f"{avg_actual:.2f}")
                                
                                with col2:
                                    avg_xpts = last_n['team_xpts'].mean()
                                    st.metric(f"Avg xPTS", f"{avg_xpts:.2f}")
                                
                                with col3:
                                    diff = avg_actual - avg_xpts
                                    st.metric(f"Difference", f"{diff:+.2f}")
                                
                                with col4:
                                    wins = (last_n['team_actual'] == 3).sum()
                                    win_rate = (wins / window) * 100
                                    st.metric(f"Win Rate", f"{win_rate:.0f}%")
                        
                        # Trend comparison chart
                        st.markdown("#### Form Comparison Across Windows")
                        
                        trend_fig = go.Figure()
                        
                        for window in [5, 10, 15]:
                            trend_fig.add_trace(go.Scatter(
                                x=list(range(1, len(team_form) + 1)),
                                y=team_form[f'rolling_actual_{window}'],
                                mode='lines',
                                name=f'{window}-game avg (Actual)',
                                line=dict(width=2)
                            ))
                        
                        trend_fig.update_layout(
                            title="Rolling Average Comparison (Actual Points)",
                            xaxis_title="Match Number",
                            yaxis_title="Points Per Game",
                            height=400,
                            template='plotly_white',
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(trend_fig, use_container_width=True)
                    
                    with form_tab4:
                        st.markdown("### Complete Match History")
                        
                        # Full match history
                        history_display = team_form[[
                            'sezonul', 'etapa', 'venue', 'opponent',
                            'team_actual', 'team_xpts', 
                            'rolling_actual_10', 'rolling_xpts_10'
                        ]].copy()
                        
                        history_display.columns = [
                            'Season', 'Round', 'Venue', 'Opponent',
                            'Pts', 'xPTS', '10G Avg (Pts)', '10G Avg (xPTS)'
                        ]
                        
                        history_display['Diff'] = history_display['Pts'] - history_display['xPTS']
                        history_display['Form Diff'] = history_display['10G Avg (Pts)'] - history_display['10G Avg (xPTS)']
                        
                        # Format numbers
                        for col in ['xPTS', '10G Avg (Pts)', '10G Avg (xPTS)', 'Diff', 'Form Diff']:
                            history_display[col] = history_display[col].round(2)
                        
                        # Reverse to show most recent first
                        history_display = history_display.iloc[::-1].reset_index(drop=True)
                        
                        st.dataframe(history_display, use_container_width=True, height=500)
                        
                        # Download button for match history
                        csv_history = history_display.to_csv(index=False)
                        st.download_button(
                            label=f"Download {selected_team} Match History (CSV)",
                            data=csv_history,
                            file_name=f"{selected_team}_match_history.csv",
                            mime="text/csv"
                        )
                
                else:
                    st.info(f"Need at least 5 matches for form analysis. {selected_team} has {len(team_form)} match(es) in selected period.")
            
            # Display filtered data
            df_filtered_display = df_filtered[display_columns].copy()
            
            for col in prob_cols:
                df_filtered_display[col] = df_filtered_display[col].apply(lambda x: f"{x:.1%}")
            
            for col in xpts_cols:
                df_filtered_display[col] = df_filtered_display[col].apply(lambda x: f"{x:.2f}")
            
            st.dataframe(df_filtered_display, use_container_width=True, height=400)
        else:
            st.warning("No data matches the selected filters")
        
        # Download processed data
        st.header("💾 Download Processed Data")
        
        # Prepare download
        csv = df_processed.to_csv(index=False)
        st.download_button(
            label="Download Complete Dataset (CSV)",
            data=csv,
            file_name="xpts_processed_data.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Explanation section
        with st.expander("ℹ️ How xPTS is Calculated"):
            st.markdown("""
            ### Expected Points (xPTS) Calculation
            
            **Step 1: Calculate Implied Probabilities**
            - Home Win: 1 / Home Odds
            - Draw: 1 / Draw Odds
            - Away Win: 1 / Away Odds
            
            **Step 2: Remove Bookmaker's Margin**
            - Total = P(Home) + P(Draw) + P(Away)
            - Margin = Total - 1
            - True Probabilities = Implied Probability / Total
            
            **Step 3: Calculate xPTS**
            - xPTS Home = 3 × P(Home Win) + 1 × P(Draw)
            - xPTS Away = 3 × P(Away Win) + 1 × P(Draw)
            
            ### Column Definitions
            - **cotaa**: Home win odds (decimal)
            - **cotae**: Draw odds (decimal)
            - **cotad**: Away win odds (decimal)
            - **xPTS**: Expected points based on true probabilities
            - **Margin**: Bookmaker's overround (profit margin)
            """)
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.exception(e)

else:
    st.info("👆 Please upload an Excel file to get started")
    
    st.markdown("""
    ### Expected File Format
    
    The Excel file should contain the following columns:
    - **country**: League/country code
    - **sezonul**: Season
    - **etapa**: Round/matchday
    - **txtechipa1**: Home team name
    - **txtechipa2**: Away team name
    - **scor1**: Home team score
    - **scor2**: Away team score
    - **cotaa**: Home win odds (decimal)
    - **cotae**: Draw odds (decimal)
    - **cotad**: Away win odds (decimal)
    """)

# Footer
st.markdown("---")
st.markdown("*xPTS Calculator - Built for betting analytics*")
