import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
