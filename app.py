import streamlit as st
import pandas as pd
import numpy as np

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
            selected_country = st.selectbox("Country", countries)
        
        with col2:
            seasons = ['All'] + sorted(df_processed['sezonul'].unique().tolist())
            selected_season = st.selectbox("Season", seasons)
        
        with col3:
            teams = ['All'] + sorted(
                set(df_processed['txtechipa1'].unique().tolist() + 
                    df_processed['txtechipa2'].unique().tolist())
            )
            selected_team = st.selectbox("Team", teams)
        
        # Apply filters
        df_filtered = df_processed.copy()
        
        if selected_country != 'All':
            df_filtered = df_filtered[df_filtered['country'] == selected_country]
        
        if selected_season != 'All':
            df_filtered = df_filtered[df_filtered['sezonul'] == selected_season]
        
        if selected_team != 'All':
            df_filtered = df_filtered[
                (df_filtered['txtechipa1'] == selected_team) | 
                (df_filtered['txtechipa2'] == selected_team)
            ]
        
        if len(df_filtered) > 0:
            st.write(f"**Filtered Results:** {len(df_filtered):,} fixtures")
            
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
