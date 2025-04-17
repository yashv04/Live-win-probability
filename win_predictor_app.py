import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and data
@st.cache_data
def load_model():
    with open("win_probability_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_data():
    deliveries = pd.read_csv("deliveries_sampled.csv")
    matches = pd.read_csv("matches_till_2024.csv")
    return deliveries, matches

model = load_model()
deliveries, matches = load_data()

# Extract unique team names from matches data
teams = sorted(matches['batting_team'].dropna().unique())

# UI - Streamlit App
def run_streamlit_app():
    st.title("Live Win Probability Predictor")

    st.sidebar.header("📋 Match Inputs")

    col1, col2 = st.columns(2)
    with col1:
        batting_team = st.selectbox("Select Batting Team", teams)
    with col2:
        bowling_team = st.selectbox("Select Bowling Team", [team for team in teams if team != batting_team])

    target = st.number_input("Target Score", min_value=1, value=150)
    score = st.number_input("Current Score", min_value=0, value=50)
    overs = st.number_input("Overs Completed", min_value=0.1, max_value=20.0, value=10.0, step=0.1)
    wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10, value=2)

    if st.button("Predict Win Probability"):
        # Feature engineering
        runs_left = target - score
        balls_left = 120 - int(overs * 6)
        wickets_left = 10 - wickets
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        # Prepare input for model
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_left],
            'target': [target],
            'current_run_rate': [crr],
            'required_run_rate': [rrr]
        })

        # Predict
        prediction = model.predict_proba(input_df)
        loss_prob = prediction[0][0]
        win_prob = prediction[0][1]

        st.markdown("### Win Probability")
        st.progress(win_prob)
        st.success(f"{batting_team} Win Probability: **{win_prob * 100:.2f}%**")
        st.error(f"{bowling_team} Win Probability: **{loss_prob * 100:.2f}%**")


# Run the app
if __name__ == "__main__":
    run_streamlit_app()
