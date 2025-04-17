import streamlit as st
import pandas as pd
import joblib

# Load the win probability model
model = joblib.load("win_probability_model.pkl")

def predict_win_probability(input_data):
    """Predicts the win probability based on input features."""
    prediction = model.predict_proba(input_data)[0][1]
    return prediction

def run_streamlit_app():
    """Runs the Streamlit app."""
    st.title("Win Predictor")

    # Input fields for features
    cum_runs = st.number_input("Cumulative Runs", min_value=0)
    cum_wickets = st.number_input("Cumulative Wickets", min_value=0, max_value=10)
    cum_balls = st.number_input("Cumulative Balls", min_value=0)
    runs_remaining = st.number_input("Runs Remaining", min_value=0)
    balls_remaining = st.number_input("Balls Remaining", min_value=0)
    crr = st.number_input("Current Run Rate", min_value=0.0)
    rrr = st.number_input("Required Run Rate", min_value=0.0)

    # Create input data as a DataFrame
    input_data = pd.DataFrame({
        "cum_runs": [cum_runs],
        "cum_wickets": [cum_wickets],
        "cum_balls": [cum_balls],
        "runs_remaining": [runs_remaining],
        "balls_remaining": [balls_remaining],
        "crr": [crr],
        "rrr": [rrr]
    })

    # Predict win probability
    if st.button("Predict"):
        win_probability = predict_win_probability(input_data)
        st.write(f"Win Probability: {win_probability:.2f}")

# Run the app
if __name__ == "__main__":
    run_streamlit_app()
