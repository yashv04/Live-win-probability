import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('win_probability_model.pkl')

# App title
st.title('Win Probability Predictor')

# Input features
cum_runs = st.number_input('Cumulative Runs', min_value=0)
cum_wickets = st.number_input('Cumulative Wickets', min_value=0, max_value=10)
cum_balls = st.number_input('Cumulative Balls', min_value=0)
runs_remaining = st.number_input('Runs Remaining', min_value=0)
balls_remaining = st.number_input('Balls Remaining', min_value=0)
crr = st.number_input('Current Run Rate')
rrr = st.number_input('Required Run Rate')

# Create a dataframe for prediction
input_data = pd.DataFrame({
    'cum_runs': [cum_runs],
    'cum_wickets': [cum_wickets],
    'cum_balls': [cum_balls],
    'runs_remaining': [runs_remaining],
    'balls_remaining': [balls_remaining],
    'crr': [crr],
    'rrr': [rrr]
})

# Make prediction
if st.button('Predict'):
    prediction = model.predict_proba(input_data)[0][1]
    st.write(f'The probability of the batting team winning is: {prediction:.2f}')


# Run the app
if __name__ == "__main__":
    run_streamlit_app() 
