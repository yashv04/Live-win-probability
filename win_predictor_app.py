import streamlit as st


def run_streamlit_app():
    st.title("Live Win Probability Predictor")
    st.write("Estimate win probability for chasing team in real-time.")

    # User inputs
    cum_runs = st.number_input("Cumulative Runs Scored", min_value=0)
    cum_wickets = st.number_input("Wickets Lost", min_value=0, max_value=10)
    cum_balls = st.number_input("Balls Bowled", min_value=1, max_value=120)
    target_runs = st.number_input("Target Runs", min_value=1)
    target_overs = st.number_input("Target Overs", min_value=1, max_value=20, value=20)

    if cum_balls >= target_overs * 6:
        st.warning("Innings complete. No win probability to predict.")
        return

    runs_remaining = target_runs - cum_runs
    balls_remaining = (target_overs * 6) - cum_balls
    crr = cum_runs / (cum_balls / 6)
    rrr = runs_remaining / (balls_remaining / 6) if balls_remaining > 0 else 0

    input_df = pd.DataFrame({
        "cum_runs": [cum_runs],
        "cum_wickets": [cum_wickets],
        "cum_balls": [cum_balls],
        "runs_remaining": [runs_remaining],
        "balls_remaining": [balls_remaining],
        "crr": [crr],
        "rrr": [rrr]
    })

    model = joblib.load("win_probability_model.pkl")
    win_prob = model.predict_proba(input_df)[0][1]

    st.metric("Win Probability (%)", f"{win_prob * 100:.2f}%")