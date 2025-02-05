import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Load pickled ARIMA model
@st.cache_resource
def load_model(model_name):
    with open(model_name, "rb") as file:
        model = pickle.load(file)
    return model

# Initialize app
st.title("Cost Forecast (ARIMA) Viewer")

st.write("""
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
""")
# Model selection
model_names = ["arima_cost.pkl", "arima_cost_prod.pkl"]
selected_model = st.sidebar.selectbox("Select Model", model_names)
# Load model
# Load selected model
model = load_model(selected_model)

st.sidebar.header("Prediction Parameters")

# Input start and end dates
start_date = st.sidebar.date_input("Select Start Date", value=datetime(2025, 1, 31), disabled=True)
end_date = st.sidebar.date_input("Select End Date", value=datetime.today())

if start_date > end_date:
    st.error("End date must be after start date.")
else:
    n_days = (end_date - start_date).days + 1

    # Generate forecast if valid input
    if st.sidebar.button("Generate Forecast"):
        try:
            forecast = model.forecast(steps=n_days)
            forecast_dates = pd.date_range(start=start_date, periods=n_days)
            
            # Display results
            results_df = pd.DataFrame({"Date": forecast_dates, "Forecast": forecast})
            # Add confidence intervals
            conf_int = model.get_forecast(steps=n_days).conf_int()
            results_df["Lower CI"] = conf_int.iloc[:, 0].values
            results_df["Upper CI"] = conf_int.iloc[:, 1].values
            results_df["Lower CI"] = results_df["Lower CI"].apply(lambda x: max(x, 0))

            st.write("### Forecast Results")
            st.dataframe(results_df)
           
            # Plot with confidence intervals
            st.write("### Forecast with Confidence Intervals")
            st.line_chart(results_df.set_index("Date")[["Forecast", "Lower CI", "Upper CI"]])

        except Exception as e:
            st.error(f"Error generating forecast: {e}")
