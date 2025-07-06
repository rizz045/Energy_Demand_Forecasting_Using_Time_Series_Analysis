import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

# --- SETTINGS ---
st.set_page_config(page_title="SARIMA Forecast", layout="wide")
st.title("📊 SARIMA Time Series Forecasting App")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv("PJMW_daily.csv", parse_dates=["Datetime"], index_col="Datetime")
    df = df.asfreq('D')  # Ensure daily frequency for SARIMA
    return df

df = load_data()
series = df["PJMW_MW"]

# --- LOAD PRE-TRAINED SARIMA MODEL ---
model: SARIMAXResults = joblib.load("PJM_Model.joblib")

# --- USER INPUT ---
st.sidebar.header("🔧 Forecast Settings")
n_periods = st.sidebar.slider("Forecast Steps (Days)", min_value=1, max_value=60, value=14)
last_n_days = st.sidebar.slider("Plot Last N Days", min_value=7, max_value=180, value=30)

# --- FORECAST ---
forecast_obj = model.get_forecast(steps=n_periods)
forecast_series = forecast_obj.predicted_mean

# Manually set forecast index to proper date range
forecast_series.index = pd.date_range(
    start=series.index[-1] + pd.Timedelta(days=1), periods=n_periods, freq='D'
)

# --- COMBINE FOR SMOOTH PLOT ---
plot_series = series[-last_n_days:]
combined_series = pd.concat([plot_series, forecast_series])

# --- PLOT ---
st.subheader("🖼️ Forecast Visualization")
fig, ax = plt.subplots(figsize=(14, 6))

ax.plot(plot_series, label="Historical", color="skyblue", linewidth=2)
ax.plot(forecast_series, label="Forecast", color="crimson", marker='o', linestyle='--')

ax.set_title(f"SARIMA Forecast ({last_n_days} Days History + {n_periods} Day Forecast)", fontsize=16)
ax.set_xlabel("Date")
ax.set_ylabel("MW")
ax.legend()
ax.grid(True)

# Format X-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
fig.autofmt_xdate()

st.pyplot(fig)

# --- DISPLAY FORECAST TABLE ---
st.subheader("🔢 Forecasted Values")
st.dataframe(
    forecast_series.round(2).reset_index().rename(columns={"index": "Date", 0: "Forecast (MW)"})
)
