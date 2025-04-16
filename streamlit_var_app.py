import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# --- Load Data ---
sp500 = pd.read_csv("sp500_data_cleaned.csv", header=[0,1], index_col=0)
returns = sp500[('Log Returns', '^GSPC')]

# --- Sidebar Inputs ---
st.sidebar.title("VaR Settings")
confidence_level = st.sidebar.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01)
dist_choice = st.sidebar.selectbox("Distribution for Monte Carlo", ["Normal", "t-distribution"])
historical_window = st.sidebar.slider("Historical Window (days)", 100, len(returns), 500, step=50)
portfolio_value = st.sidebar.number_input("Portfolio Value ($)", min_value=1_000.0, value=100_000.0, step=1_000.0, format="%.2f")

# --- Common Parameters ---
percentile = (1 - confidence_level) * 100
mu = returns[-historical_window:].mean()
sigma = returns[-historical_window:].std()
z = norm.ppf(1 - confidence_level)

# --- VaR Calculations ---
historical_VaR = np.percentile(returns[-historical_window:], percentile)
parametric_VaR = mu + z * sigma

# Monte Carlo Simulation
df = 5
n_simulations = 10000
if dist_choice == "t-distribution":
    sim_returns = np.random.standard_t(df, n_simulations) * sigma / np.sqrt(df / (df - 2)) + mu
else:
    sim_returns = np.random.normal(mu, sigma, n_simulations)
monte_carlo_VaR = np.percentile(sim_returns, percentile)

# --- Plot: Histogram + All VaRs ---
st.subheader("Overlay of VaR Estimates")
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(returns[-historical_window:] * 100, bins=50, color='#d6eaf8', edgecolor='black', alpha=0.8)
ax.axvline(historical_VaR * 100, color='red', linestyle='--', linewidth=2, label=f"Historical VaR ({historical_VaR*100:.2f}%)")
ax.axvline(parametric_VaR * 100, color='green', linestyle='--', linewidth=2, label=f"Parametric VaR ({parametric_VaR*100:.2f}%)")
ax.axvline(monte_carlo_VaR * 100, color='blue', linestyle='--', linewidth=2, label=f"Monte Carlo VaR ({monte_carlo_VaR*100:.2f}%)")
ax.set_title("Overlay of 1-Day VaR Methods", fontsize=16, fontweight='bold')
ax.set_xlabel("Daily Log Return (%)")
ax.set_ylabel("Frequency")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.3)
st.pyplot(fig)

# --- Display VaR Table ---
st.subheader("Value at Risk Table")
var_methods = ["Historical", "Parametric", "Monte Carlo"]
var_percents = np.array([historical_VaR, parametric_VaR, monte_carlo_VaR]) * 100
var_amounts = np.array([historical_VaR, parametric_VaR, monte_carlo_VaR]) * portfolio_value

var_table = pd.DataFrame({
    "Method": var_methods,
    "VaR (%)": [f"{v:.2f}%" for v in var_percents],
    "VaR ($)": [f"${v:,.2f}" for v in var_amounts]
})

st.dataframe(
    var_table.style.set_properties(**{
        'background-color': '#f9f9f9',
        'color': '#222',
        'border-color': 'black',
        'font-size': '24px',          # Adjusted font size
        'text-align': 'center',      
        'vertical-align': 'middle',
        'padding': '15px',           # Added padding
        'height': '50px'             # Added height
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#d6eaf8'),
            ('color', '#222'),
            ('font-size', '26px'),    # Slightly larger header font
            ('text-align', 'center'),
            ('vertical-align', 'middle'),
            ('padding', '15px'),
            ('height', '50px')
        ]},
        {'selector': '', 'props': [   # Add border to table
            ('border', '2px solid black'),
        ]}
    ]),
    use_container_width=True,
    height=200  # Added fixed height to make cells larger
)