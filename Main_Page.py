import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

# --- Page Configuration & Theme ---
st.set_page_config(page_title="Visualizing Risk", layout="wide", page_icon="📊")

# --- Section: Introduction ---
st.markdown(
    """
    ## Portfolio Risk Overview
    This interactive dashboard illustrates **Value at Risk (VaR)** and **Expected Shortfall (ES)** for a customizable multi-asset portfolio. Recruiters can inspect both the quantitative results and the underlying code snippets (expand the sections below).
    """
)

# --- Default Portfolio Weights & Session State ---
# We use session state to persist slider values and ensure total allocation = 100%
def get_default_weights():
    return {
        'AAPL': 0.15, 'MSFT': 0.15, 'JNJ': 0.10, 'JPM': 0.10,
        'XOM': 0.10, 'NVDA': 0.05, 'VTI': 0.10, 'TLT': 0.10,
        'GLD': 0.05, 'VNQ': 0.05, '^GSPC': 0.05
    }
asset_list = list(get_default_weights().keys())

# Initialize weights dict if missing
st.session_state.setdefault('weights', get_default_weights().copy())
# Initialize each slider key and sync back to weights dict
for asset, default_w in get_default_weights().items():
    key = f"weight_{asset}"
    st.session_state.setdefault(key, default_w)
    # Ensure main weights dict stays aligned
    st.session_state['weights'][asset] = st.session_state[key]

# Callback: redistribute weights when one slider changes
def update_weights(changed_asset):
    changed_key = f"weight_{changed_asset}"
    new_val = st.session_state[changed_key]
    # Update main dict
    st.session_state['weights'][changed_asset] = new_val

    remaining = 1 - new_val
    others = [a for a in asset_list if a != changed_asset]
    total_others = sum(st.session_state['weights'][a] for a in others)
    if total_others == 0:
        for a in others:
            st.session_state['weights'][a] = 0.0
            st.session_state[f"weight_{a}"] = 0.0
    else:
        for a in others:
            prop = st.session_state['weights'][a] / total_others
            w = prop * remaining
            st.session_state['weights'][a] = w
            st.session_state[f"weight_{a}"] = w

# --- Sidebar: Controls ---
st.sidebar.title("VaR Settings")
st.sidebar.markdown("#### 1) Adjust Asset Weights (sum must be 100%)")
for asset in asset_list:
    st.sidebar.slider(
        label=f"{asset}", min_value=0.0, max_value=1.0, step=0.01,
        key=f"weight_{asset}", on_change=update_weights, args=(asset,)
    )

st.sidebar.markdown("#### 2) Risk Parameters")
confidence = st.sidebar.slider("Confidence Level", 0.90, 0.995, 0.99, step=0.005)
horizon = st.sidebar.selectbox("VaR Horizon (days)", [1, 5, 10], index=0)
historical_window = st.sidebar.slider("Historical Window (days)", 250, 2000, 1000, step=50)
show_es = st.sidebar.checkbox("Display Expected Shortfall (ES)?", value=True)
portfolio_value = st.sidebar.number_input(
    "Portfolio Value (USD)", min_value=1_000.0, value=100_000.0, step=1_000.0, format="%.2f"
)

# Extract final weights dictionary
weights = st.session_state['weights'].copy()

# --- Data Loading Function (cached) ---
with st.expander("Show data loading code snippet"):  
    st.code(
        '''@st.cache_data
def load_returns(weights_dict):
    """Load cleaned log-return CSVs and return DataFrame."""
    dfs = {}
    for sym in weights_dict:
        fname = sym.lstrip('^').lower() + '_data_cleaned.csv'
        df = pd.read_csv(fname, header=[0,1], index_col=0, parse_dates=True)
        dfs[sym] = df[('Log Returns', sym)]
    return pd.DataFrame(dfs)''', language='python'
    )

@st.cache_data
def load_returns(weights_dict):
    dfs = {}
    for sym in weights_dict:
        fname = sym.lstrip('^').lower() + '_data_cleaned.csv'
        df = pd.read_csv(fname, header=[0,1], index_col=0, parse_dates=True)
        dfs[sym] = df[('Log Returns', sym)]
    return pd.DataFrame(dfs)

returns_df = load_returns(weights)
portfolio_returns = (returns_df * pd.Series(weights)).sum(axis=1)

# --- VaR & ES Computations ---
alpha = 1 - confidence
windowed = portfolio_returns[-historical_window:]
# Historical VaR 1-day
hist_var_1d = np.percentile(windowed, alpha * 100)
# Scale VaR to multi-day horizon (square-root-of-time)
tot_var = hist_var_1d * np.sqrt(horizon)
# Expected Shortfall (ES)
es_1d = windowed[windowed <= hist_var_1d].mean()
es = es_1d * np.sqrt(horizon)

# --- Metrics Display ---
st.markdown("### Key Risk Metrics")
col1, col2 = st.columns(2)
col1.metric(f"{horizon}-day VaR ({confidence:.1%})", f"{tot_var*100:.2f}%")
if show_es:
    col2.metric(f"{horizon}-day ES", f"{es*100:.2f}%")

# --- Histogram Visualization ---
st.markdown("### Return Distribution with VaR & ES")
fig = go.Figure()
fig.add_trace(go.Histogram(x=windowed * 100, nbinsx=50, opacity=0.75, name="Returns"))
fig.add_vline(x=tot_var * 100, line=dict(color="red", dash="dash"), annotation_text="VaR", annotation_position="top right")
if show_es:
    fig.add_vline(x=es * 100, line=dict(color="purple", dash="dash"), annotation_text="ES", annotation_position="top right")
fig.update_layout(xaxis_title="Daily Return (%)", yaxis_title="Frequency", bargap=0.2)
st.plotly_chart(fig, use_container_width=True)

with st.expander("Show risk calculation snippet"):
    st.code(
        '''# Historical VaR
hist_var_1d = np.percentile(windowed, alpha * 100)
# Scale to horizon
tot_var = hist_var_1d * np.sqrt(horizon)
# ES: average loss beyond VaR
es_1d = windowed[windowed <= hist_var_1d].mean()
es = es_1d * np.sqrt(horizon)''', language='python'
    )

# --- Metrics Table & Export ---
st.markdown("### Summary Table & Download")
metrics = ["VaR"] + (["ES"] if show_es else [])
values_pct = [tot_var] + ([es] if show_es else [])
values_usd = [tot_var * portfolio_value] + ([es * portfolio_value] if show_es else [])
df = pd.DataFrame({"Metric": metrics,
                   "Value (%)": [f"{v*100:.2f}%" for v in values_pct],
                   "Value (USD)": [f"${v:,.2f}" for v in values_usd]})
st.table(df)
st.download_button("Download Metrics as CSV", df.to_csv(index=False), file_name="var_es_metrics.csv")

# --- Explanatory Caption ---
st.caption(
    f"These results are based on the most recent {historical_window} days of returns. "
    f"VaR and ES are scaled to a {horizon}-day horizon via the square-root-of-time rule, as commonly used in practice."
)
