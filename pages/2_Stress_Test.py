import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from scipy.stats import norm

# --- Page Configuration ---
st.set_page_config(page_title="Stress Test Scenarios", layout="wide", page_icon="⚡️")
st.title("LLM‑Generated Stress Test Scenarios")

# --- Retrieve Weights from Session State ---
def get_default_weights():
    return {
        'AAPL':0.15, 'MSFT':0.15, 'JNJ':0.10, 'JPM':0.10,
        'XOM':0.10, 'NVDA':0.05, 'VTI':0.10, 'TLT':0.10,
        'GLD':0.05, 'VNQ':0.05, '^GSPC':0.05
    }
weights = st.session_state.get('weights', get_default_weights())
portfolio_value = st.sidebar.number_input("Portfolio Value (USD)", min_value=1_000.0, value=100_000.0, step=1_000.0)

# --- Load Pre‑Built Scenarios ---
with open("stress_scenarios.json", "r") as f:
    scenarios = json.load(f)
names = [s['name'] for s in scenarios]

# --- Sidebar: Select Scenario & Multiplier ---
st.sidebar.markdown("### Pick a Stress Scenario")
choice = st.sidebar.selectbox("Scenario", names)
multiplier = st.sidebar.slider("Intensity Multiplier", 0.5, 2.0, 1.0, step=0.1)

# --- Scenario Metadata ---
scenario = next(s for s in scenarios if s['name'] == choice)
st.markdown(f"#### {scenario['name']}")
st.markdown(scenario.get('description', 'No description available.'))
st.caption(f"*Generated on: {scenario.get('date_generated','N/A')}*")

# --- Compute Shock Impact ---
shock_pct = pd.Series(scenario['shocks']) * multiplier
shock_pct.index.name = 'Asset'

# Compute P&L
pnl_pct = shock_pct * pd.Series(weights)
pnl_usd = pnl_pct * portfolio_value

df = pd.DataFrame({
    'Shock (%)': shock_pct.round(2),
    'P&L (%)': pnl_pct.round(3) * 100,
    'P&L (USD)': pnl_usd.round(2)
}).reset_index().rename(columns={'index':'Asset'})

# --- Display P&L Table & Download ---
st.markdown("### Portfolio P&L Under Scenario")
st.table(df)
st.download_button("Download Scenario P&L", df.to_csv(index=False), file_name="scenario_pnl.csv")

# --- Waterfall Chart of P&L ---
st.markdown("### Waterfall of P&L by Asset")
waterfall = go.Figure()
waterfall.add_trace(go.Bar(
    x=df['Asset'], y=df['P&L (USD)'],
    marker_color=df['P&L (USD)'],
    name='P&L'
))
waterfall.update_layout(
    xaxis_title='Asset', yaxis_title='P&L (USD)',
    showlegend=False
)
st.plotly_chart(waterfall, use_container_width=True)

# --- Recompute VaR & ES Including Shock ---
@st.cache_data
def load_returns(weights_dict):
    frames = {}
    for sym in weights_dict:
        df = pd.read_csv(
            sym.lstrip('^').lower() + '_data_cleaned.csv',
            header=[0,1], index_col=0, parse_dates=True
        )
        frames[sym] = df[('Log Returns', sym)]
    return pd.DataFrame(frames)

returns_df = load_returns(weights)
portfolio_returns = (returns_df * pd.Series(weights)).sum(axis=1)

# Sidebar inputs for VaR/ES
confidence = st.sidebar.slider("Recompute Confidence Level", 0.90, 0.995, 0.99, step=0.005)
horizon = st.sidebar.selectbox("Recompute Horizon (days)", [1,5,10], index=0)
historical_window = st.sidebar.slider("Recompute Window (days)", 250, 2000, 1000, step=50)

# Append shock as newest return using .loc
shocked_returns = portfolio_returns.copy()
latest_index  = returns_df.index.max()
shock_value   = shock_pct.dot(pd.Series(weights))
shocked_returns.loc[latest_index] = shock_value

# Compute window, VaR and ES
window  = shocked_returns[-historical_window:]
alpha   = 1 - confidence
var_1d  = np.percentile(window, alpha * 100)
var_h   = var_1d * np.sqrt(horizon)
tail    = window[window <= var_1d]
es_1d   = tail.mean()
es_h    = es_1d * np.sqrt(horizon)

# --- Display Recomputed Metrics ---
st.markdown("### VaR & ES Under Scenario")
col1, col2 = st.columns(2)
col1.metric(f"{horizon}-day VaR", f"{var_h*100:.2f}%")
col2.metric(f"{horizon}-day ES", f"{es_h*100:.2f}%")
