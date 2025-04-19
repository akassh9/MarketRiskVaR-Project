import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm, t, binom
import datetime

# --- Page Configuration ---
st.set_page_config(page_title="Backtest & Breach Analysis", layout="wide", page_icon="📈")
st.title("Rolling VaR Backtest & Breach Analysis")

# --- Sidebar: Risk Parameters & Stress Window ---
levels = [90.0, 95.0, 97.5, 99.0, 99.5]
pct = st.sidebar.select_slider(
    "Confidence Level",
    options=levels,
    value=99.0,
    format_func=lambda x: f"{x:.1f}%"
)
confidence = pct / 100
horizon = st.sidebar.selectbox("VaR Horizon (days)", [1, 5, 10], index=0)
historical_window = st.sidebar.slider("Rolling Window (days)", 250, 2000, 1000, step=50)
start_default = datetime.date(2020, 2, 20)
end_default = datetime.date(2020, 4, 30)
date_start, date_end = st.sidebar.date_input("Stress Window (for breach analysis)", [start_default, end_default])
show_es = st.sidebar.checkbox("Include ES in breach stats?", value=False)

# --- Retrieve Portfolio Weights ---
def get_default_weights():
    return {'AAPL':0.15, 'MSFT':0.15, 'JNJ':0.10, 'JPM':0.10,
            'XOM':0.10, 'NVDA':0.05, 'VTI':0.10, 'TLT':0.10,
            'GLD':0.05, 'VNQ':0.05, '^GSPC':0.05}
asset_list = list(get_default_weights().keys())
weights = st.session_state.get('weights', get_default_weights())

# --- Load Returns (cached) ---
@st.cache_data
def load_returns(weights_dict):
    frames = {}
    for sym in weights_dict:
        fname = sym.lstrip('^').lower() + '_data_cleaned.csv'
        df = pd.read_csv(fname, header=[0,1], index_col=0, parse_dates=True)
        frames[sym] = df[('Log Returns', sym)]
    return pd.DataFrame(frames)

returns_df = load_returns(weights)
portfolio_returns = (returns_df * pd.Series(weights)).sum(axis=1)

# --- Rolling VaR & ES Calculations ---
alpha = 1 - confidence
hist_var = portfolio_returns.rolling(window=historical_window, min_periods=1).quantile(alpha)
drift = portfolio_returns.rolling(window=historical_window, min_periods=1).mean()
vol = portfolio_returns.rolling(window=historical_window, min_periods=1).std()
z = norm.ppf(alpha)
param_var = drift + z * vol
df_t = 5
t_q = t.ppf(alpha, df_t)
mc_var = drift + t_q * vol / np.sqrt(df_t / (df_t - 2))
if show_es:
    def es_calc(x): return x[x <= np.percentile(x, alpha*100)].mean()
    hist_es = portfolio_returns.rolling(window=historical_window, min_periods=1).apply(es_calc)

# --- Define Stress Period Masks ---
idx = portfolio_returns.index
mask_pre = idx < pd.to_datetime(date_start)
mask_mid = (idx >= pd.to_datetime(date_start)) & (idx <= pd.to_datetime(date_end))
mask_post = idx > pd.to_datetime(date_end)

# --- Breach Rate Helper & Computation ---
def breach_rate(mask, series, var_series):
    return (series[mask] < var_series[mask]).sum() / mask.sum() if mask.sum() else np.nan

df_breach = pd.DataFrame({
    'Pre-COVID': [
        breach_rate(mask_pre, portfolio_returns, hist_var),
        breach_rate(mask_pre, portfolio_returns, param_var),
        breach_rate(mask_pre, portfolio_returns, mc_var)
    ],
    'COVID Window': [
        breach_rate(mask_mid, portfolio_returns, hist_var),
        breach_rate(mask_mid, portfolio_returns, param_var),
        breach_rate(mask_mid, portfolio_returns, mc_var)
    ],
    'Post-COVID': [
        breach_rate(mask_post, portfolio_returns, hist_var),
        breach_rate(mask_post, portfolio_returns, param_var),
        breach_rate(mask_post, portfolio_returns, mc_var)
    ]
}, index=['Historical VaR', 'Parametric VaR', 'Monte Carlo VaR'])
if show_es:
    df_breach['ES Breaches'] = [
        breach_rate(mask_pre, portfolio_returns, hist_es),
        breach_rate(mask_mid, portfolio_returns, hist_es),
        breach_rate(mask_post, portfolio_returns, hist_es)
    ]

# --- Display Breach Rate Table (Centered Text) ---
st.markdown(
    "### Breach Rates by Period\n\n"
    "A breach occurs when the portfolio's return is less than the VaR on that date. "
    "Breaches are shown as a percentage of the total number of observations in that period. "
    "As seen on the table during periods of high volatility (e.g. COVID), the breach rate is higher. "
    "A classic pitfall of VaR is that it can be breached multiple times in a row."
    "ES tries to address this by averaging the worst losses, but it can also be breached multiple times."
)
st.table(
    df_breach.style
        .format({col: "{:.2%}" for col in df_breach.columns})
        .set_properties(**{
            'text-align': 'center'
        })
)

# --- Dynamic Basel Traffic‑Light (uses the slider) ---
bt_days = historical_window

# slice the last bt_days for returns & VaR
returns_bt = portfolio_returns[-bt_days:]
var_bt     = hist_var[-bt_days:]

# count exceptions
exceptions = int((returns_bt < var_bt).sum())

# compute 95th & 99th percentile thresholds for a Binomial(bt_days, p)
p = 1 - confidence  # e.g. 0.01 for 99% VaR
green_max = int(binom.ppf(0.95, bt_days, p))
amber_max = int(binom.ppf(0.99, bt_days, p))

# classify
if exceptions <= green_max:
    zone_label, zone_color = "Green", "green"
elif exceptions <= amber_max:
    zone_label, zone_color = "Amber", "orange"
else:
    zone_label, zone_color = "Red",   "red"

# render
st.markdown("### Basel Traffic‑Light Status")
st.markdown(
    """
    Basel II/III accords require banks to hold capital for market risk based on VaR. Specifically, under Basel II’s internal models approach, banks compute a 10-day 99% VaR for their trading portfolios and backtest it daily. 
    Regulators classify the model’s performance using a “traffic light” system based on backtesting results: a model is in the green zone if it has 4 or fewer VaR exceptions in 250 days, yellow if 5–9 exceptions, and red if 10 or more exceptions.
    While our implementation here is simplified, it serves as a useful diagnostic tool for assessing the performance of the VaR model.
    """
)
st.markdown(
    f"<span style='color:{zone_color}; font-weight:bold;'>{zone_label} Zone</span> — "
    f"{exceptions} exceptions over last {bt_days} days  "
    f"(green ≤ {green_max}, amber ≤ {amber_max}, red > {amber_max})",
    unsafe_allow_html=True
)


st.markdown("### Rolling VaR Backtest")
st.markdown(
    f" This plot shows the portfolio's daily returns and its rolling VaR "
)
# --- Rolling VaR Plot with Stress Shading ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=idx, y=portfolio_returns, name='Daily Returns', line=dict(color='#4C72B0', width=1), opacity=0.7))
fig.add_trace(go.Scatter(x=idx, y=hist_var, name=f'Historical VaR ({confidence:.0%})', line=dict(color='#C44E52', width=2)))
fig.add_trace(go.Scatter(x=idx, y=param_var, name='Parametric VaR', line=dict(color='#55A868', width=2)))
fig.add_trace(go.Scatter(x=idx, y=mc_var, name='Monte Carlo VaR', line=dict(color='#8172B2', width=2)))
fig.add_vrect(x0=date_start, x1=date_end, fillcolor='grey', opacity=0.3, line_width=0, annotation_text='Stress Window', annotation_position='top left')
fig.update_layout(
    xaxis_title='Date', yaxis_title='Log Return',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)
st.plotly_chart(fig, use_container_width=True)

# --- Violation Severity During Stress Window ---
st.markdown("### Violation Severity During Stress Window")
st.markdown(
    f" This plot shows how much the portfolio's loss exceeded its Historical VaR during the selected stress window. "
    "The darker the red, the larger the breach."
    "This is to show the scale of the problem just using VaR as a risk measure poses."
    "Each bar shows how far the portfolio's loss exceeded its Historical VaR on that date. "
    "Darker reds indicate larger breaches."
)
period_returns = portfolio_returns[mask_mid]
period_var = hist_var[mask_mid]
severity = (-period_returns - period_var).clip(lower=0) * 100
fig2 = go.Figure(
    go.Bar(
        x=severity.index, y=severity.values,
        marker=dict(color=severity.values, colorscale='Reds', showscale=True),
        name='Severity (%)'
    )
)
fig2.update_layout(xaxis_tickformat='%b %Y', xaxis_title='Date', yaxis_title='Loss Beyond VaR (%)')
st.plotly_chart(fig2, use_container_width=True)

# --- Assumptions & Limitations ---
with st.expander("Assumptions & Limitations"):
    st.markdown("""
    - **Window choice:**  Traffic‑light thresholds are calibrated to a Binomial(bt_days, p) model; very small or large windows can dilute regulatory relevance.  
    - **Stress window selection:**  Manually chosen dates (e.g. COVID period) may not capture all stress events.  
    - **Sample independence:**  Assumes daily returns are iid; serial correlation or volatility clustering can bias breach counts.  
    - **Not capital calculation:**  This is back‑test diagnostics only, not a full Pillar 1 capital requirement.
    """)