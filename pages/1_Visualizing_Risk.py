import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm, t

# Page setup
st.set_page_config(page_title="Visualizing Risk", layout="wide")
st.title("Visualizing Risk")

# 1. Portfolio weights (default or user-set)
def get_default_weights():
    return {
        'AAPL': 0.15, 'MSFT': 0.15, 'JNJ': 0.10, 'JPM': 0.10,
        'XOM': 0.10, 'NVDA': 0.05, 'VTI': 0.10, 'TLT': 0.10,
        'GLD': 0.05, 'VNQ': 0.05, '^GSPC': 0.05
    }
weights = st.session_state.get('weights', get_default_weights())

# 2. Load log-returns, parse dates
@st.cache_data
def load_returns(weights):
    dfs = {}
    for sym in weights:
        fname = sym.lstrip('^').lower() + '_data_cleaned.csv'
        df = pd.read_csv(
            fname,
            header=[0,1], index_col=0,
            parse_dates=True, infer_datetime_format=True
        )
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[~df.index.isna()]
        dfs[sym] = df[('Log Returns', sym)]
    data = pd.DataFrame(dfs)
    data.index.name = 'Date'
    return data

returns_df = load_returns(weights)
returns_df.index = pd.to_datetime(returns_df.index)

# 3. Compute daily portfolio returns
portfolio_returns = (returns_df * pd.Series(weights)).sum(axis=1)

# 4. VaR parameters
confidence = 0.99
alpha = 1 - confidence
window = 1000

# 5. Rolling VaR calculations
hist_var = portfolio_returns.rolling(window=window, min_periods=1).quantile(alpha)
mu = portfolio_returns.rolling(window=window, min_periods=1).mean()
sigma = portfolio_returns.rolling(window=window, min_periods=1).std()
z = norm.ppf(alpha)
param_var = mu + z * sigma
df_t = 5
t_q = t.ppf(alpha, df_t)
mc_var = mu + t_q * sigma / np.sqrt(df_t / (df_t - 2))

# 6. Define COVID window and masks
date_start = pd.to_datetime('2020-02-20')
date_end = pd.to_datetime('2020-04-30')
mask_pre = portfolio_returns.index < date_start
mask_mid = (portfolio_returns.index >= date_start) & (portfolio_returns.index <= date_end)
mask_post = portfolio_returns.index > date_end

# 7. Breach rates calculation helper
def calc_rate(mask, series, var_series):
    days = mask.sum()
    breaches = (series[mask] < var_series[mask]).sum()
    return breaches / days if days else 0

# 8. Compute breach rates per period
hist_pre = calc_rate(mask_pre, portfolio_returns, hist_var)
param_pre = calc_rate(mask_pre, portfolio_returns, param_var)
mc_pre = calc_rate(mask_pre, portfolio_returns, mc_var)
hist_mid = calc_rate(mask_mid, portfolio_returns, hist_var)
param_mid = calc_rate(mask_mid, portfolio_returns, param_var)
mc_mid = calc_rate(mask_mid, portfolio_returns, mc_var)
hist_post = calc_rate(mask_post, portfolio_returns, hist_var)
param_post = calc_rate(mask_post, portfolio_returns, param_var)
mc_post = calc_rate(mask_post, portfolio_returns, mc_var)

# 9. Plot: Rolling VaR with COVID shading
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(portfolio_returns.index, portfolio_returns, color='#4C72B0', alpha=0.7, lw=1, label='Daily Returns')
ax.plot(hist_var.index, hist_var, color='#C44E52', lw=2, label='Historical VaR (99%, 1000d)')
ax.plot(param_var.index, param_var, color='#55A868', lw=2, label='Parametric VaR')
ax.plot(mc_var.index, mc_var, color='#8172B2', lw=2, label='Monte Carlo VaR')
ax.axvspan(date_start, date_end, color='gray', alpha=0.3, label='COVID Crash')
ax.set_title('Rolling VaR Backtest with COVID Window', fontsize=18)
ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Daily Log Return', fontsize=14)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.legend(loc='upper left', fontsize=12)
st.pyplot(fig)

# 10. Breach Rate Table
table_html = f"""
<div style="background-color:#dbe2ef; color:#222; padding:20px; border-radius:8px; margin-top:20px; width:80%; border:1px solid #ccc;">
  <h3 style="font-size:20px; color:#222;">Breach Rate by Period (%)</h3>
  <table style="width:100%; font-size:16px; border-collapse:collapse;">
    <thead>
      <tr style="background-color:#a3bffa; color:#222;">
        <th style="padding:8px; text-align:left;">Period</th>
        <th style="padding:8px; text-align:right;">Historical</th>
        <th style="padding:8px; text-align:right;">Parametric</th>
        <th style="padding:8px; text-align:right;">Monte Carlo</th>
      </tr>
    </thead>
    <tbody>
      <tr style="background-color:#eef4fd;">
        <td style="padding:8px;">Pre‑COVID</td>
        <td style="padding:8px; text-align:right;">{hist_pre:.2%}</td>
        <td style="padding:8px; text-align:right;">{param_pre:.2%}</td>
        <td style="padding:8px; text-align:right;">{mc_pre:.2%}</td>
      </tr>
      <tr style="background-color:#dbe2ef;">
        <td style="padding:8px;">COVID Window</td>
        <td style="padding:8px; text-align:right;">{hist_mid:.2%}</td>
        <td style="padding:8px; text-align:right;">{param_mid:.2%}</td>
        <td style="padding:8px; text-align:right;">{mc_mid:.2%}</td>
      </tr>
      <tr style="background-color:#eef4fd;">
        <td style="padding:8px;">Post‑COVID</td>
        <td style="padding:8px; text-align:right;">{hist_post:.2%}</td>
        <td style="padding:8px; text-align:right;">{param_post:.2%}</td>
        <td style="padding:8px; text-align:right;">{mc_post:.2%}</td>
      </tr>
    </tbody>
  </table>
</div>
"""
st.markdown(table_html, unsafe_allow_html=True)

# 11. Violation Severity Plot
st.subheader("Violation Severity During COVID Window")
# Calculate severity: amount by which loss exceeded historical VaR
period_returns = portfolio_returns[mask_mid]
hist_cutoff = hist_var[mask_mid]
severity = (-period_returns - hist_cutoff).clip(lower=0)

# Convert to percent
severity_pct = severity * 100

fig3, ax3 = plt.subplots(figsize=(14, 6))

# Choose a colormap so bigger bars stand out
cmap = plt.get_cmap('Reds')
# Normalize colour by severity
norm = plt.Normalize(severity_pct.min(), severity_pct.max())
colors = cmap(norm(severity_pct.values))

bars = ax3.bar(
    severity_pct.index, 
    severity_pct.values, 
    width=2, 
    color=colors, 
    alpha=0.8
)

# Annotate bars with their values
for bar in bars:
    h = bar.get_height()
    if h > 0:
        ax3.text(
            bar.get_x() + bar.get_width()/2, 
            h + 0.2, 
            f"{h:.1f}%", 
            ha='center', 
            va='bottom', 
            fontsize=9
        )

ax3.set_title('Severity of VaR Violations (Historical, 99%)', fontsize=18)
ax3.set_xlabel('Date', fontsize=14)
ax3.set_ylabel('Loss Beyond VaR (%)', fontsize=14)

# Tighter y‑axis so labels don’t crowd
ax3.set_ylim(0, severity_pct.max() * 1.1)

ax3.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[2,3,4]))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax3.tick_params(axis='x', rotation=45, labelsize=12)
ax3.tick_params(axis='y', labelsize=12)

# Add a light horizontal grid for reference
ax3.yaxis.grid(True, linestyle='--', alpha=0.4)

# A small caption below the chart
plt.tight_layout()
st.pyplot(fig3)

st.markdown(
    "_Each bar shows how far the portfolio’s loss exceeded its 1‑day 99% historical VaR on that date. "
    "Bars are annotated in % points, and deeper reds indicate larger breaches._"
)