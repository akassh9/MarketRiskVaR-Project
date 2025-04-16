import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t, skewnorm
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# --- Default Weights ---
default_weights = {
    'AAPL': 0.15,
    'MSFT': 0.15,
    'JNJ': 0.10,
    'JPM': 0.10,
    'XOM': 0.10,
    'NVDA': 0.05,
    'VTI': 0.10,
    'TLT': 0.10,
    'GLD': 0.05,
    'VNQ': 0.05,
    '^GSPC': 0.05
}
asset_list = list(default_weights.keys())

# --- Initialize session state for weights ---
if 'weights' not in st.session_state:
    st.session_state.weights = default_weights.copy()

# Initialize each slider key in session state if not already present
for asset in asset_list:
    key = f"weight_{asset}"
    if key not in st.session_state:
        st.session_state[key] = default_weights[asset]

# A helper that keeps the slider keys consistent with our dictionary
def get_weight_key(asset):
    return f"weight_{asset}"

# --- Callback: Redistribute Weights ---
def update_weights(changed_asset):
    changed_key = get_weight_key(changed_asset)
    new_val = st.session_state[changed_key]
    st.session_state.weights[changed_asset] = new_val

    remaining_weight = 1.0 - new_val
    other_assets = [a for a in asset_list if a != changed_asset]
    current_other_total = sum(st.session_state.weights[a] for a in other_assets)
    
    if current_other_total == 0:
        for a in other_assets:
            st.session_state.weights[a] = 0.0
            st.session_state[get_weight_key(a)] = 0.0
    else:
        for a in other_assets:
            proportion = st.session_state.weights[a] / current_other_total
            new_weight = proportion * remaining_weight
            st.session_state.weights[a] = new_weight
            st.session_state[get_weight_key(a)] = new_weight

# --- Sidebar: Reactive Slider Inputs ---
st.sidebar.title("VaR Settings")
st.sidebar.markdown("### Adjust Asset Weights\n(The sliders are reactive and will keep the total allocation equal to 100%)")
for asset in asset_list:
    st.sidebar.slider(
        label=f"{asset} Weight",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        key=get_weight_key(asset),
        on_change=update_weights,
        args=(asset,)
    )
    
# After the callbacks run, extract the reactive weights dictionary
weights = {asset: st.session_state[get_weight_key(asset)] for asset in asset_list}

# --- (Rest of your application code follows) ---

# Example: Loading data for individual assets (paths assumed correct)
aapl = pd.read_csv("apple_data_cleaned.csv", header=[0,1], index_col=0)
msft = pd.read_csv("msft_data_cleaned.csv", header=[0,1], index_col=0)
jnj = pd.read_csv("jnj_data_cleaned.csv", header=[0,1], index_col=0)
jpm = pd.read_csv("jpm_data_cleaned.csv", header=[0,1], index_col=0)
xom = pd.read_csv("xom_data_cleaned.csv", header=[0,1], index_col=0)
nvda = pd.read_csv("nvda_data_cleaned.csv", header=[0,1], index_col=0)
vti = pd.read_csv("vti_data_cleaned.csv", header=[0,1], index_col=0)
tlt = pd.read_csv("tlt_data_cleaned.csv", header=[0,1], index_col=0)
gld = pd.read_csv("gld_data_cleaned.csv", header=[0,1], index_col=0)
vnq = pd.read_csv("vnq_data_cleaned.csv", header=[0,1], index_col=0)
sp500 = pd.read_csv("sp500_data_cleaned.csv", header=[0,1], index_col=0)

# --- Extract returns and compute portfolio returns ---
aapl_returns = aapl[('Log Returns', 'AAPL')]
msft_returns = msft[('Log Returns', 'MSFT')]
jnj_returns = jnj[('Log Returns', 'JNJ')]
jpm_returns = jpm[('Log Returns', 'JPM')]
xom_returns = xom[('Log Returns', 'XOM')]
nvda_returns = nvda[('Log Returns', 'NVDA')]
vti_returns = vti[('Log Returns', 'VTI')]
tlt_returns = tlt[('Log Returns', 'TLT')]
gld_returns = gld[('Log Returns', 'GLD')]
vnq_returns = vnq[('Log Returns', 'VNQ')]
sp500_returns = sp500[('Log Returns', '^GSPC')]

portfolio_returns = pd.DataFrame({
    'AAPL': aapl_returns,
    'MSFT': msft_returns,
    'JNJ': jnj_returns,
    'JPM': jpm_returns,
    'XOM': xom_returns,
    'NVDA': nvda_returns,
    'VTI': vti_returns,
    'TLT': tlt_returns,
    'GLD': gld_returns,
    'VNQ': vnq_returns,
    '^GSPC': sp500_returns
})

returns = (portfolio_returns * weights).sum(axis=1)

# --- Other Sidebar Inputs ---
confidence_level = st.sidebar.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01)
historical_window = 1000  # Set a fixed historical window of 1000 days
portfolio_value = st.sidebar.number_input("Portfolio Value ($)", min_value=1_000.0, value=100_000.0, step=1_000.0, format="%.2f")

# --- VaR Calculations ---
percentile = (1 - confidence_level) * 100
mu = returns[-historical_window:].mean()
sigma = returns[-historical_window:].std()
z = norm.ppf(1 - confidence_level)

historical_VaR = np.percentile(returns[-historical_window:], percentile)
parametric_VaR = mu + z * sigma

# Monte Carlo Simulation
df = 5  # for t-distribution
n_simulations = 10000
sim_returns = np.random.standard_t(df, n_simulations) * sigma / np.sqrt(df / (df - 2)) + mu
monte_carlo_VaR = np.percentile(sim_returns, percentile)

# --- Plotting ---
st.subheader("Overlay of VaR Estimates")
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(returns[-historical_window:] * 100, bins=50, color='#d6eaf8', edgecolor='black', alpha=0.8)
ax.axvline(historical_VaR * 100, color='red', linestyle='--', linewidth=2, label=f"Historical VaR (1000-day, {historical_VaR*100:.2f}%)")
ax.axvline(parametric_VaR * 100, color='green', linestyle='--', linewidth=2, label=f"Parametric VaR ({parametric_VaR*100:.2f}%)")
ax.axvline(monte_carlo_VaR * 100, color='blue', linestyle='--', linewidth=2, label=f"Monte Carlo VaR (t-dist, {monte_carlo_VaR*100:.2f}%)")
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
        'font-size': '20px',
        'text-align': 'center',
        'vertical-align': 'middle',
        'padding': '8px',
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#d6eaf8'),
            ('color', '#222'),
            ('font-size', '22px'),
            ('text-align', 'center'),
            ('vertical-align', 'middle'),
            ('padding', '8px'),
        ]},
        {'selector': '', 'props': [
            ('border', '2px solid black'),
        ]}
    ]),
    use_container_width=True,
    height=150
)
