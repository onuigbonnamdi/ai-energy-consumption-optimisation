import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Energy Optimiser",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a3a2a, #2d7a4f);
        border-radius: 12px;
        padding: 16px 20px;
        color: white;
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-card h2 { font-size: 2rem; margin: 4px 0; }
    .metric-card p  { margin: 0; font-size: 0.82rem; opacity: 0.85; }
    .result-box {
        border-radius: 12px;
        padding: 20px 24px;
        color: white;
        font-size: 1.1rem;
        margin-top: 16px;
    }
    .normal   { background: linear-gradient(135deg, #1a5c2a, #27ae60); }
    .monitor  { background: linear-gradient(135deg, #7a5a00, #f39c12); }
    .reduce   { background: linear-gradient(135deg, #7a1a1a, #e74c3c); }
    .rec-icon { font-size: 2.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Synthetic training data (mirrors UCI household power stats) ───────────────
@st.cache_resource(show_spinner=False)
def build_model():
    """
    Train a Random Forest on synthetic household power data.
    Features: hour, day_of_week, month, lag_1, lag_2, lag_3
    Target: next 15-min active power (kW)
    """
    rng = np.random.default_rng(42)
    n = 50_000

    hours   = rng.integers(0, 24, n)
    days    = rng.integers(0, 7, n)
    months  = rng.integers(1, 13, n)

    # Realistic household power pattern
    base    = 0.5 + 0.4 * np.sin(np.pi * hours / 12)          # daily cycle
    weekend = np.where(days >= 5, 0.15, 0.0)                   # weekend boost
    season  = 0.1 * np.sin(2 * np.pi * (months - 1) / 12)     # seasonal
    noise   = rng.normal(0, 0.08, n)
    power   = np.clip(base + weekend + season + noise, 0.05, 3.5)

    lag1 = np.roll(power, 1)
    lag2 = np.roll(power, 2)
    lag3 = np.roll(power, 3)

    X = np.column_stack([hours, days, months, lag1, lag2, lag3])
    y = power

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=100, max_depth=12, n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "MAE":  mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R²":   r2_score(y_test, y_pred),
    }
    return model, X_test, y_test, y_pred, metrics


def get_recommendation(kw):
    if kw < 0.5:
        return "Normal", "normal", "✅", "Energy usage is low. No action required."
    elif kw < 1.2:
        return "Monitor", "monitor", "⚠️", "Moderate usage detected. Monitor connected appliances."
    else:
        return "Reduce Load", "reduce", "🔴", "High consumption detected. Consider switching off non-essential appliances."


# ── Build model ───────────────────────────────────────────────────────────────
with st.spinner("Initialising AI model…"):
    model, X_test, y_test, y_pred, metrics = build_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/lightning-bolt.png", width=72)
    st.title("⚡ Energy Optimiser")
    st.markdown("---")

    st.subheader("🕐 Current Time Context")
    hour  = st.slider("Hour of day", 0, 23, 14,
                      help="0 = midnight, 12 = noon, 23 = 11 pm")
    day   = st.selectbox("Day of week",
                         ["Monday","Tuesday","Wednesday","Thursday",
                          "Friday","Saturday","Sunday"])
    month = st.selectbox("Month",
                         ["January","February","March","April","May","June",
                          "July","August","September","October","November","December"])

    st.markdown("---")
    st.subheader("🔌 Recent Power Readings (kW)")
    st.caption("Enter your last 3 meter readings (every 15 min). "
               "Typical household: 0.1 – 3.5 kW")
    lag1 = st.number_input("15 min ago",  min_value=0.0, max_value=10.0, value=0.80, step=0.05)
    lag2 = st.number_input("30 min ago",  min_value=0.0, max_value=10.0, value=0.75, step=0.05)
    lag3 = st.number_input("45 min ago",  min_value=0.0, max_value=10.0, value=0.70, step=0.05)

    predict_btn = st.button("⚡ Predict & Optimise", use_container_width=True, type="primary")

    st.markdown("---")
    st.caption("**Author:** Nnamdi Onuigbo  \nAI Systems Engineer | SmartFlow Systems")

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("⚡ AI Energy Consumption Predictor")
st.markdown(
    "Enter your **current time context** and **recent power readings** in the sidebar, "
    "then click **Predict & Optimise** to get your next 15-minute energy forecast and "
    "load management recommendation."
)
st.markdown("---")

# Model performance metrics
col1, col2, col3, col4 = st.columns(4)
kv = [
    ("MAE",  f"{metrics['MAE']:.4f} kW",  "Mean Absolute Error"),
    ("RMSE", f"{metrics['RMSE']:.4f} kW", "Root Mean Squared Error"),
    ("R²",   f"{metrics['R²']:.4f}",       "Variance Explained"),
    ("Model","Random Forest","100 estimators"),
]
for col, (title, val, sub) in zip([col1, col2, col3, col4], kv):
    col.markdown(
        f'<div class="metric-card"><p>{title}</p><h2>{val}</h2><p>{sub}</p></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["⚡ Prediction", "📊 Model Performance", "📈 Daily Profile"])

# ── Tab 1: Prediction ─────────────────────────────────────────────────────────
with tab1:
    if predict_btn or True:   # show panel on load with defaults
        day_num   = ["Monday","Tuesday","Wednesday","Thursday",
                     "Friday","Saturday","Sunday"].index(day)
        month_num = ["January","February","March","April","May","June",
                     "July","August","September","October","November","December"].index(month) + 1

        features = np.array([[hour, day_num, month_num, lag1, lag2, lag3]])
        predicted_kw = float(model.predict(features)[0])
        predicted_kw = max(0.0, predicted_kw)

        label, css_class, icon, advice = get_recommendation(predicted_kw)

        lcol, rcol = st.columns([1, 1])

        with lcol:
            st.subheader("🔮 Forecast Result")
            st.markdown(
                f'<div class="result-box {css_class}">'
                f'<div class="rec-icon">{icon}</div>'
                f'<h2 style="margin:8px 0">{predicted_kw:.3f} kW</h2>'
                f'<b>Predicted next 15-min consumption</b><br><br>'
                f'<b>Status: {label}</b><br>'
                f'{advice}'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown("#### 📋 Input Summary")
            summary = pd.DataFrame({
                "Parameter": ["Hour", "Day", "Month", "Lag 1 (15 min ago)",
                               "Lag 2 (30 min ago)", "Lag 3 (45 min ago)"],
                "Value":     [f"{hour}:00", day, month,
                               f"{lag1} kW", f"{lag2} kW", f"{lag3} kW"],
            })
            st.dataframe(summary, hide_index=True, use_container_width=True)

        with rcol:
            st.subheader("📊 Usage Thresholds")
            fig, ax = plt.subplots(figsize=(6, 4))
            categories = ["Low\n(< 0.5 kW)", "Moderate\n(0.5–1.2 kW)", "High\n(> 1.2 kW)"]
            vals = [0.5, 0.7, 1.0]
            colours = ["#27ae60", "#f39c12", "#e74c3c"]
            bars = ax.bar(categories, vals, color=colours, width=0.5, edgecolor="white")
            ax.axhline(predicted_kw, color="white", linestyle="--",
                       linewidth=2, label=f"Prediction: {predicted_kw:.3f} kW")
            ax.set_ylabel("Power (kW)")
            ax.set_title("Prediction vs Thresholds")
            ax.legend()
            ax.set_facecolor("#111")
            fig.patch.set_facecolor("#111")
            ax.tick_params(colors="white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")
            ax.spines[:].set_color("#444")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Estimated cost
            st.markdown("#### 💰 Estimated Cost (next 15 min)")
            rate_usd = 0.16   # USD per kWh (US average)
            rate_gbp = 0.24   # GBP per kWh (UK average)
            cost_kwh = predicted_kw * 0.25  # 15 min = 0.25 hours
            c1, c2 = st.columns(2)
            c1.metric("🇺🇸 USA (USD)", f"${cost_kwh * rate_usd:.4f}")
            c2.metric("🇬🇧 UK (GBP)",  f"£{cost_kwh * rate_gbp:.4f}")

# ── Tab 2: Model Performance ──────────────────────────────────────────────────
with tab2:
    st.subheader("Model Evaluation — Test Set")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Scatter
    sample = min(3000, len(y_test))
    idx = np.random.choice(len(y_test), sample, replace=False)
    axes[0].scatter(y_test[idx], y_pred[idx], alpha=0.3, s=5, color="#2ecc71")
    lims = [y_test.min(), y_test.max()]
    axes[0].plot(lims, lims, "r--", linewidth=1.5)
    axes[0].set_xlabel("Actual (kW)")
    axes[0].set_ylabel("Predicted (kW)")
    axes[0].set_title("Actual vs Predicted")
    axes[0].grid(True, alpha=0.3)

    # Error histogram
    errors = y_pred - y_test
    axes[1].hist(errors, bins=80, color="#3498db", edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Prediction Error (kW)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.info(
        f"**Test set stats** — Samples: {len(y_test):,} | "
        f"MAE: {metrics['MAE']:.4f} kW | "
        f"RMSE: {metrics['RMSE']:.4f} kW | "
        f"R²: {metrics['R²']:.4f}"
    )

# ── Tab 3: Daily Profile ──────────────────────────────────────────────────────
with tab3:
    st.subheader("Predicted 24-Hour Energy Profile")
    st.caption("Shows predicted consumption for every hour of the selected day/month "
               "using your current lag readings as the baseline.")

    day_num   = ["Monday","Tuesday","Wednesday","Thursday",
                 "Friday","Saturday","Sunday"].index(day)
    month_num = ["January","February","March","April","May","June",
                 "July","August","September","October","November","December"].index(month) + 1

    hours_range = np.arange(0, 24)
    preds_24 = []
    l1, l2, l3 = lag1, lag2, lag3
    for h in hours_range:
        feat = np.array([[h, day_num, month_num, l1, l2, l3]])
        p = float(model.predict(feat)[0])
        preds_24.append(max(0.0, p))
        l3, l2, l1 = l2, l1, p   # roll lags forward

    fig, ax = plt.subplots(figsize=(12, 4))
    colours_24 = ["#27ae60" if v < 0.5 else "#f39c12" if v < 1.2 else "#e74c3c"
                  for v in preds_24]
    bars = ax.bar(hours_range, preds_24, color=colours_24, edgecolor="none", width=0.8)
    ax.axhline(0.5,  color="#27ae60", linestyle="--", linewidth=1, alpha=0.7, label="Low threshold")
    ax.axhline(1.2,  color="#e74c3c", linestyle="--", linewidth=1, alpha=0.7, label="High threshold")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Predicted Power (kW)")
    ax.set_title(f"24-Hour Profile — {day}, {month}")
    ax.set_xticks(hours_range)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Table
    profile_df = pd.DataFrame({
        "Hour":           [f"{h:02d}:00" for h in hours_range],
        "Predicted (kW)": [round(v, 3) for v in preds_24],
        "Status":         [get_recommendation(v)[0] for v in preds_24],
        "Est. Cost USD":  [f"${v*0.25*0.16:.4f}" for v in preds_24],
        "Est. Cost GBP":  [f"£{v*0.25*0.24:.4f}" for v in preds_24],
    })

    def style_status(val):
        c = {"Normal": "#27ae60", "Monitor": "#f39c12", "Reduce Load": "#e74c3c"}.get(val, "#666")
        return f"background-color: {c}; color: white;"

    st.dataframe(
        profile_df.style.map(style_status, subset=["Status"]),
        use_container_width=True,
        hide_index=True,
        height=400,
    )

    csv = profile_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download 24h Profile CSV", csv, "energy_profile.csv", "text/csv")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #0d2318, #1a5c35);
        border: 2px solid #4ade80;
        border-radius: 14px;
        padding: 20px 28px;
        text-align: center;
        margin-bottom: 16px;
    ">
        <div style="font-size:0.8rem; color:#86efac; letter-spacing:1px; text-transform:uppercase; margin-bottom:6px;">
            🔬 Research Demo · UCI Dataset
        </div>
        <div style="font-size:1rem; color:#e2e8f0; margin-bottom:14px;">
            This app uses <b>synthetic data</b> modelled on the 
            <a href="https://doi.org/10.24432/C58K54" target="_blank" style="color:#86efac;">
            UCI Household Power Dataset</a> (Hebrail & Berard, 2006).
        </div>
        <div style="font-size:1.1rem; color:white; font-weight:600; margin-bottom:12px;">
            Want <b style="color:#4ade80">live UK grid prices</b>, real-time predictions & accurate cost analysis?
        </div>
        <a href="https://gridsense-labs.streamlit.app" target="_blank" style="
            background: #4ade80;
            color: #050f0a;
            font-weight: 700;
            font-size: 1rem;
            padding: 10px 28px;
            border-radius: 8px;
            text-decoration: none;
            display: inline-block;
        ">⚡ Try GridSense Labs →</a>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "<center style='color:grey;font-size:0.8rem;'>"
    "AI Energy Consumption Optimiser · Nnamdi Onuigbo · SmartFlow Systems · "
    "<a href='https://github.com/onuigbonnamdi/ai-energy-consumption-optimisation' "
    "target='_blank'>GitHub</a>"
    "</center>",
    unsafe_allow_html=True,
)
