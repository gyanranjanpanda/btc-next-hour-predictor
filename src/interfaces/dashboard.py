"""
AlphaI × Polaris — Bitcoin Next-Hour Predictor Dashboard.

Run:
    streamlit run src/interfaces/dashboard.py
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.application.use_cases import PredictNextHourUseCase
from src.domain.simulator import GBMSimulator
from src.infrastructure.binance_client import BinanceDataProvider
from src.infrastructure.jsonl_repository import JsonlPredictionRepository

# ───────────────────────────── Page Config ──────────────────────────────

st.set_page_config(
    page_title="BTC Predictor — AlphaI × Polaris",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ───────────────────────────── Inline-Style CSS ─────────────────────────

_BG = "background:linear-gradient(145deg,#161b22 0%,#1c2333 100%)"
_CARD = f"{_BG};border:1px solid #21262d;border-radius:16px;padding:1.2rem 1rem;text-align:center;border-top:3px solid #F7931A"
_LABEL = "font-size:0.72rem;font-weight:600;color:#7d8590;text-transform:uppercase;letter-spacing:1px;margin-bottom:0.4rem"
_HINT = "font-size:0.68rem;color:#484f58;margin-top:0.3rem"

_COLORS = {
    "green": "#3fb950",
    "cyan": "#58a6ff",
    "purple": "#bc8cff",
    "bitcoin": "#F7931A",
    "white": "#e6edf3",
    "red": "#f85149",
}


def _kpi_html(label: str, value: str, hint: str, color: str = "white") -> str:
    """Returns a styled KPI card using inline CSS only."""
    c = _COLORS.get(color, "#e6edf3")
    return (
        f'<div style="{_CARD}">'
        f'<div style="{_LABEL}">{label}</div>'
        f'<div style="font-size:1.3rem;font-weight:800;color:{c};line-height:1.2;white-space:nowrap">{value}</div>'
        f'<div style="{_HINT}">{hint}</div>'
        f'</div>'
    )


def _section(text: str) -> str:
    """Section header with orange dot."""
    return (
        f'<div style="font-size:1.1rem;font-weight:700;color:#e6edf3;margin:1.5rem 0 0.8rem 0;display:flex;align-items:center;gap:0.5rem">'
        f'<span style="width:8px;height:8px;background:#F7931A;border-radius:50%;display:inline-block"></span>'
        f' {text}</div>'
    )


# Inject global overrides that DO survive on Streamlit Cloud
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
.stApp {background:linear-gradient(180deg,#0a0a0f 0%,#0d1117 50%,#0a0a0f 100%) !important;font-family:'Inter',sans-serif !important}
.block-container {padding-top:2rem}
#MainMenu,footer,header {visibility:hidden}
[data-testid="stMetricValue"] {font-family:'Inter',sans-serif;font-weight:800}
</style>""", unsafe_allow_html=True)

# Auto-refresh every 5 minutes via HTML meta tag
st.markdown('<meta http-equiv="refresh" content="300">', unsafe_allow_html=True)


# ───────────────────────────── Dependencies ─────────────────────────────

@st.cache_resource
def get_dependencies(_version: str = "v2"):
    data_provider = BinanceDataProvider()
    simulator = GBMSimulator(
        num_simulations=10_000,
        volatility_lookback=24,
        ewma_span=12,
    )
    use_case = PredictNextHourUseCase(data_provider, simulator)
    repo = JsonlPredictionRepository("live_predictions.jsonl")
    return data_provider, use_case, repo


def load_backtest_metrics() -> tuple[dict[str, str], list[dict]]:
    """Reads backtest_results.jsonl and returns metrics + raw rows."""
    filepath = "backtest_results.jsonl"
    empty_metrics = {"coverage": "—", "avg_width": "—", "winkler": "—", "total": "—"}
    if not os.path.exists(filepath):
        return empty_metrics, []

    try:
        with open(filepath) as fh:
            rows = [json.loads(line) for line in fh if line.strip()]

        if not rows:
            return empty_metrics, []

        hits = sum(
            1
            for r in rows
            if r.get("actual_close")
            and r["lower_bound"] <= r["actual_close"] <= r["upper_bound"]
        )
        coverage = hits / len(rows)
        avg_width = sum(r["upper_bound"] - r["lower_bound"] for r in rows) / len(rows)

        winkler_scores = []
        for r in rows:
            actual = r.get("actual_close")
            if actual is None:
                continue
            alpha = 1.0 - r["confidence_interval"]
            width = r["upper_bound"] - r["lower_bound"]
            if actual < r["lower_bound"]:
                score = width + (2.0 / alpha) * (r["lower_bound"] - actual)
            elif actual > r["upper_bound"]:
                score = width + (2.0 / alpha) * (actual - r["upper_bound"])
            else:
                score = width
            winkler_scores.append(score)

        mean_winkler = (
            sum(winkler_scores) / len(winkler_scores) if winkler_scores else 0.0
        )

        metrics = {
            "coverage": f"{coverage:.2%}",
            "avg_width": f"${avg_width:,.2f}",
            "winkler": f"{mean_winkler:,.2f}",
            "total": f"{len(rows):,}",
        }
        return metrics, rows
    except Exception:
        return {"coverage": "Error", "avg_width": "Error", "winkler": "Error", "total": "Error"}, []


def build_chart(
    candles_df: pd.DataFrame,
    prediction_lower: float,
    prediction_upper: float,
    prediction_time: datetime,
    past_predictions: list | None = None,
) -> go.Figure:
    """Builds a premium dark-themed candlestick chart with prediction ribbon."""

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=candles_df["time"],
            open=candles_df["open"],
            high=candles_df["high"],
            low=candles_df["low"],
            close=candles_df["close"],
            name="BTCUSDT",
            increasing=dict(line=dict(color="#3fb950", width=1), fillcolor="rgba(35,134,54,0.5)"),
            decreasing=dict(line=dict(color="#f85149", width=1), fillcolor="rgba(218,54,54,0.5)"),
        )
    )

    if past_predictions:
        hist_df = pd.DataFrame(
            [{"time": p.timestamp, "lower": p.lower_bound, "upper": p.upper_bound}
             for p in past_predictions]
        )
        hist_df = hist_df[hist_df["time"] >= candles_df["time"].min()]

        if not hist_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([hist_df["time"], hist_df["time"][::-1]]),
                    y=pd.concat([hist_df["upper"], hist_df["lower"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(247,147,26,0.08)",
                    line=dict(color="rgba(247,147,26,0.25)", width=1),
                    hoverinfo="skip",
                    showlegend=True,
                    name="Historical Predictions",
                )
            )

    current_time = candles_df["time"].iloc[-1]
    fig.add_trace(
        go.Scatter(
            x=[current_time, prediction_time, prediction_time, current_time, current_time],
            y=[prediction_upper, prediction_upper, prediction_lower, prediction_lower, prediction_upper],
            fill="toself",
            fillcolor="rgba(88,166,255,0.15)",
            line=dict(color="rgba(88,166,255,0.8)", width=2, dash="dot"),
            name="Next Hour Forecast",
        )
    )

    current_close = candles_df["close"].iloc[-1]
    fig.add_hline(
        y=current_close, line_dash="dash", line_color="#F7931A", line_width=1,
        annotation_text=f"${current_close:,.0f}",
        annotation_position="right",
        annotation_font=dict(color="#F7931A", size=11),
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,17,23,0.95)",
        autosize=True,
        xaxis=dict(
            rangeslider=dict(visible=False),
            gridcolor="#21262d",
            linecolor="#21262d",
            showspikes=True,
            spikecolor="#F7931A",
            spikethickness=1,
            spikedash="dot",
            spikemode="across",
            autorange=True,
        ),
        yaxis=dict(
            gridcolor="#21262d",
            linecolor="#21262d",
            tickprefix="$",
            tickformat=",",
            showspikes=True,
            spikecolor="#F7931A",
            spikethickness=1,
            spikedash="dot",
            autorange=True,
            fixedrange=False,
        ),
        height=550,
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color="#7d8590", size=11), bgcolor="rgba(0,0,0,0)",
        ),
        font=dict(family="Inter, sans-serif", color="#e6edf3"),
        hovermode="x unified",
        dragmode="pan",
    )

    return fig


def build_distribution_chart(
    simulated_prices: np.ndarray,
    lower: float,
    upper: float,
    current_price: float,
) -> go.Figure:
    """Histogram of Monte Carlo simulated prices with percentile bounds marked."""
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=simulated_prices,
            nbinsx=80,
            marker_color="rgba(247,147,26,0.35)",
            marker_line=dict(color="#F7931A", width=0.5),
            name="Simulated Prices",
            hovertemplate="Price: $%{x:,.0f}<br>Count: %{y}<extra></extra>",
        )
    )

    # 2.5th percentile (lower bound)
    fig.add_vline(
        x=lower, line_dash="dash", line_color="#58a6ff", line_width=2,
        annotation_text=f"2.5th: ${lower:,.0f}",
        annotation_position="top left",
        annotation_font=dict(color="#58a6ff", size=10),
    )
    # 97.5th percentile (upper bound)
    fig.add_vline(
        x=upper, line_dash="dash", line_color="#58a6ff", line_width=2,
        annotation_text=f"97.5th: ${upper:,.0f}",
        annotation_position="top right",
        annotation_font=dict(color="#58a6ff", size=10),
    )
    # Current price
    fig.add_vline(
        x=current_price, line_dash="solid", line_color="#F7931A", line_width=2,
        annotation_text=f"Current: ${current_price:,.0f}",
        annotation_position="top",
        annotation_font=dict(color="#F7931A", size=10),
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,17,23,0.95)",
        xaxis=dict(
            title="Simulated Price ($)",
            gridcolor="#21262d",
            tickprefix="$",
            tickformat=",",
        ),
        yaxis=dict(title="Frequency", gridcolor="#21262d"),
        height=350,
        margin=dict(l=50, r=20, t=30, b=50),
        font=dict(family="Inter, sans-serif", color="#e6edf3"),
        showlegend=False,
    )

    return fig


def build_rolling_coverage_chart(backtest_rows: list[dict]) -> go.Figure | None:
    """Plots 50-bar rolling coverage from backtest data."""
    if len(backtest_rows) < 60:
        return None

    hits = []
    for r in backtest_rows:
        actual = r.get("actual_close")
        if actual is None:
            hits.append(0)
            continue
        hit = 1 if r["lower_bound"] <= actual <= r["upper_bound"] else 0
        hits.append(hit)

    window = 50
    rolling = pd.Series(hits).rolling(window).mean()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(len(rolling))),
            y=rolling,
            mode="lines",
            line=dict(color="#3fb950", width=2),
            name="Rolling Coverage",
            fill="tozeroy",
            fillcolor="rgba(63,185,80,0.08)",
        )
    )

    # 95% target line
    fig.add_hline(
        y=0.95, line_dash="dash", line_color="#F7931A", line_width=1,
        annotation_text="95% Target",
        annotation_position="right",
        annotation_font=dict(color="#F7931A", size=10),
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,17,23,0.95)",
        xaxis=dict(
            title="Prediction #",
            gridcolor="#21262d",
        ),
        yaxis=dict(
            title="Coverage",
            gridcolor="#21262d",
            tickformat=".0%",
            range=[0.75, 1.05],
        ),
        height=350,
        margin=dict(l=50, r=20, t=30, b=50),
        font=dict(family="Inter, sans-serif", color="#e6edf3"),
        showlegend=False,
    )

    return fig


# ───────────────────────────── Main App ─────────────────────────────────

def main() -> None:
    # Hero header with inline gradient
    st.markdown(
        '<div style="font-size:2.8rem;font-weight:900;'
        'background:linear-gradient(135deg,#F7931A 0%,#FFD93D 50%,#F7931A 100%);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
        'text-align:center;margin-bottom:0.2rem;letter-spacing:-1px">'
        '₿ Bitcoin Next-Hour Predictor</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="text-align:center;color:#6c7a89;font-size:1rem;margin-bottom:0.3rem;font-weight:400">'
        'GBM + GARCH(1,1) with Student-t innovations · 95% confidence interval · Live from Binance</div>',
        unsafe_allow_html=True,
    )

    # Countdown to next hourly prediction
    now_utc = datetime.now(timezone.utc)
    next_hour = now_utc.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    remaining = next_hour - now_utc
    mins_left = int(remaining.total_seconds()) // 60
    updated_ago = now_utc.strftime("%H:%M UTC")

    st.markdown(
        f'<div style="text-align:center;color:#484f58;font-size:0.78rem;margin-bottom:1.8rem">'
        f'Last updated: {updated_ago} · Next prediction in: ~{mins_left} min · Auto-refreshes every 5 min</div>',
        unsafe_allow_html=True,
    )

    data_provider, use_case, repo = get_dependencies()

    # ── Backtest Metrics ──
    metrics, backtest_rows = load_backtest_metrics()
    st.markdown(_section("Backtest Performance (30 Days)"), unsafe_allow_html=True)

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.markdown(_kpi_html("Coverage", metrics["coverage"], "Target ≈ 95%", "green"), unsafe_allow_html=True)
    with b2:
        st.markdown(_kpi_html("Average Width", metrics["avg_width"], "Narrower = Better", "cyan"), unsafe_allow_html=True)
    with b3:
        st.markdown(_kpi_html("Mean Winkler Score", metrics["winkler"], "Lower = Better", "purple"), unsafe_allow_html=True)
    with b4:
        st.markdown(_kpi_html("Total Predictions", metrics["total"], "Out-of-sample bars"), unsafe_allow_html=True)

    # ── Live Prediction ──
    st.markdown(_section("Live Prediction"), unsafe_allow_html=True)

    with st.spinner("Fetching live data from Binance & running Monte Carlo simulation …"):
        try:
            prediction, candles, sim_result = use_case.execute(lookback=500)
            current_price = candles[-1].close_price
            repo.save(prediction)
        except Exception as exc:
            st.error(f"Failed to fetch data or run simulation: {exc}")
            return

    price_change = candles[-1].close_price - candles[-2].close_price
    price_change_pct = (price_change / candles[-2].close_price) * 100
    change_arrow = "▲" if price_change >= 0 else "▼"
    change_color = "green" if price_change >= 0 else "red"
    change_text = f"{change_arrow} ${abs(price_change):,.2f} ({price_change_pct:+.2f}%)"

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown(_kpi_html("Current BTC Price", f"${current_price:,.2f}", change_text, "bitcoin"), unsafe_allow_html=True)
    with p2:
        st.markdown(_kpi_html("Predicted Lower Bound", f"${prediction.lower_bound:,.2f}", "2.5th percentile", "cyan"), unsafe_allow_html=True)
    with p3:
        st.markdown(_kpi_html("Predicted Upper Bound", f"${prediction.upper_bound:,.2f}", "97.5th percentile", "cyan"), unsafe_allow_html=True)
    with p4:
        st.markdown(_kpi_html("Prediction Width", f"${prediction.width:,.2f}", "95% confidence band"), unsafe_allow_html=True)

    # ── Price Chart ──
    st.markdown(_section("Last 100 Hours + Forecast Ribbon"), unsafe_allow_html=True)

    display_candles = candles[-100:]
    candles_df = pd.DataFrame([
        {"time": c.timestamp, "open": c.open_price, "high": c.high_price, "low": c.low_price, "close": c.close_price}
        for c in display_candles
    ])

    past_preds = repo.get_all()[-50:]
    chart = build_chart(candles_df, prediction.lower_bound, prediction.upper_bound, prediction.timestamp, past_predictions=past_preds)
    st.plotly_chart(
        chart,
        use_container_width=True,
        config={"responsive": True, "displayModeBar": False, "scrollZoom": True},
    )

    # ── Monte Carlo Distribution + Model Parameters ──
    dist_col, params_col = st.columns([3, 2])

    with dist_col:
        st.markdown(_section("Monte Carlo Price Distribution (10,000 paths)"), unsafe_allow_html=True)
        dist_chart = build_distribution_chart(
            sim_result.simulated_prices,
            prediction.lower_bound,
            prediction.upper_bound,
            current_price,
        )
        st.plotly_chart(
            dist_chart,
            use_container_width=True,
            config={"responsive": True, "displayModeBar": False},
        )

    with params_col:
        st.markdown(_section("Fitted Model Parameters"), unsafe_allow_html=True)
        regime_emoji = {"low": "🟢 Low", "normal": "🟡 Normal", "high": "🔴 High"}
        method_label = "GARCH(1,1)" if sim_result.fitting_method == "garch" else "EWMA Fallback"
        annualized_vol = sim_result.fitted_sigma * np.sqrt(8760) * 100

        param_rows = [
            ("Fitting Method", method_label),
            ("Volatility Regime", regime_emoji.get(sim_result.volatility_regime, sim_result.volatility_regime)),
            ("Hourly σ (sigma)", f"{sim_result.fitted_sigma * 100:.4f}%"),
            ("Annualised Vol", f"{annualized_vol:.1f}%"),
            ("Drift μ (mu)", f"{sim_result.fitted_mu * 100:.6f}%"),
            ("Student-t ν (nu)", f"{sim_result.fitted_nu:.2f}"),
            ("Sigma Multiplier", f"×{sim_result.sigma_multiplier:.2f}"),
        ]

        params_html = '<div style="' + _BG + ';border:1px solid #21262d;border-radius:16px;padding:1.2rem;border-top:3px solid #F7931A">'
        for label, value in param_rows:
            params_html += (
                f'<div style="display:flex;justify-content:space-between;padding:0.45rem 0;border-bottom:1px solid #21262d">'
                f'<span style="color:#7d8590;font-size:0.82rem">{label}</span>'
                f'<span style="color:#e6edf3;font-weight:600;font-size:0.82rem">{value}</span>'
                f'</div>'
            )
        params_html += '</div>'
        st.markdown(params_html, unsafe_allow_html=True)

    # ── Rolling Coverage Chart ──
    if backtest_rows:
        st.markdown(_section("Backtest Rolling Coverage (50-bar window)"), unsafe_allow_html=True)
        rolling_chart = build_rolling_coverage_chart(backtest_rows)
        if rolling_chart:
            st.plotly_chart(
                rolling_chart,
                use_container_width=True,
                config={"responsive": True, "displayModeBar": False},
            )

    # ── How It Works + Download ──
    with st.expander("📐 How It Works"):
        st.markdown("""
**Geometric Brownian Motion (GBM)** models BTC's hourly log-returns as a stochastic process with drift and diffusion.

1. **Volatility Estimation**: GARCH(1,1) captures volatility clustering — calm hours produce narrow ranges, volatile hours produce wider ones.
2. **Fat Tails**: Student-t distributed innovations (not Gaussian) handle Bitcoin's frequent large moves. Degrees of freedom (ν) are fit via MLE, clamped to [3, 30].
3. **Regime Detection**: Recent volatility is compared against historical baseline to classify the market as Low / Normal / High volatility. Each regime uses a different model-risk buffer.
4. **10,000 Monte Carlo Paths**: Simulated prices are generated using `S(t+1) = S(t) × exp((μ - σ²/2)dt + σ√dt × Z)` where Z ~ t(ν) variance-normalised.
5. **95% Interval**: The 2.5th and 97.5th percentiles of simulated prices form the prediction bounds.
        """)

    if os.path.exists("backtest_results.jsonl"):
        with open("backtest_results.jsonl", "rb") as fh:
            st.download_button(
                "📥 Download Backtest Results (JSONL)",
                data=fh,
                file_name="backtest_results.jsonl",
                mime="application/jsonl",
            )


if __name__ == "__main__":
    main()
