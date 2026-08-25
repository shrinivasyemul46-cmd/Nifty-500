
import io
import time
import warnings
import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AI Swing & Positional Stock Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
.main .block-container {padding-top: 1rem; padding-bottom: 2rem;}
.card {
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 14px;
    padding: 14px;
    margin-bottom: 10px;
}
.small {font-size: .85rem; opacity: .8;}
.good {font-weight: 700;}
</style>
""", unsafe_allow_html=True)

st.title("📈 AI Swing & Positional Stock Assistant")
st.caption("Nifty 500 universe → Nifty 200 excluded → Technical + volume + breakout + fundamental quality scoring")

st.warning(
    "⚠️ This is a decision-support screener, not a profit guarantee. "
    "Always verify live NSE price, liquidity, corporate actions and your broker's execution before trading."
)

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def read_index_page(index_name):
    url = f"https://en.wikipedia.org/wiki/{index_name.replace(' ', '_')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    html = requests.get(url, headers=headers, timeout=20).text
    tables = pd.read_html(io.StringIO(html))

    candidates = []
    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        if "symbol" in cols:
            candidates.append(t)

    if not candidates:
        raise ValueError(f"Could not find a Symbol table for {index_name}")

    df = candidates[0].copy()
    df.columns = [str(c).strip() for c in df.columns]

    symbol_col = next(c for c in df.columns if c.lower() == "symbol")
    df["Symbol"] = (
        df[symbol_col].astype(str)
        .str.replace(".", "-", regex=False)
        .str.strip()
    )

    # Keep NSE-style symbols only
    df = df[df["Symbol"].str.len().between(1, 30)]
    return df.drop_duplicates("Symbol").reset_index(drop=True)


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def get_universe():
    n500 = read_index_page("NIFTY 500")
    n200 = read_index_page("NIFTY 200")
    excluded = set(n200["Symbol"])
    universe = n500[~n500["Symbol"].isin(excluded)].copy()
    return universe, n200


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df, period=14):
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def technical_features(df):
    d = df.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)

    d["EMA20"] = d["Close"].ewm(span=20, adjust=False).mean()
    d["EMA50"] = d["Close"].ewm(span=50, adjust=False).mean()
    d["EMA200"] = d["Close"].ewm(span=200, adjust=False).mean()
    d["RSI"] = rsi(d["Close"])
    d["ATR"] = atr(d)

    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACDSignal"] = d["MACD"].ewm(span=9, adjust=False).mean()

    d["VolAvg20"] = d["Volume"].rolling(20).mean()
    d["VolRatio"] = d["Volume"] / d["VolAvg20"]

    # Pivot = highest high of previous 20 sessions.
    d["Pivot20"] = d["High"].shift(1).rolling(20).max()
    d["Low20"] = d["Low"].rolling(20).min()
    d["High52"] = d["High"].rolling(252).max()
    d["Return20"] = d["Close"].pct_change(20) * 100
    d["Return60"] = d["Close"].pct_change(60) * 100

    # Tightness proxy: 20-day range relative to price.
    d["Range20Pct"] = (d["High"].rolling(20).max() - d["Low"].rolling(20).min()) / d["Close"] * 100

    return d


def score_row(d, fundamental=None, mode="Swing"):
    x = d.iloc[-1]
    prev = d.iloc[-2]

    score = 0
    reasons = []
    risks = []

    # Trend: 25 points
    if x["Close"] > x["EMA20"]:
        score += 5
        reasons.append("Close > EMA20")
    if x["EMA20"] > x["EMA50"]:
        score += 7
        reasons.append("EMA20 > EMA50")
    if x["EMA50"] > x["EMA200"]:
        score += 8
        reasons.append("EMA50 > EMA200")
    if x["Close"] > x["EMA200"]:
        score += 5
        reasons.append("Above EMA200")

    # Momentum: 20 points
    if 55 <= x["RSI"] <= 75:
        score += 10
        reasons.append("RSI in preferred zone")
    elif 50 <= x["RSI"] < 55:
        score += 5
    if x["MACD"] > x["MACDSignal"]:
        score += 7
        reasons.append("MACD bullish")
    if x["Return20"] > 0:
        score += 3
        reasons.append("20D momentum positive")

    # Breakout / volume: 30 points
    if x["Close"] > x["Pivot20"]:
        score += 12
        reasons.append("20D pivot breakout")
    elif x["Close"] >= x["Pivot20"] * 0.985:
        score += 5
        reasons.append("Near pivot")

    if x["VolRatio"] >= 2:
        score += 12
        reasons.append("Volume >= 2x average")
    elif x["VolRatio"] >= 1.5:
        score += 8
        reasons.append("Volume >= 1.5x average")
    elif x["VolRatio"] >= 1.2:
        score += 4

    if x["Close"] > prev["Close"]:
        score += 3

    # Structure: 10 points
    if x["Range20Pct"] <= 12:
        score += 6
        reasons.append("Relatively tight 20D range")
    elif x["Range20Pct"] <= 18:
        score += 3
    if x["Close"] > x["EMA20"] and x["Low"] >= d["Low"].tail(10).min():
        score += 4

    # Fundamental quality: 15 points, only from available verified fields
    if fundamental:
        rg = fundamental.get("revenueGrowth")
        eg = fundamental.get("earningsGrowth")
        roe = fundamental.get("returnOnEquity")
        de = fundamental.get("debtToEquity")

        if rg is not None and rg > 0:
            score += 3
        if eg is not None and eg > 0:
            score += 4
        if roe is not None and roe > 0.12:
            score += 4
        if de is not None and de < 120:
            score += 4

    # Risk flags
    if x["RSI"] > 78:
        risks.append("Overbought")
    if x["Close"] > x["EMA20"] + 2.5 * x["ATR"]:
        risks.append("Extended from EMA20")
    if x["VolRatio"] < 1:
        risks.append("Weak volume")
    if x["Close"] < x["EMA50"]:
        risks.append("Below EMA50")

    # Mode-specific minimums
    if mode == "Positional":
        if x["Return60"] > 0:
            score += 2
        if x["Close"] > x["High52"] * 0.90:
            score += 2

    score = min(100, int(round(score)))

    if score >= 80:
        grade = "🟢 A+"
    elif score >= 70:
        grade = "🟢 A"
    elif score >= 60:
        grade = "🟡 B"
    elif score >= 50:
        grade = "🟠 C"
    else:
        grade = "🔴 Avoid"

    return score, grade, reasons, risks


@st.cache_data(ttl=60 * 20, show_spinner=False)
def download_prices(symbols):
    tickers = [f"{s}.NS" for s in symbols]
    raw = yf.download(
        tickers=tickers,
        period="1y",
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )
    return raw


@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def fundamentals_for_symbols(symbols):
    out = {}
    for s in symbols:
        try:
            info = yf.Ticker(f"{s}.NS").get_info()
            out[s] = {
                "marketCap": info.get("marketCap"),
                "revenueGrowth": info.get("revenueGrowth"),
                "earningsGrowth": info.get("earningsGrowth"),
                "returnOnEquity": info.get("returnOnEquity"),
                "debtToEquity": info.get("debtToEquity"),
                "trailingPE": info.get("trailingPE"),
                "dividendYield": info.get("dividendYield"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
            }
        except Exception:
            out[s] = {}
    return out


def money(x):
    if x is None or pd.isna(x):
        return "NA"
    return f"₹{x:,.2f}"


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Screener Controls")

    mode = st.radio(
        "Trading style",
        ["Swing", "Positional"],
        help="Swing focuses on 20D breakout/momentum. Positional gives more weight to 60D trend and long-term structure."
    )

    min_score = st.slider("Minimum score", 50, 90, 70, 5)
    min_volume = st.slider("Minimum volume / 20D average", 1.0, 3.0, 1.5, 0.1)
    max_results = st.slider("Stocks to display", 5, 30, 15)

    st.divider()
    st.subheader("🧪 Optional filters")

    require_trend = st.checkbox("Require Close > EMA20 > EMA50 > EMA200", True)
    require_rsi = st.checkbox("Prefer RSI 55–75", True)
    require_breakout = st.checkbox("Require 20D breakout", False)
    avoid_extended = st.checkbox("Avoid price > EMA20 + 2.5 ATR", True)

    st.divider()
    st.caption("Universe: Nifty 500 minus Nifty 200.")
    st.caption("Data: Yahoo Finance for prototype; verify important values with NSE/broker before trading.")

# -----------------------------
# Universe
# -----------------------------
try:
    universe, n200 = get_universe()
except Exception as e:
    st.error("Automatic Nifty 500/Nifty 200 list could not be loaded.")
    st.info("Use the manual symbol box below. Enter NSE symbols separated by commas.")
    manual = st.text_area(
        "Manual NSE symbols",
        "TATAMOTORS, HINDALCO, PFC, POWERGRID, COFORGE, CHOLAFIN"
    )
    symbols = [x.strip().upper() for x in manual.split(",") if x.strip()]
    universe = pd.DataFrame({"Symbol": symbols})
    n200 = pd.DataFrame({"Symbol": []})

st.sidebar.success(f"Universe available: {len(universe)} stocks")

# -----------------------------
# Main actions
# -----------------------------
c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    run = st.button("🚀 RUN AI SCREENER", type="primary", use_container_width=True)
with c2:
    st.metric("Nifty 200 excluded", len(n200))
with c3:
    st.metric("Stocks scanned", len(universe))

if run:
    symbols = universe["Symbol"].tolist()

    # Keep scan practical for free cloud resources.
    # All symbols are attempted, but bad/empty symbols are skipped.
    with st.spinner(f"Downloading daily data for {len(symbols)} stocks..."):
        raw = download_prices(symbols)

    results = []
    charts_data = {}

    progress = st.progress(0)
    total = len(symbols)

    for i, s in enumerate(symbols):
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if s + ".NS" not in raw.columns.get_level_values(0):
                    progress.progress((i + 1) / total)
                    continue
                d = raw[s + ".NS"].dropna(how="all").copy()
            else:
                d = raw.copy()

            needed = {"Open", "High", "Low", "Close", "Volume"}
            if not needed.issubset(d.columns) or len(d) < 210:
                progress.progress((i + 1) / total)
                continue

            d = technical_features(d).dropna(subset=["EMA200", "RSI", "ATR", "VolRatio", "Pivot20"])
            if len(d) < 20:
                progress.progress((i + 1) / total)
                continue

            x = d.iloc[-1]

            trend_ok = (
                x["Close"] > x["EMA20"] > x["EMA50"] > x["EMA200"]
            )
            rsi_ok = 55 <= x["RSI"] <= 75
            breakout_ok = x["Close"] > x["Pivot20"]
            volume_ok = x["VolRatio"] >= min_volume
            extended = x["Close"] > x["EMA20"] + 2.5 * x["ATR"]

            if require_trend and not trend_ok:
                progress.progress((i + 1) / total)
                continue
            if require_rsi and not rsi_ok:
                progress.progress((i + 1) / total)
                continue
            if require_breakout and not breakout_ok:
                progress.progress((i + 1) / total)
                continue
            if not volume_ok:
                progress.progress((i + 1) / total)
                continue
            if avoid_extended and extended:
                progress.progress((i + 1) / total)
                continue

            score, grade, reasons, risks = score_row(d, None, mode)

            entry = float(x["Close"])
            atr_value = float(x["ATR"])
            pivot = float(x["Pivot20"])
            base_low = float(d["Low"].tail(10).min())

            # Conservative risk model; not a prediction.
            stop = min(base_low, entry - 1.5 * atr_value)
            if stop >= entry:
                stop = entry - 1.5 * atr_value

            risk = max(entry - stop, 0.01)
            target1 = entry + 2 * risk
            target2 = entry + 3 * risk

            results.append({
                "Symbol": s,
                "Score": score,
                "Grade": grade,
                "Close": round(entry, 2),
                "Pivot20": round(pivot, 2),
                "RSI": round(float(x["RSI"]), 1),
                "Vol x": round(float(x["VolRatio"]), 2),
                "EMA20": round(float(x["EMA20"]), 2),
                "EMA50": round(float(x["EMA50"]), 2),
                "EMA200": round(float(x["EMA200"]), 2),
                "20D %": round(float(x["Return20"]), 1),
                "60D %": round(float(x["Return60"]), 1),
                "SL": round(stop, 2),
                "T1 2R": round(target1, 2),
                "T2 3R": round(target2, 2),
                "Breakout": "YES" if breakout_ok else "WATCH",
                "Reasons": " • ".join(reasons[:6]),
                "Risk flags": " • ".join(risks) if risks else "None",
            })
            charts_data[s] = d

        except Exception:
            pass

        progress.progress((i + 1) / total)

    progress.empty()

    if not results:
        st.error(
            "No stocks matched the current filters. Try Minimum score 60, "
            "uncheck 'Require 20D breakout', or lower the volume filter."
        )
        st.stop()

    result_df = pd.DataFrame(results).sort_values(
        ["Score", "Vol x"], ascending=[False, False]
    ).reset_index(drop=True)

    st.session_state["results"] = result_df
    st.session_state["charts"] = charts_data

# -----------------------------
# Results
# -----------------------------
if "results" not in st.session_state:
    st.info("👆 Set your filters and press **RUN AI SCREENER**.")
    st.markdown("""
### What this app checks

**1. Trend**
- Close > EMA20 > EMA50 > EMA200

**2. Momentum**
- RSI 55–75
- Bullish MACD
- Positive 20-day momentum

**3. Breakout**
- Previous 20-day high as pivot
- Volume preferably ≥ 1.5× 20-day average
- 2× volume gets extra score

**4. Structure**
- Tight 20-day range
- Price strength above moving averages

**5. Risk**
- ATR/base-low based stop
- 2R and 3R reference targets

**6. Fundamental layer**
- Revenue growth
- Earnings growth
- ROE
- Debt/equity
- PE/dividend information when available

The score is a ranking aid—not a probability of profit.
""")
    st.stop()

df = st.session_state["results"]
charts = st.session_state["charts"]

# Top summary
top = df.iloc[0]
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("🏆 Top Score", f"{top['Score']}/100")
m2.metric("🎯 Top Stock", top["Symbol"])
m3.metric("📈 RSI", top["RSI"])
m4.metric("🔊 Volume", f"{top['Vol x']}x")
m5.metric("🚦 Grade", top["Grade"])

st.subheader("🏆 Best Setups")

display_cols = [
    "Symbol", "Score", "Grade", "Close", "Pivot20", "RSI",
    "Vol x", "20D %", "60D %", "SL", "T1 2R", "T2 3R", "Breakout"
]
st.dataframe(
    df.head(max_results)[display_cols],
    use_container_width=True,
    hide_index=True,
)

# -----------------------------
# Trade plan
# -----------------------------
st.subheader("🎯 New-Trader Trade Plan")

selected = st.selectbox(
    "Select a stock to inspect",
    df.head(max_results)["Symbol"].tolist()
)
row = df[df["Symbol"] == selected].iloc[0]

a, b, c, d, e = st.columns(5)
a.metric("Entry reference", money(row["Close"]))
b.metric("Pivot", money(row["Pivot20"]))
c.metric("Stop", money(row["SL"]))
d.metric("Target 1", money(row["T1 2R"]))
e.metric("Target 2", money(row["T2 3R"]))

st.markdown(
    f"**{row['Grade']} {selected} — Score {row['Score']}/100**  \n"
    f"**Why:** {row['Reasons']}  \n"
    f"**Risk flags:** {row['Risk flags']}"
)

st.info(
    "Execution rule for beginners: do not buy merely because a stock appears in the list. "
    "Prefer a clean breakout/close above the pivot with adequate volume, then define the stop "
    "before entry. Position size should be based on the rupee amount you are willing to lose."
)

# -----------------------------
# Chart
# -----------------------------
if selected in charts:
    d = charts[selected].tail(150)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=d.index,
        open=d["Open"],
        high=d["High"],
        low=d["Low"],
        close=d["Close"],
        name="Price",
    ))
    fig.add_trace(go.Scatter(x=d.index, y=d["EMA20"], name="EMA20"))
    fig.add_trace(go.Scatter(x=d.index, y=d["EMA50"], name="EMA50"))
    fig.add_trace(go.Scatter(x=d.index, y=d["EMA200"], name="EMA200"))
    fig.add_trace(go.Scatter(
        x=d.index, y=d["Pivot20"], name="Pivot20",
        line=dict(dash="dot")
    ))
    fig.update_layout(
        height=550,
        xaxis_rangeslider_visible=False,
        title=f"{selected} — Trend / Breakout View"
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Fundamental check
# -----------------------------
st.subheader("🏢 Fundamental Quality Check")

with st.spinner("Loading fundamental information for the selected stock..."):
    f = fundamentals_for_symbols([selected]).get(selected, {})

fund_cols = st.columns(6)
labels = [
    ("Market Cap", f.get("marketCap")),
    ("Revenue Growth", f.get("revenueGrowth")),
    ("Earnings Growth", f.get("earningsGrowth")),
    ("ROE", f.get("returnOnEquity")),
    ("Debt/Equity", f.get("debtToEquity")),
    ("PE", f.get("trailingPE")),
]
for col, (label, value) in zip(fund_cols, labels):
    if value is None:
        shown = "NA"
    elif label in ("Revenue Growth", "Earnings Growth", "ROE"):
        shown = f"{value * 100:.1f}%"
    else:
        shown = f"{value:,.1f}"
    col.metric(label, shown)

st.caption(
    f"Sector: {f.get('sector', 'NA')} | Industry: {f.get('industry', 'NA')} | "
    "Fundamental fields may be unavailable or delayed depending on the data source."
)

# -----------------------------
# Industry strength
# -----------------------------
st.subheader("🔥 Relative Opportunity View")

all_symbols = df["Symbol"].tolist()
with st.spinner("Loading industry labels for shortlisted stocks..."):
    f_all = fundamentals_for_symbols(all_symbols[:min(len(all_symbols), 20)])

industry_rows = []
for s, info in f_all.items():
    industry_rows.append({
        "Symbol": s,
        "Industry": info.get("industry", "Unknown"),
        "Score": int(df.loc[df["Symbol"] == s, "Score"].iloc[0]),
    })

ind = pd.DataFrame(industry_rows)
if not ind.empty:
    ind_summary = (
        ind.groupby("Industry", dropna=False)
        .agg(Stocks=("Symbol", "count"), AvgScore=("Score", "mean"))
        .sort_values(["AvgScore", "Stocks"], ascending=[False, False])
        .head(10)
        .reset_index()
    )
    ind_summary["AvgScore"] = ind_summary["AvgScore"].round(1)
    st.dataframe(ind_summary, use_container_width=True, hide_index=True)

# -----------------------------
# Export
# -----------------------------
st.subheader("📥 Export Watchlist")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Today's Swing/Positional Watchlist",
    csv,
    file_name="ai_swing_positional_watchlist.csv",
    mime="text/csv",
)

st.divider()
st.caption(
    "Data note: this prototype uses Yahoo Finance market/fundamental data. "
    "It does not place orders. Verify live prices and corporate events with NSE/your broker."
)
