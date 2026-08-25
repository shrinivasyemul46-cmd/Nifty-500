
import io
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AI Market Decision System V3",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: .8rem; padding-bottom: 2rem;}
div[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.20);
    border-radius: 12px;
    padding: 8px;
}
.small-note {font-size:.82rem; opacity:.78;}
</style>
""", unsafe_allow_html=True)

st.title("📊 AI Market Decision System V3")
st.caption(
    "NIFTY 500 • Market → Segment → Sector → Stock → Setup → Risk → Decision"
)
st.warning(
    "⚠️ Decision-support tool only. No score guarantees profit. "
    "Verify live NSE/broker prices, liquidity, corporate actions, results and news before trading."
)

HEADERS = {"User-Agent": "Mozilla/5.0"}

# ---------------------------------------------------------
# Universe
# ---------------------------------------------------------
@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def get_nifty500():
    url = "https://en.wikipedia.org/wiki/NIFTY_500"
    r = requests.get(url, headers=HEADERS, timeout=25)
    r.raise_for_status()
    tables = pd.read_html(io.StringIO(r.text))
    table = None
    for t in tables:
        cols = [str(c).strip().lower() for c in t.columns]
        if "symbol" in cols:
            table = t
            break
    if table is None:
        raise ValueError("NIFTY 500 constituent table not found")
    table.columns = [str(c).strip() for c in table.columns]
    sym = next(c for c in table.columns if c.lower() == "symbol")
    out = table[[sym]].rename(columns={sym: "Symbol"})
    out["Symbol"] = (
        out["Symbol"].astype(str)
        .str.replace(".", "-", regex=False)
        .str.strip()
        .str.upper()
    )
    return out.drop_duplicates("Symbol").reset_index(drop=True)

# ---------------------------------------------------------
# Market / sector index proxies
# ---------------------------------------------------------
INDEX_TICKERS = {
    "Nifty 50": "^NSEI",
    "India VIX": "^INDIAVIX",
    "Nifty Next 50": "^NSMIDCP",
    "Nifty Midcap 150": "^NIFTYMIDCAP150.NS",
    "Nifty Smallcap 250": "^NIFTYSMLCAP250.NS",
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Nikkei": "^N225",
    "Hang Seng": "^HSI",
}

SECTOR_TICKERS = {
    "Auto": "^CNXAUTO",
    "Bank": "^NSEBANK",
    "Financial Services": "NIFTY_FIN_SERVICE.NS",
    "IT": "^CNXIT",
    "Pharma": "^CNXPHARMA",
    "Metal": "^CNXMETAL",
    "FMCG": "^CNXFMCG",
    "Realty": "^CNXREALTY",
    "Energy": "^CNXENERGY",
    "Media": "^CNXMEDIA",
    "PSU Bank": "^CNXPSUBANK",
}

@st.cache_data(ttl=60 * 20, show_spinner=False)
def index_snapshot(tickers):
    rows = []
    for name, ticker in tickers.items():
        try:
            d = yf.download(
                ticker, period="9mo", interval="1d",
                auto_adjust=False, progress=False, threads=False
            )
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            d = d.dropna(subset=["Close"])
            if len(d) < 25:
                continue
            c = float(d["Close"].iloc[-1])
            r5 = float(d["Close"].iloc[-6]) if len(d) > 6 else float(d["Close"].iloc[0])
            r20 = float(d["Close"].iloc[-21]) if len(d) > 21 else float(d["Close"].iloc[0])
            r60 = float(d["Close"].iloc[-61]) if len(d) > 61 else float(d["Close"].iloc[0])
            ema50 = float(d["Close"].ewm(span=50, adjust=False).mean().iloc[-1])
            ema200 = float(d["Close"].ewm(span=200, adjust=False).mean().iloc[-1])
            rows.append({
                "Index": name, "Last": c,
                "5D": (c / r5 - 1) * 100,
                "20D": (c / r20 - 1) * 100,
                "60D": (c / r60 - 1) * 100,
                "Above50": c > ema50,
                "Above200": c > ema200,
            })
        except Exception:
            pass
    return pd.DataFrame(rows)

# ---------------------------------------------------------
# Indicators
# ---------------------------------------------------------
def calc_rsi(s, period=14):
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / period, adjust=False).mean()
    al = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def calc_atr(d, period=14):
    prev = d["Close"].shift(1)
    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - prev).abs(),
        (d["Low"] - prev).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()

def add_indicators(d):
    d = d.copy()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)

    d["EMA20"] = d["Close"].ewm(span=20, adjust=False).mean()
    d["EMA50"] = d["Close"].ewm(span=50, adjust=False).mean()
    d["EMA200"] = d["Close"].ewm(span=200, adjust=False).mean()
    d["RSI"] = calc_rsi(d["Close"])
    d["ATR"] = calc_atr(d)

    e12 = d["Close"].ewm(span=12, adjust=False).mean()
    e26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = e12 - e26
    d["MACDSignal"] = d["MACD"].ewm(span=9, adjust=False).mean()

    d["Vol20"] = d["Volume"].rolling(20).mean()
    d["VolX"] = d["Volume"] / d["Vol20"]
    d["Pivot20"] = d["High"].shift(1).rolling(20).max()
    d["Pivot55"] = d["High"].shift(1).rolling(55).max()
    d["Range20"] = (
        d["High"].rolling(20).max() -
        d["Low"].rolling(20).min()
    ) / d["Close"] * 100
    d["Ret5"] = d["Close"].pct_change(5) * 100
    d["Ret20"] = d["Close"].pct_change(20) * 100
    d["Ret60"] = d["Close"].pct_change(60) * 100
    d["High252"] = d["High"].rolling(252).max()
    return d

@st.cache_data(ttl=60 * 20, show_spinner=False)
def download_prices(symbols):
    return yf.download(
        tickers=[f"{s}.NS" for s in symbols],
        period="1y",
        interval="1d",
        auto_adjust=False,
        group_by="ticker",
        threads=True,
        progress=False,
    )

def extract(raw, symbol):
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            lev = set(raw.columns.get_level_values(0))
            if symbol + ".NS" in lev:
                d = raw[symbol + ".NS"].copy()
            elif symbol in lev:
                d = raw[symbol].copy()
            else:
                return None
        else:
            d = raw.copy()
        if not {"Open", "High", "Low", "Close", "Volume"}.issubset(d.columns):
            return None
        return d.dropna(subset=["Close"])
    except Exception:
        return None

@st.cache_data(ttl=60 * 60 * 12, show_spinner=False)
def get_fundamentals(symbols):
    out = {}
    for s in symbols:
        try:
            info = yf.Ticker(f"{s}.NS").get_info()
            out[s] = {
                "marketCap": info.get("marketCap"),
                "revenueGrowth": info.get("revenueGrowth"),
                "earningsGrowth": info.get("earningsGrowth"),
                "returnOnEquity": info.get("returnOnEquity"),
                "returnOnAssets": info.get("returnOnAssets"),
                "debtToEquity": info.get("debtToEquity"),
                "profitMargins": info.get("profitMargins"),
                "trailingPE": info.get("trailingPE"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
            }
        except Exception:
            out[s] = {}
    return out

def fnum(x):
    try:
        return None if x is None or pd.isna(x) else float(x)
    except Exception:
        return None

def fundamental_score(f):
    score = 0
    why, risk = [], []

    rg = fnum(f.get("revenueGrowth"))
    eg = fnum(f.get("earningsGrowth"))
    roe = fnum(f.get("returnOnEquity"))
    roa = fnum(f.get("returnOnAssets"))
    de = fnum(f.get("debtToEquity"))
    margin = fnum(f.get("profitMargins"))
    pe = fnum(f.get("trailingPE"))

    if rg is not None:
        if rg > .10: score += 3; why.append("Revenue growth >10%")
        elif rg > 0: score += 1
        else: risk.append("Revenue growth weak/negative")
    if eg is not None:
        if eg > .10: score += 3; why.append("Earnings growth >10%")
        elif eg > 0: score += 1
        else: risk.append("Earnings growth weak/negative")
    if roe is not None:
        if roe > .15: score += 3; why.append("ROE >15%")
        elif roe > .10: score += 1
    if roa is not None and roa > .06:
        score += 1
    if de is not None:
        if de < 80: score += 2; why.append("Debt/equity controlled")
        elif de > 200: risk.append("High debt/equity")
    if margin is not None and margin > .08:
        score += 1
    if pe is not None and pe > 120:
        risk.append("Very high PE")
    elif pe is not None and pe > 0:
        score += 1
    return min(score, 15), why, risk

# Technical = 65, relative/market = 15, fundamental = 20 => 100
def score_stock(d, nifty20, sector20, mode):
    x = d.iloc[-1]
    tech = 0
    why, risk = [], []

    # Trend 20
    if x["Close"] > x["EMA20"]: tech += 4; why.append("Above EMA20")
    if x["EMA20"] > x["EMA50"]: tech += 5; why.append("EMA20 > EMA50")
    if x["EMA50"] > x["EMA200"]: tech += 6; why.append("EMA50 > EMA200")
    if x["Close"] > x["EMA200"]: tech += 5; why.append("Above EMA200")

    # Momentum 15
    if 55 <= x["RSI"] <= 75: tech += 7; why.append("RSI 55–75")
    elif 50 <= x["RSI"] < 55: tech += 3
    if x["MACD"] > x["MACDSignal"]: tech += 5; why.append("MACD bullish")
    if x["Ret20"] > 0: tech += 3; why.append("20D momentum positive")

    # Structure / breakout 20
    if x["Close"] > x["Pivot20"]:
        tech += 10; why.append("20D breakout")
    elif x["Close"] >= .985 * x["Pivot20"]:
        tech += 3; why.append("Near breakout")
    if x["Range20"] <= 12: tech += 6; why.append("Tight base")
    elif x["Range20"] <= 18: tech += 3
    if x["Close"] > x["Pivot55"]:
        tech += 4; why.append("55D breakout")

    # Volume 10
    if x["VolX"] >= 2: tech += 10; why.append("Volume ≥2x")
    elif x["VolX"] >= 1.5: tech += 7; why.append("Volume ≥1.5x")
    elif x["VolX"] >= 1.2: tech += 3

    # Relative strength 15
    rel_nifty = x["Ret20"] - nifty20
    rel_sector = x["Ret20"] - sector20
    rel = (rel_nifty + rel_sector) / 2
    if rel >= 8: rel_score = 15; why.append("Strong vs market & sector")
    elif rel >= 4: rel_score = 11; why.append("Outperforming market/sector")
    elif rel > 0: rel_score = 6
    else: rel_score = 0
    if mode == "Positional" and x["Ret60"] > 0:
        rel_score = min(15, rel_score + 2)
        why.append("60D trend positive")

    if x["RSI"] > 78: risk.append("Overbought")
    if x["Close"] > x["EMA20"] + 2.5 * x["ATR"]: risk.append("Extended")
    if x["VolX"] < 1: risk.append("Weak volume")
    if x["Close"] < x["EMA50"]: risk.append("Below EMA50")

    return min(65, tech), min(15, rel_score), why, risk

def trade_plan(d, capital, risk_pct):
    x = d.iloc[-1]
    entry = float(x["Close"])
    atr = float(x["ATR"])
    pivot = float(x["Pivot20"])
    base_low = float(d["Low"].tail(10).min())

    stop = min(base_low, entry - 1.5 * atr)
    if stop >= entry:
        stop = entry - 1.5 * atr

    risk_share = max(entry - stop, .01)
    t1 = entry + 2 * risk_share
    t2 = entry + 3 * risk_share
    risk_cash = capital * risk_pct / 100
    qty = int(risk_cash / risk_share)
    qty = min(qty, int(capital / entry))
    return entry, pivot, stop, t1, t2, qty, qty * risk_share

def decision(row, market_regime):
    score = row["Score"]
    breakout = row["Breakout"] == "YES"
    risk = row["RiskFlag"] != "None"
    if market_regime == "🔴 Defensive":
        if score >= 85 and breakout and not risk:
            return "🟡 WATCH / WAIT"
        return "🔴 AVOID"
    if score >= 85 and breakout and not risk:
        return "🟢 BUY CANDIDATE"
    if score >= 75:
        return "🟢 WATCH"
    if score >= 65:
        return "🟡 WAIT"
    return "🔴 AVOID"

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ V3 Settings")
    mode = st.radio("Trading style", ["Swing", "Positional"])
    capital = st.number_input("💰 Capital ₹", 5000, 10000000, 50000, 5000)
    risk_pct = st.slider("🛡️ Risk per trade %", .25, 3.0, 1.0, .25)

    st.divider()
    min_volume = st.slider("🔊 Minimum volume x20D", 1.0, 3.0, 1.5, .1)
    min_score = st.slider("⭐ Minimum score", 50, 90, 70, 5)
    top_n = st.slider("🏆 Shortlist", 5, 30, 15)

    st.divider()
    strict_trend = st.checkbox("Strict EMA trend", True)
    breakout_only = st.checkbox("Breakout only", False)
    avoid_extended = st.checkbox("Avoid extended", True)

    st.divider()
    st.caption("Universe: complete Nifty 500. No Nifty 200 exclusion.")

# ---------------------------------------------------------
# Market Dashboard
# ---------------------------------------------------------
st.subheader("🌍 Step 1 — Market Decision")

mkt = index_snapshot(INDEX_TICKERS)
nifty20 = 0.0
vix = None
if not mkt.empty:
    nr = mkt[mkt["Index"] == "Nifty 50"]
    if not nr.empty: nifty20 = float(nr.iloc[0]["20D"])
    vr = mkt[mkt["Index"] == "India VIX"]
    if not vr.empty: vix = float(vr.iloc[0]["Last"])

if nifty20 > 2:
    regime = "🟢 Favourable"
    market_regime = "🟢 Favourable"
elif nifty20 < -5:
    regime = "🔴 Defensive"
    market_regime = "🔴 Defensive"
else:
    regime = "🟡 Selective"
    market_regime = "🟡 Selective"

cards = st.columns(min(6, max(1, len(mkt))))
for col, (_, r) in zip(cards, mkt.head(6).iterrows()):
    col.metric(r["Index"], f"{r['Last']:,.2f}", f"{r['20D']:+.1f}% 20D")

st.info(
    f"**Market regime: {regime}** | Nifty 50 20D: {nifty20:+.1f}% | "
    f"India VIX: {'NA' if vix is None else f'{vix:.2f}'}"
)

# ---------------------------------------------------------
# Sector Dashboard
# ---------------------------------------------------------
st.subheader("🔥 Step 2 — Sector Leadership")

sec = index_snapshot(SECTOR_TICKERS)
if not sec.empty:
    sec = sec.sort_values(["20D", "5D"], ascending=False).reset_index(drop=True)
    sec["Rank"] = np.arange(1, len(sec) + 1)
    st.dataframe(
        sec[["Rank", "Index", "5D", "20D", "60D", "Above50", "Above200"]]
        .rename(columns={"Index": "Sector"}),
        use_container_width=True,
        hide_index=True
    )
    top_sector = str(sec.iloc[0]["Index"])
    top_sector_20 = float(sec.iloc[0]["20D"])
else:
    top_sector = "Unknown"
    top_sector_20 = nifty20

# ---------------------------------------------------------
# Universe
# ---------------------------------------------------------
try:
    universe = get_nifty500()
except Exception:
    st.error("Nifty 500 list could not be loaded automatically.")
    manual = st.text_area(
        "Manual NSE symbols",
        "TATAMOTORS,HINDALCO,PFC,POWERGRID,COFORGE,CHOLAFIN"
    )
    universe = pd.DataFrame({
        "Symbol": [x.strip().upper() for x in manual.split(",") if x.strip()]
    })

u1, u2, u3 = st.columns(3)
u1.metric("Nifty 500", len(universe))
u2.metric("Top sector", top_sector)
u3.metric("Top sector 20D", f"{top_sector_20:+.1f}%")

# ---------------------------------------------------------
# Run scanner
# ---------------------------------------------------------
run = st.button("🚀 RUN V3 FULL NIFTY 500 SCREENER", type="primary", use_container_width=True)

if run:
    symbols = universe["Symbol"].tolist()
    with st.spinner(f"Downloading daily history for {len(symbols)} Nifty 500 stocks..."):
        raw = download_prices(symbols)

    rows = []
    charts = {}
    progress = st.progress(0)

    for i, s in enumerate(symbols):
        try:
            d0 = extract(raw, s)
            if d0 is None or len(d0) < 210:
                progress.progress((i + 1) / len(symbols))
                continue
            d = add_indicators(d0).dropna(
                subset=["EMA200", "RSI", "ATR", "VolX", "Pivot20"]
            )
            if len(d) < 30:
                progress.progress((i + 1) / len(symbols))
                continue

            x = d.iloc[-1]
            trend = x["Close"] > x["EMA20"] > x["EMA50"] > x["EMA200"]
            breakout = x["Close"] > x["Pivot20"]
            volume = x["VolX"] >= min_volume
            extended = x["Close"] > x["EMA20"] + 2.5 * x["ATR"]

            if strict_trend and not trend:
                progress.progress((i + 1) / len(symbols)); continue
            if breakout_only and not breakout:
                progress.progress((i + 1) / len(symbols)); continue
            if not volume:
                progress.progress((i + 1) / len(symbols)); continue
            if avoid_extended and extended:
                progress.progress((i + 1) / len(symbols)); continue

            tech, rel, why, risks = score_stock(
                d, nifty20, top_sector_20, mode
            )

            rows.append({
                "Symbol": s,
                "Technical": tech,
                "Relative": rel,
                "Close": float(x["Close"]),
                "RSI": float(x["RSI"]),
                "VolX": float(x["VolX"]),
                "Pivot": float(x["Pivot20"]),
                "Ret20": float(x["Ret20"]),
                "Ret60": float(x["Ret60"]),
                "EMA20": float(x["EMA20"]),
                "EMA50": float(x["EMA50"]),
                "EMA200": float(x["EMA200"]),
                "Range20": float(x["Range20"]),
                "Breakout": "YES" if breakout else "WATCH",
                "Why": " • ".join(why),
                "RiskFlag": " • ".join(risks) if risks else "None",
            })
            charts[s] = d
        except Exception:
            pass
        progress.progress((i + 1) / len(symbols))

    progress.empty()

    if not rows:
        st.error(
            "No stocks passed the current filters. Reduce minimum volume/score "
            "or switch off strict trend/breakout."
        )
        st.stop()

    pre = pd.DataFrame(rows).sort_values(
        ["Technical", "Relative", "VolX"],
        ascending=False
    ).head(60)

    # Fundamental layer is intentionally limited to the strongest 60
    # to keep a free cloud deployment practical.
    with st.spinner("Checking fundamentals for the strongest candidates..."):
        fdata = get_fundamentals(pre["Symbol"].tolist())

    final = []
    for _, r in pre.iterrows():
        f = fdata.get(r["Symbol"], {})
        fs, fwhy, frisk = fundamental_score(f)
        score = int(round(r["Technical"] + r["Relative"] + fs))

        entry, pivot, sl, t1, t2, qty, risk_cash = trade_plan(
            charts[r["Symbol"]], capital, risk_pct
        )

        final.append({
            **r.to_dict(),
            "Fundamental": fs,
            "Score": min(100, score),
            "Sector": f.get("sector", "NA"),
            "Industry": f.get("industry", "NA"),
            "Revenue Growth": fnum(f.get("revenueGrowth")),
            "Earnings Growth": fnum(f.get("earningsGrowth")),
            "ROE": fnum(f.get("returnOnEquity")),
            "Debt/Equity": fnum(f.get("debtToEquity")),
            "PE": fnum(f.get("trailingPE")),
            "FWhy": " • ".join(fwhy) if fwhy else "Limited fundamental signal",
            "FRisk": " • ".join(frisk) if frisk else "None",
            "Entry": entry,
            "SL": sl,
            "T1": t1,
            "T2": t2,
            "SL%": (entry - sl) / entry * 100,
            "Qty": qty,
            "Risk₹": risk_cash,
        })

    result = pd.DataFrame(final)
    result = result[result["Score"] >= min_score].copy()

    result["Decision"] = result.apply(
        lambda x: decision(x, market_regime), axis=1
    )
    result = result.sort_values(
        ["Score", "Relative", "VolX"], ascending=False
    ).reset_index(drop=True)

    st.session_state["v3_result"] = result
    st.session_state["v3_charts"] = charts

# ---------------------------------------------------------
# Results
# ---------------------------------------------------------
if "v3_result" not in st.session_state:
    st.subheader("🧭 V3 workflow")
    st.markdown("""
**Use the app in this order:**

**🌍 Market → 🔥 Sector → 🏆 Stock → 📈 Setup → 🛑 Risk → 🎯 Decision**

The purpose is to stop beginners from selecting a stock only because one indicator says BUY.
""")
    st.stop()

res = st.session_state["v3_result"]
if res.empty:
    st.warning("No qualifying Nifty 500 setup at the current settings.")
    st.stop()

# ---------------------------------------------------------
# Top decision
# ---------------------------------------------------------
top = res.iloc[0]
a, b, c, d, e, f = st.columns(6)
a.metric("🏆 Top", top["Symbol"])
b.metric("⭐ Score", f"{int(top['Score'])}/100")
c.metric("📈 RSI", f"{top['RSI']:.1f}")
d.metric("🔊 Vol", f"{top['VolX']:.2f}x")
e.metric("🎯 Breakout", top["Breakout"])
f.metric("🤖 Decision", top["Decision"])

st.subheader("🏆 Top Nifty 500 Opportunities")

display = res[[
    "Symbol", "Score", "Decision", "Sector", "Industry",
    "Close", "RSI", "VolX", "Ret20", "Ret60",
    "Breakout", "Entry", "SL", "T1", "T2", "Qty"
]].head(top_n).copy()

for c in ["Close", "RSI", "VolX", "Ret20", "Ret60", "Entry", "SL", "T1", "T2"]:
    display[c] = display[c].round(2)

st.dataframe(display, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# Sector among shortlisted stocks
# ---------------------------------------------------------
st.subheader("🔥 Sector Concentration in Shortlist")

sector_rank = (
    res.groupby("Sector")
    .agg(
        Stocks=("Symbol", "count"),
        AvgScore=("Score", "mean"),
        Avg20D=("Ret20", "mean")
    )
    .sort_values(["AvgScore", "Avg20D"], ascending=False)
    .reset_index()
)
sector_rank["AvgScore"] = sector_rank["AvgScore"].round(1)
sector_rank["Avg20D"] = sector_rank["Avg20D"].round(1)
st.dataframe(sector_rank.head(10), use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# Trade planner
# ---------------------------------------------------------
st.subheader("🎯 Step 3 — Beginner Trade Planner")

selected = st.selectbox(
    "Select stock",
    res.head(top_n)["Symbol"].tolist()
)
r = res[res["Symbol"] == selected].iloc[0]

p = st.columns(7)
p[0].metric("Decision", r["Decision"])
p[1].metric("Score", int(r["Score"]))
p[2].metric("Entry", f"₹{r['Entry']:,.2f}")
p[3].metric("Pivot", f"₹{r['Pivot']:,.2f}")
p[4].metric("Stop", f"₹{r['SL']:,.2f}")
p[5].metric("2R", f"₹{r['T1']:,.2f}")
p[6].metric("Qty", int(r["Qty"]))

st.write(f"**Technical:** {r['Why']}")
st.write(f"**Fundamental:** {r['FWhy']}")
st.write(f"**Risk flags:** {r['RiskFlag']} | Fundamental: {r['FRisk']}")

st.info(
    f"With ₹{capital:,.0f} capital and {risk_pct:.2f}% planned risk, "
    f"the calculated quantity risks about ₹{r['Risk₹']:,.0f} if the reference stop is hit. "
    "If the quantity is 0, skip the trade rather than increasing risk."
)

# ---------------------------------------------------------
# Chart
# ---------------------------------------------------------
charts = st.session_state["v3_charts"]
if selected in charts:
    d = charts[selected].tail(180)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"],
        low=d["Low"], close=d["Close"], name="Price"
    ))
    for col in ["EMA20", "EMA50", "EMA200"]:
        fig.add_trace(go.Scatter(x=d.index, y=d[col], name=col))
    fig.add_trace(go.Scatter(
        x=d.index, y=d["Pivot20"], name="20D Pivot",
        line=dict(dash="dot")
    ))
    fig.update_layout(
        title=f"{selected} — Trend + Breakout",
        height=560,
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# Fundamental panel
# ---------------------------------------------------------
st.subheader("🏢 Step 4 — Fundamental Quality")

fund_cols = st.columns(6)
fund_items = [
    ("Revenue Growth", r["Revenue Growth"], True),
    ("Earnings Growth", r["Earnings Growth"], True),
    ("ROE", r["ROE"], True),
    ("Debt/Equity", r["Debt/Equity"], False),
    ("PE", r["PE"], False),
    ("Fundamental Score", r["Fundamental"], False),
]
for col, (label, value, pct) in zip(fund_cols, fund_items):
    if value is None or pd.isna(value):
        text = "NA"
    elif pct:
        text = f"{value * 100:.1f}%"
    else:
        text = f"{value:.1f}"
    col.metric(label, text)

# ---------------------------------------------------------
# Decision rules
# ---------------------------------------------------------
st.subheader("🤖 Final Decision Rules")

st.markdown("""
| Decision | Meaning |
|---|---|
| 🟢 **BUY CANDIDATE** | High score + confirmed breakout + no major risk flag, in a non-defensive market |
| 🟢 **WATCH** | Strong setup but one important confirmation is still missing |
| 🟡 **WAIT** | Setup is developing; do not chase |
| 🟡 **WATCH / WAIT** | Market is defensive; preserve capital |
| 🔴 **AVOID** | Weak score or unacceptable setup/risk |

**Important:** “BUY CANDIDATE” is not an instruction to place an order. It means the stock has passed the app's predefined checklist.
""")

# ---------------------------------------------------------
# Checklist
# ---------------------------------------------------------
st.subheader("✅ Before You Trade")

for item in [
    "Market regime is favourable or selective, not strongly defensive",
    "Sector is showing relative strength",
    "Stock score meets my minimum",
    "Price trend is aligned with EMA20/50/200",
    "Breakout has volume confirmation",
    "Stock is not excessively extended",
    "Entry and stop are decided before the order",
    "Position size fits my risk limit",
    "I checked results/news/corporate actions",
    "If setup invalidates, I will exit instead of averaging blindly",
]:
    st.checkbox(item, key=f"{selected}_{item}")

# ---------------------------------------------------------
# Export
# ---------------------------------------------------------
st.subheader("📥 Export V3")

csv = res.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Nifty 500 V3 Watchlist",
    csv,
    file_name=f"Nifty500_AI_V3_{datetime.now().strftime('%Y%m%d')}.csv",
    mime="text/csv",
    use_container_width=True,
)

st.divider()
st.caption(
    "Prototype data source: Yahoo Finance. Nifty 500 universe reference is based on "
    "the broad-market index; verify current constituents and live market values with NSE."
)
