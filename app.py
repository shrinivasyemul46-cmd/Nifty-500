"""
NIFTY 500 SWING 200 — VERSION 2
Professional research dashboard:
Market regime → sector leadership → stock quality → volume breakout → risk plan.

Data source: Yahoo Finance via yfinance.
This app is a research/screening tool. It does not place trades and does not
promise returns or probabilities of success.
"""

import io
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# -------------------- PAGE --------------------
st.set_page_config(
    page_title="NIFTY 500 Swing 200 V2",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container{padding-top:1rem;padding-bottom:2rem}
.hero{padding:20px;border-radius:20px;background:linear-gradient(135deg,#111827,#374151);color:#fff;margin-bottom:15px}
.hero h1{margin:0;font-size:2rem}.hero p{margin:5px 0 0;opacity:.82}
.card{padding:14px;border:1px solid rgba(128,128,128,.25);border-radius:16px;background:rgba(128,128,128,.04);height:100%}
.small{font-size:.82rem;opacity:.75}
</style>
""", unsafe_allow_html=True)

NIFTY500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"

# A small fallback only for cases where the NSE constituent file cannot be read.
FALLBACK = [
"RELIANCE","HDFCBANK","ICICIBANK","BHARTIARTL","TCS","INFY","SBIN",
"BAJFINANCE","HINDUNILVR","ITC","LT","AXISBANK","KOTAKBANK","M&M",
"SUNPHARMA","MARUTI","TITAN","HCLTECH","ADANIENT","NTPC","POWERGRID",
"TATASTEEL","COALINDIA","ONGC","WIPRO","ULTRACEMCO","ASIANPAINT",
"BAJAJFINSV","NESTLEIND","JSWSTEEL","TATAMOTORS","HINDALCO","ADANIPORTS",
"BEL","TRENT","HAL","DLF","SIEMENS"
]

# -------------------- UTILITIES --------------------
def classify_sector(text):
    x = str(text).lower()
    rules = {
        "Financial Services":["bank","finance","financial","insurance","nbfc","capital market","housing finance"],
        "IT & Technology":["software","it services","information technology","computer","technology"],
        "Pharma & Healthcare":["pharma","pharmaceutical","healthcare","hospital","diagnostic","biotech","drug"],
        "Automobile":["automobile","auto","vehicle","tyre","tractor","two wheeler"],
        "Oil Gas & Energy":["oil","gas","petroleum","refinery","power","energy","coal","utility"],
        "Metals & Mining":["metal","steel","aluminium","copper","mining","mineral"],
        "Consumer & FMCG":["consumer","fmcg","food","beverage","retail","personal care","household"],
        "Capital Goods":["engineering","capital goods","industrial","machinery","electrical"],
        "Telecom":["telecom","telecommunication"],
        "Cement & Building":["cement","building material","construction material"],
        "Chemicals":["chemical","fertilizer","agro chemical"],
        "Realty":["real estate","realty"],
        "Media":["media","entertainment","broadcasting"],
    }
    for name, words in rules.items():
        if any(w in x for w in words):
            return name
    return "Other"

def normalize_ohlcv(x):
    if x is None or x.empty:
        return pd.DataFrame()
    x = x.copy()
    if isinstance(x.columns, pd.MultiIndex):
        l0 = list(x.columns.get_level_values(0))
        if set(["Open","High","Low","Close","Volume"]).issubset(l0):
            x.columns = x.columns.get_level_values(0)
    cols = ["Open","High","Low","Close","Volume"]
    if not all(c in x.columns for c in cols):
        return pd.DataFrame()
    return x[cols].dropna(subset=["Close"])

def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/n, adjust=False).mean()
    al = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - 100/(1+rs)

def atr(high, low, close, n=14):
    prev = close.shift(1)
    tr = pd.concat([high-low, (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def macd(close):
    fast = close.ewm(span=12, adjust=False).mean()
    slow = close.ewm(span=26, adjust=False).mean()
    line = fast - slow
    signal = line.ewm(span=9, adjust=False).mean()
    hist = line - signal
    return line, signal, hist

# -------------------- UNIVERSE --------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_universe():
    try:
        req = urllib.request.Request(NIFTY500_URL, headers={"User-Agent":"Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=15).read().decode("utf-8")
        d = pd.read_csv(io.StringIO(raw))
        sc = next(c for c in d.columns if str(c).lower() == "symbol")
        ic = next((c for c in d.columns if str(c).lower() in
                   {"industry","industry name","sector"}), None)
        u = pd.DataFrame({"Symbol": d[sc].astype(str).str.upper().str.replace(".NS","",regex=False)})
        u["Industry"] = d[ic].fillna("Other").astype(str) if ic else "Other"
        u["Sector"] = u["Industry"].map(classify_sector)
        u["Ticker"] = u["Symbol"] + ".NS"
        return u.drop_duplicates("Symbol").reset_index(drop=True), True
    except Exception:
        u = pd.DataFrame({"Symbol":FALLBACK,"Industry":"Other","Sector":"Other"})
        u["Ticker"] = u["Symbol"] + ".NS"
        return u, False

# -------------------- DATA --------------------
@st.cache_data(ttl=900, show_spinner=False)
def download_prices(tickers):
    tickers = list(tickers)
    if not tickers:
        return {}
    try:
        raw = yf.download(
            tickers, period="1y", interval="1d", auto_adjust=False,
            progress=False, threads=True, group_by="ticker"
        )
    except Exception:
        return {}
    if raw.empty:
        return {}
    out = {}
    if len(tickers) == 1:
        x = normalize_ohlcv(raw)
        if not x.empty:
            out[tickers[0]] = x
        return out
    for t in tickers:
        try:
            if t in raw.columns.get_level_values(0):
                x = raw[t]
            elif t in raw.columns.get_level_values(1):
                x = raw.xs(t, axis=1, level=1)
            else:
                continue
            x = normalize_ohlcv(x)
            if not x.empty:
                out[t] = x
        except Exception:
            pass
    return out

@st.cache_data(ttl=900, show_spinner=False)
def get_benchmarks():
    result = {}
    for name, ticker in {
        "NIFTY 50":"^NSEI",
        "BANK NIFTY":"^NSEBANK",
    }.items():
        try:
            x = yf.download(ticker, period="1y", interval="1d", auto_adjust=False, progress=False)
            x = normalize_ohlcv(x)
            if not x.empty:
                result[name] = x
        except Exception:
            pass
    return result

# -------------------- STOCK ENGINE --------------------
def analyze_stock(symbol, sector, x, nifty):
    if x.empty or len(x) < 220:
        return None

    c=x.Close.astype(float); h=x.High.astype(float); l=x.Low.astype(float); v=x.Volume.astype(float)
    e20=c.ewm(span=20,adjust=False).mean()
    e50=c.ewm(span=50,adjust=False).mean()
    e200=c.ewm(span=200,adjust=False).mean()
    rr=rsi(c); aa=atr(h,l,c); av20=v.rolling(20).mean()
    ml, ms, mh=macd(c)

    p=float(c.iloc[-1]); e20n=float(e20.iloc[-1]); e50n=float(e50.iloc[-1]); e200n=float(e200.iloc[-1])
    rsin=float(rr.iloc[-1]); atrn=float(aa.iloc[-1])
    vr=float(v.iloc[-1]/av20.iloc[-1]) if av20.iloc[-1]>0 else np.nan
    macd_hist=float(mh.iloc[-1])

    ret5=(p/float(c.iloc[-6])-1)*100
    ret20=(p/float(c.iloc[-21])-1)*100
    ret60=(p/float(c.iloc[-61])-1)*100

    pivot=float(h.iloc[-21:-1].max())
    low20=float(l.iloc[-20:].min())
    high60=float(h.iloc[-61:-1].max())
    range20=(float(h.iloc[-20:].max())-float(l.iloc[-20:].min()))/p*100

    trend_score=(8 if p>e20n else 0)+(6 if e20n>e50n else 0)+(6 if e50n>e200n else 0)
    momentum_score=15 if 55<=rsin<=75 else (8 if 50<=rsin<55 else (2 if rsin>75 else 0))
    macd_score=8 if macd_hist>0 else 0
    volume_score=25 if 2<=vr<=5 else (22 if vr>5 else (12 if vr>=1.5 else (6 if vr>=1.2 else 0)))
    structure_score=8 if p>=pivot*.98 else 0
    breakout=p>pivot and vr>=1.5
    breakout_score=15 if breakout else 0
    momentum_score += 5 if ret20>0 else 0
    momentum_score += 5 if ret60>0 else 0
    score=int(min(100,trend_score+momentum_score+macd_score+volume_score+structure_score+breakout_score))

    # Relative strength against NIFTY 50.
    rs_vs_nifty=np.nan
    if not nifty.empty and len(nifty)>=21:
        nc=nifty.Close.astype(float)
        nret=(float(nc.iloc[-1])/float(nc.iloc[-21])-1)*100
        rs_vs_nifty=ret20-nret

    # Setup quality is stricter than raw score.
    setup_flags = {
        "Trend": p>e20n>e50n>e200n,
        "RSI": 55<=rsin<=75,
        "MACD": macd_hist>0,
        "Volume": vr>=2,
        "Near Pivot": p>=pivot*.98,
        "Breakout": breakout,
        "RS": (not np.isnan(rs_vs_nifty) and rs_vs_nifty>0),
    }
    confirmations=sum(bool(x) for x in setup_flags.values())

    if score>=85 and confirmations>=5 and vr>=2:
        grade="🔥 A+ SWING"
    elif score>=75 and confirmations>=4:
        grade="🟢 A SWING"
    elif score>=65:
        grade="🟡 B WATCH"
    else:
        grade="🔵 DEVELOPING"

    # Risk model: 1.5 ATR or 20-day low, whichever gives the tighter logical stop.
    stop=max(low20,p-1.5*atrn)
    if stop>=p: stop=p*.97
    risk=p-stop
    t1=p+2*risk
    t2=p+3*risk

    zone="🔥 2x–5x PREFERRED" if 2<=vr<=5 else ("🚀 >5x EXTREME" if vr>5 else ("🟡 1.5x–2x" if vr>=1.5 else "⚪ Normal"))

    return {
        "Symbol":symbol,"Sector":sector,"Price":p,
        "5D %":ret5,"20D %":ret20,"60D %":ret60,
        "RSI":rsin,"MACD Hist":macd_hist,"Volume x":vr,"Volume Zone":zone,
        "EMA20":e20n,"EMA50":e50n,"EMA200":e200n,
        "Trend":"BULLISH" if p>e20n>e50n>e200n else "MIXED",
        "Pivot":pivot,"60D High":high60,"20D Range %":range20,
        "Breakout":"YES" if breakout else "NO",
        "RS vs Nifty":rs_vs_nifty,"Confirmations":confirmations,
        "Entry":p,"Stop Loss":stop,"Target 1":t1,"Target 2":t2,"R:R":"1:2 / 1:3",
        "Score":score,"Grade":grade,
        "Trend OK":setup_flags["Trend"],"RSI OK":setup_flags["RSI"],
        "MACD OK":setup_flags["MACD"],"Volume OK":setup_flags["Volume"],
        "Near Pivot":setup_flags["Near Pivot"],"Breakout OK":setup_flags["Breakout"],
        "RS OK":setup_flags["RS"],
    }

# -------------------- MARKET ENGINE --------------------
def benchmark_metrics(benchmarks):
    rows=[]
    for name,x in benchmarks.items():
        if len(x)<61: continue
        c=x.Close.astype(float)
        e20=c.ewm(span=20,adjust=False).mean()
        e50=c.ewm(span=50,adjust=False).mean()
        rows.append({
            "Index":name,
            "Close":float(c.iloc[-1]),
            "5D %":(float(c.iloc[-1])/float(c.iloc[-6])-1)*100,
            "20D %":(float(c.iloc[-1])/float(c.iloc[-21])-1)*100,
            "60D %":(float(c.iloc[-1])/float(c.iloc[-61])-1)*100,
            "Trend":"BULLISH" if c.iloc[-1]>e20.iloc[-1]>e50.iloc[-1] else "MIXED"
        })
    return pd.DataFrame(rows)

def market_strength(d, benchmarks):
    if d.empty: return 0
    breadth=(d.Price>d.EMA20).mean()*100
    trend=(d.Trend=="BULLISH").mean()*100
    momentum=(d.RSI>=50).mean()*100
    volume=(d["Volume x"]>=1).mean()*100
    nifty20=0
    if "NIFTY 50" in benchmarks and len(benchmarks["NIFTY 50"])>=21:
        c=benchmarks["NIFTY 50"].Close.astype(float)
        nifty20=(float(c.iloc[-1])/float(c.iloc[-21])-1)*100
    return int(round(.35*breadth+.25*trend+.15*momentum+.10*volume+.15*max(0,min(100,50+nifty20*10))))

def sector_table(d):
    z=d[d.Sector!="Other"].copy()
    if z.empty:return pd.DataFrame()
    rows=[]
    for sec,g in z.groupby("Sector"):
        ret=g["20D %"].mean()
        breadth=(g.Price>g.EMA20).mean()*100
        trend=(g.Trend=="BULLISH").mean()*100
        vol=g["Volume x"].mean()
        rs=g["RS vs Nifty"].mean()
        score=.30*max(0,min(100,50+ret*5))+.25*breadth+.20*trend+.15*max(0,min(100,vol*50))+.10*max(0,min(100,50+(0 if np.isnan(rs) else rs*5)))
        rows.append({
            "Sector":sec,"Sector Score":score,"20D Return":ret,
            "Breadth":breadth,"Trend Breadth":trend,"Avg Volume x":vol,
            "RS vs Nifty":rs,"≥2x Stocks":int((g["Volume x"]>=2).sum()),
            "Breakouts":int((g.Breakout=="YES").sum()),"Stocks":len(g)
        })
    return pd.DataFrame(rows).sort_values("Sector Score",ascending=False)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("🎛️ V2 Controls")
    min_score=st.slider("🎯 Minimum score",50,95,70,5)
    min_confirm=st.slider("✅ Minimum confirmations",2,7,4,1)
    sector_count=st.slider("🔥 Top sectors",3,8,3,1)
    result_count=st.slider("🏆 Swing results",25,200,200,25)

    st.divider()
    st.subheader("🔥 Volume")
    high_volume_only=st.checkbox("Require ≥2x volume",True)
    include_extreme=st.checkbox("Include >5x extreme",True)

    st.divider()
    st.subheader("💰 Risk")
    capital=st.number_input("Trading capital (₹)",min_value=5000.0,value=50000.0,step=5000.0)
    risk_pct=st.number_input("Risk per trade (%)",min_value=0.25,max_value=5.0,value=1.0,step=0.25)
    max_positions=st.number_input("Max positions",min_value=1,max_value=20,value=5,step=1)

    st.divider()
    if st.button("🔄 Refresh market data",use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# -------------------- HEADER --------------------
st.markdown("""
<div class="hero">
<h1>🚀 NIFTY 500 → SWING 200 V2</h1>
<p>Market regime → sector leadership → A+ setup → volume confirmation → risk-based position sizing</p>
</div>
""",unsafe_allow_html=True)

u,live=get_universe()
with st.spinner(f"📡 Scanning NIFTY 500 ({len(u)} stocks)..."):
    prices=download_prices(tuple(u.Ticker.tolist()))
benchmarks=get_benchmarks()

rows=[]
for _,q in u.iterrows():
    if q.Ticker in prices:
        z=analyze_stock(q.Symbol,q.Sector,prices[q.Ticker],benchmarks.get("NIFTY 50",pd.DataFrame()))
        if z: rows.append(z)
d=pd.DataFrame(rows)

if d.empty:
    st.error("No usable market data returned. Tap Refresh market data and try again.")
    st.stop()

# -------------------- MARKET REGIME --------------------
ms=market_strength(d,benchmarks)
if ms>=80: regime="🟢 VERY STRONG"; advice="Broad participation is strong. Prefer leaders and confirmed breakouts."
elif ms>=65: regime="🟢 STRONG"; advice="Prefer top sectors, relative-strength leaders and volume confirmation."
elif ms>=50: regime="🟡 NEUTRAL"; advice="Be selective; reduce chasing and wait for confirmation."
elif ms>=35: regime="🟠 WEAK"; advice="Favour only exceptional setups; avoid aggressive exposure."
else: regime="🔴 VERY WEAK"; advice="Defensive mode. Wait for breadth and index trend to improve."

st.subheader("🌐 1. Market Strength & Regime")
a,b,c,e,f=st.columns(5)
a.metric("Strength",f"{ms}/100"); a.progress(ms/100)
b.metric("Above EMA20",f'{(d.Price>d.EMA20).mean()*100:.0f}%')
c.metric("Bullish Trend",f'{(d.Trend=="BULLISH").mean()*100:.0f}%')
e.metric("≥2x Volume",int((d["Volume x"]>=2).sum()))
f.metric("Regime",regime)
st.info("💡 "+advice)

bm=benchmark_metrics(benchmarks)
if not bm.empty:
    st.dataframe(bm.round(2),use_container_width=True,hide_index=True)

# -------------------- SECTORS --------------------
st.subheader(f"🔥 2. Top {sector_count} Trending Sectors")
s=sector_table(d)
if not s.empty:
    cards=st.columns(min(3,sector_count))
    for i,(_,q) in enumerate(s.head(sector_count).iterrows()):
        with cards[i%len(cards)]:
            st.markdown('<div class="card">',unsafe_allow_html=True)
            st.markdown(f"### {i+1}. 🏭 {q.Sector}")
            st.metric("Sector Score",f"{q['Sector Score']:.0f}/100")
            st.write(f"📈 20D **{q['20D Return']:.1f}%**")
            st.write(f"🟢 Breadth **{q.Breadth:.0f}%** • RS **{q['RS vs Nifty']:.1f}**")
            st.write(f"🔥 ≥2x **{int(q['≥2x Stocks'])}** • 💥 Breakouts **{int(q.Breakouts)}**")
            st.markdown("</div>",unsafe_allow_html=True)
    st.dataframe(s.head(sector_count).round(1),use_container_width=True,hide_index=True)

# -------------------- SECTOR LEADERS --------------------
st.subheader("🏆 3. Best 5 Stocks in Each Top Sector")
top_sector_names=s.head(sector_count).Sector.tolist()
for i,sec in enumerate(top_sector_names,1):
    x=d[d.Sector==sec].copy()
    x=x.sort_values(["Score","Confirmations","Volume x"],ascending=False).head(5)
    st.markdown(f"#### {i}. 🏭 {sec}")
    st.dataframe(
        x[["Symbol","Price","20D %","RSI","MACD Hist","Volume x","Trend","Breakout","RS vs Nifty","Score","Grade"]].round(2),
        use_container_width=True,hide_index=True
    )

# -------------------- A+ SETUPS --------------------
st.subheader("💎 4. A+ Swing Setups")
a_plus=d[(d.Score>=85)&(d.Confirmations>=5)&(d["Volume x"]>=2)].copy()
if not include_extreme:
    a_plus=a_plus[a_plus["Volume x"]<=5]
a_plus=a_plus.sort_values(["Score","Volume x","RS vs Nifty"],ascending=False)

if a_plus.empty:
    st.warning("No A+ setup currently meets all strict filters. This is intentional: the app should be selective.")
else:
    st.dataframe(
        a_plus[["Symbol","Sector","Price","20D %","RSI","Volume x","Volume Zone",
                "Trend","Breakout","RS vs Nifty","Confirmations","Score","Grade"]].round(2),
        use_container_width=True,hide_index=True
    )

# -------------------- SWING 200 --------------------
st.subheader("🎯 5. BEST SWING 200")
sw=d[(d.Score>=min_score)&(d.Confirmations>=min_confirm)].copy()
if high_volume_only:
    sw=sw[sw["Volume x"]>=2]
    if not include_extreme:
        sw=sw[sw["Volume x"]<=5]
sw=sw.sort_values(["Score","Confirmations","Volume x","RS vs Nifty"],ascending=False).head(result_count)

x1,x2,x3=st.columns(3)
x1.metric("Stocks analysed",len(d))
x2.metric("Preferred 2x–5x",int(((d["Volume x"]>=2)&(d["Volume x"]<=5)).sum()))
x3.metric("Extreme >5x",int((d["Volume x"]>5).sum()))

if sw.empty:
    st.warning("No stocks match the current V2 filters. Lower the score/confirmation filter or allow lower volume.")
else:
    st.dataframe(
        sw[["Symbol","Sector","Price","20D %","60D %","RSI","Volume x","Volume Zone",
            "Trend","Breakout","RS vs Nifty","Confirmations","Entry","Stop Loss",
            "Target 1","Target 2","R:R","Score","Grade"]].round(2),
        use_container_width=True,hide_index=True
    )
    st.download_button(
        "⬇️ Download Swing 200 CSV",
        sw.to_csv(index=False).encode("utf-8"),
        f"nifty500_swing200_v2_{datetime.now():%Y%m%d_%H%M}.csv",
        "text/csv"
    )

# -------------------- POSITION SIZING --------------------
st.subheader("💰 6. Risk-Based Position Sizing")
st.caption("Position sizing is based on the selected capital and risk percentage. It is a planning calculation, not a trade recommendation.")

if not sw.empty:
    sizing=sw.head(max_positions).copy()
    risk_amount=capital*risk_pct/100
    sizing["Risk ₹/share"]=(sizing["Entry"]-sizing["Stop Loss"]).clip(lower=0.01)
    sizing["Qty"]=np.floor(risk_amount/sizing["Risk ₹/share"]).astype(int)
    sizing["Position Value ₹"]=sizing["Qty"]*sizing["Entry"]
    sizing["Risk Budget ₹"]=sizing["Qty"]*sizing["Risk ₹/share"]
    st.dataframe(
        sizing[["Symbol","Sector","Entry","Stop Loss","Target 1","Target 2","Risk ₹/share","Qty","Position Value ₹","Risk Budget ₹","Score","Grade"]].round(2),
        use_container_width=True,hide_index=True
    )

# -------------------- DETAIL --------------------
st.subheader("🔬 7. Detailed Stock Decision Card")
sym=st.selectbox("Select a NIFTY 500 stock",sorted(d.Symbol.tolist()))
q=d[d.Symbol==sym].iloc[0]

a,b,c,e,f=st.columns(5)
a.metric("Price",f"₹{q.Price:,.2f}")
b.metric("RSI",f"{q.RSI:.1f}")
c.metric("Volume",f"{q['Volume x']:.2f}x")
e.metric("Score",f"{int(q.Score)}/100")
f.metric("Grade",q.Grade)

left,right=st.columns(2)
with left:
    st.markdown("### 📐 Technical checklist")
    checks=[
        ("Trend",q["Trend OK"]),("RSI 55–75",q["RSI OK"]),("MACD bullish",q["MACD OK"]),
        ("Volume ≥2x",q["Volume OK"]),("Near pivot",q["Near Pivot"]),
        ("Breakout",q["Breakout OK"]),("Relative strength",q["RS OK"])
    ]
    for name,ok in checks:
        st.write(("✅" if ok else "⬜")+f" {name}")
with right:
    st.markdown("### 🛡️ Trade planning")
    st.write(
        f"**Entry:** ₹{q.Entry:,.2f}  \n"
        f"**Stop:** ₹{q['Stop Loss']:,.2f}  \n"
        f"**Target 1:** ₹{q['Target 1']:,.2f}  \n"
        f"**Target 2:** ₹{q['Target 2']:,.2f}  \n"
        f"**R:R:** {q['R:R']}  \n"
        f"**Pivot:** ₹{q.Pivot:,.2f}"
    )

# -------------------- FOOTER --------------------
st.divider()
st.caption(
    f"Universe: {'Live NIFTY 500 list' if live else 'Fallback list'} • "
    f"Usable stocks: {len(d)} • Updated: {datetime.now():%Y-%m-%d %H:%M:%S}"
)
st.caption(
    "⚠️ Educational/research tool only. Data may be delayed, missing or inaccurate. "
    "Volume spikes can be caused by news/events. Verify live price, liquidity, "
    "corporate actions and news before trading. No guaranteed return or probability."
)
