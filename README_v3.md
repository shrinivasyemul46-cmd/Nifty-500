Nifty 500 AI Swing & Positional Trading Assistant
Clean Streamlit Cloud version for research and decision support.
Workflow
Market → Sector → Nifty 500 → Technical → Fundamental → Risk → Decision
Market
Nifty 50
India VIX
S&P 500
Nasdaq
Nikkei
Hang Seng
Technical
Close > EMA20 > EMA50 > EMA200
RSI 55–75 preferred
MACD
20D momentum
20D / 55D breakout
Tight 20D base
Volume expansion
Relative strength
Fundamental
Top candidates are checked for:
Revenue growth
Earnings growth
ROE / ROA
Debt/equity
PE
Risk planner
Enter capital and risk %. The app calculates a reference:
Entry
Stop loss
2R target
3R target
Quantity
Approximate rupee risk
Score
100-point ranking:
Technical + relative strength: up to 80
Fundamental: up to 20
The score is not a probability of profit and does not guarantee returns.
GitHub repository
Keep the deployment repository simple:
app.py
requirements.txt
README.md
Replace the old files with these exact three files. Do not point Streamlit to app_v3.py; the main file is app.py.
Streamlit Community Cloud
Open your GitHub repository.
Replace app.py.
Replace requirements.txt.
Replace README.md.
Commit changes.
Open your Streamlit app and use Reboot app / redeploy.
Main file: app.py.
If NSE blocks the constituent download
The app shows an upload box. Download the current Nifty 500 constituent CSV from the official NSE Nifty 500 page and upload it. The CSV must contain a Symbol column.
Data
Nifty 500 universe: official NSE constituent CSV.
Prototype market data: Yahoo Finance.
Data may be delayed or temporarily unavailable. Verify live NSE/broker prices, volume, liquidity, results, corporate actions and news before trading.
Safety
This is a research/decision-support tool. It does not place orders and does not guarantee profit.
