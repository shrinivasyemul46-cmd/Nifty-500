# AI Market Decision System V3 — Nifty 500

## Main objective

A beginner-friendly decision-support system for Indian swing and positional trading.

## Universe

**Complete Nifty 500**

Unlike V1/V2, V3 does NOT exclude Nifty 200.

The Nifty 500 is a broad-market index covering large-, mid- and small-cap segments. NSE Indices describes it as the top 500 companies from the eligible universe based on full market capitalisation and average daily turnover.

## Workflow

### 1. Market
- Nifty 50
- India VIX
- Global index proxies where available
- Simple market regime: favourable / selective / defensive

### 2. Sector
- Sector index momentum
- 5D / 20D / 60D
- Above 50/200 EMA
- Shortlist sector leadership

### 3. Stock
- Trend: Close > EMA20 > EMA50 > EMA200
- RSI 55–75 preferred
- MACD
- 20D and 55D breakout
- Tight 20D structure
- Volume expansion
- Relative strength vs market/sector

### 4. Fundamental
- Revenue growth
- Earnings growth
- ROE
- ROA
- Debt/equity
- Profit margin
- PE warning

### 5. Risk
- ATR/base-low stop reference
- 2R / 3R targets
- Position sizing from capital and risk %
- Rupee risk displayed

### 6. Decision
- BUY CANDIDATE
- WATCH
- WAIT
- WATCH / WAIT in defensive market
- AVOID

## Score

- Technical: 65
- Relative strength / market: 15
- Fundamentals: 20
- Total: 100

The score is a ranking/checklist score, NOT a probability of profit.

## Deployment

Repository root:
- app.py
- requirements.txt
- README.md

Streamlit Community Cloud:
- Repository = your GitHub repository
- Branch = main
- Main file = app.py

## Data

Prototype uses Yahoo Finance. Data can be delayed, incomplete or unavailable. Verify live NSE/broker data before execution.

## Safety

This application does not place orders and does not guarantee returns. It is intended as a research and decision-support prototype.
