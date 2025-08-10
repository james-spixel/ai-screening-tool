
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

st.set_page_config(page_title="AI Company Screening — Auto-Fetch", layout="wide")
st.title("AI Company Screening — Auto-Fetch (FSS / FPS / VS)")
st.caption("Type tickers → fetch fundamentals automatically → rank companies.")

# ----- Scoring weights (same as original) -----
WEIGHTS_FSS = {"roic": 12, "fcf_stability": 8, "interest_coverage": 7, "gross_margin_stability": 8, "debt_to_equity": 6, "altman_z": 5, "blackford_spread": 4}
WEIGHTS_FPS = {"revenue_growth_quality": 8, "market_share_trend": 7, "rnd_intensity": 6, "earnings_reinvestment_rate": 5, "insider_alignment": 4}
WEIGHTS_VS  = {"fcf_yield": 8, "earnings_yield": 7, "sales_yield": 5}

# ----- Helper functions copied from original logic -----
def _parse_series(s):
    if s is None or (isinstance(s, float) and np.isnan(s)): 
        return None
    try:
        arr = np.array([float(x) for x in str(s).split(";") if str(x).strip()], dtype=float)
        return arr if arr.size > 0 else None
    except Exception:
        return None

def _stability(series):
    if series is None or len(series) < 3: return None
    deltas = np.diff(series) / np.maximum(np.abs(series[:-1]), 1e-9)
    return 1.0 / (1.0 + np.std(deltas))

def _trend_slope(series):
    if series is None or len(series) < 2: return None
    y = np.asarray(series, dtype=float)
    x = np.arange(len(y))
    x_mean = x.mean()
    denom = (x ** 2).sum() - len(x) * (x_mean ** 2)
    if denom == 0: return None
    return ((x * y).sum() - len(x) * x_mean * y.mean()) / denom

def _pct_cagr(series):
    if series is None or len(series) < 2: return None
    start, end = series[0], series[-1]
    if start <= 0 or end <= 0: return None
    return (end / start) ** (1 / (len(series) - 1)) - 1

def _calculate_raw_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['debt_to_equity'] = df['total_debt'] / df['shareholders_equity'].replace(0, np.nan)
    df['interest_coverage'] = df['ebit'] / df['interest_expense'].replace(0, 1e-9)
    df['roic'] = df['nopat'] / (df['total_debt'] + df['shareholders_equity']).replace(0, np.nan)
    df['cost_of_debt_pretax'] = df['interest_expense'] / df['total_debt'].replace(0, np.nan)
    df['roe'] = df['net_income'] / df['shareholders_equity'].replace(0, np.nan)
    df['blackford_spread'] = df['roe'] - df['cost_of_debt_pretax']
    X1 = df['working_capital'] / df['total_assets']
    X2 = df['retained_earnings'] / df['total_assets']
    X3 = df['ebit'] / df['total_assets']
    X4 = df['market_cap'] / df['total_debt'].replace(0, np.nan)
    X5 = df['sales'] / df['total_assets']
    df['altman_z'] = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    df['fcf'] = df['operating_cash_flow'] - df['capital_expenditures']
    df['fcf_yield'] = df['fcf'] / df['enterprise_value'].replace(0, np.nan)
    df['earnings_yield'] = df['ebit'] / df['enterprise_value'].replace(0, np.nan)
    df['sales_yield'] = df['sales'] / df['enterprise_value'].replace(0, np.nan)
    return df

def _score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    scores = pd.DataFrame(index=df.index)
    def peer_rank_score(metric_col: str, weight: float, higher_is_better: bool = True):
        rank = df.groupby('sector')[metric_col].rank(pct=True)
        if not higher_is_better:
            rank = 1.0 - rank
        return rank.fillna(0.5) * weight
    # FSS
    scores['fss_roic'] = peer_rank_score('roic', WEIGHTS_FSS['roic'])
    scores['fss_fcf_stability'] = peer_rank_score('fcf_stability', WEIGHTS_FSS['fcf_stability'])
    scores['fss_interest_coverage'] = peer_rank_score('interest_coverage', WEIGHTS_FSS['interest_coverage'])
    scores['fss_gross_margin_stability'] = peer_rank_score('gross_margin_stability', WEIGHTS_FSS['gross_margin_stability'])
    scores['fss_debt_to_equity'] = peer_rank_score('debt_to_equity', WEIGHTS_FSS['debt_to_equity'], higher_is_better=False)
    scores['fss_altman_z'] = peer_rank_score('altman_z', WEIGHTS_FSS['altman_z'])
    scores['fss_blackford_spread'] = peer_rank_score('blackford_spread', WEIGHTS_FSS['blackford_spread'])
    # FPS
    scores['fps_revenue_growth_quality'] = peer_rank_score('revenue_growth_quality', WEIGHTS_FPS['revenue_growth_quality'])
    scores['fps_market_share_trend'] = peer_rank_score('market_share_trend', WEIGHTS_FPS['market_share_trend'])
    scores['fps_rnd_intensity'] = peer_rank_score('rnd_intensity', WEIGHTS_FPS['rnd_intensity'])
    scores['fps_earnings_reinvestment_rate'] = peer_rank_score('earnings_reinvestment_rate', WEIGHTS_FPS['earnings_reinvestment_rate'])
    scores['fps_insider_alignment'] = peer_rank_score('insider_ownership_pct', WEIGHTS_FPS['insider_alignment'])
    # VS
    scores['vs_fcf_yield'] = peer_rank_score('fcf_yield', WEIGHTS_VS['fcf_yield'])
    scores['vs_earnings_yield'] = peer_rank_score('earnings_yield', WEIGHTS_VS['earnings_yield'])
    scores['vs_sales_yield'] = peer_rank_score('sales_yield', WEIGHTS_VS['sales_yield'])
    df['FSS'] = scores.filter(like='fss_').sum(axis=1)
    df['FPS'] = scores.filter(like='fps_').sum(axis=1)
    df['VS']  = scores.filter(like='vs_').sum(axis=1)
    df['TotalScore'] = df['FSS'] + df['FPS'] + df['VS']
    return df

def _identify_red_flags(df: pd.DataFrame) -> pd.DataFrame:
    flags = []
    for _, row in df.iterrows():
        row_flags = []
        if row.get('altman_z', 3) < 1.8: row_flags.append("Altman Z Distress")
        if row.get('interest_coverage', 3) < 1.5: row_flags.append("Low Interest Coverage")
        if row.get('fcf', 1) < 0: row_flags.append("Negative Free Cash Flow")
        if row.get('revenue_growth_quality', 0.1) < 0: row_flags.append("Negative Revenue Growth")
        if row.get('debt_to_equity', 1) > 2.5 and row['sector'] != 'Financial Services':
            row_flags.append("High D/E")
        if row.get('insider_net_buy_6m', 1) < 0: row_flags.append("Net Insider Selling")
        flags.append(", ".join(row_flags) if row_flags else "None")
    df['Red_Flags'] = flags
    return df

def classify_company(total_score: float) -> str:
    if total_score >= 75: return "Tier 1: Potential Leader"
    elif total_score >= 60: return "Tier 2: Strong Contender"
    elif total_score >= 45: return "Tier 3: Needs Further Review"
    else: return "Tier 4: High Risk / Low Score"

def _series_from_is(is_df, key):
    try:
        s = is_df.loc[key].dropna()
        # Return oldest -> latest
        return ";".join([str(float(x)) for x in s.iloc[::-1].values])
    except Exception:
        return ""

def _series_from_bs(bs_df, key):
    try:
        s = bs_df.loc[key].dropna()
        return ";".join([str(float(x)) for x in s.iloc[::-1].values])
    except Exception:
        return ""

def _series_from_cf(cf_df, key):
    try:
        s = cf_df.loc[key].dropna()
        return ";".join([str(float(x)) for x in s.iloc[::-1].values])
    except Exception:
        return ""

def fetch_one(ticker):
    t = yf.Ticker(ticker)
    info = t.info or {}
    # Financial statements (annual)
    is_df = t.financials if t.financials is not None else pd.DataFrame()
    bs_df = t.balance_sheet if t.balance_sheet is not None else pd.DataFrame()
    cf_df = t.cashflow if t.cashflow is not None else pd.DataFrame()

    def get_is(key):
        try: return float(is_df.loc[key].iloc[0])
        except: return np.nan

    def get_bs(key):
        try: return float(bs_df.loc[key].iloc[0])
        except: return np.nan

    def get_cf(key):
        try: return float(cf_df.loc[key].iloc[0])
        except: return np.nan

    # Core values
    sector = info.get("sector") or "Unknown"
    market_cap = info.get("marketCap", np.nan)
    ev = info.get("enterpriseValue", np.nan)
    sales = info.get("totalRevenue", get_is("Total Revenue"))
    ebit = info.get("ebitda")  # yfinance has EBITDA; use IS for EBIT if available
    ebit_is = get_is("Ebit") if "Ebit" in is_df.index else get_is("EBIT")
    if np.isnan(ebit_is) == False:
        ebit = ebit_is

    net_income = get_is("Net Income")
    interest_expense = abs(get_is("Interest Expense"))
    total_debt = info.get("totalDebt", np.nan)
    if np.isnan(total_debt):
        # fallback from BS
        total_debt = get_bs("Total Debt")
        if np.isnan(total_debt):
            total_debt = (get_bs("Short Long Term Debt") or 0) + (get_bs("Long Term Debt") or 0)

    shareholders_equity = get_bs("Total Stockholder Equity")
    operating_cf = get_cf("Total Cash From Operating Activities")
    if np.isnan(operating_cf):
        operating_cf = info.get("operatingCashflow", np.nan)
    capex = abs(get_cf("Capital Expenditures"))
    tax_expense = abs(get_is("Income Tax Expense"))
    income_before_tax = get_is("Income Before Tax")
    tax_rate = np.nan
    if income_before_tax and income_before_tax != 0 and not np.isnan(income_before_tax) and not np.isnan(tax_expense):
        tr = tax_expense / abs(income_before_tax)
        if tr >= 0 and tr <= 1.0:
            tax_rate = tr
    if np.isnan(tax_rate):
        tax_rate = 0.21  # default

    # NOPAT approx
    ebit_val = ebit if (ebit is not None and not np.isnan(ebit)) else get_is("Ebit")
    if ebit_val is None or np.isnan(ebit_val):
        ebit_val = get_is("Operating Income")  # fallback
    nopat = ebit_val * (1 - tax_rate) if ebit_val is not None and not np.isnan(ebit_val) else np.nan

    working_capital = get_bs("Total Current Assets") - get_bs("Total Current Liabilities")
    total_assets = get_bs("Total Assets")
    retained_earnings = get_bs("Retained Earnings")
    cash_eq = get_bs("Cash And Cash Equivalents")

    # Build time-series strings (oldest -> latest)
    revenue_series = _series_from_is(is_df, "Total Revenue")
    ebit_series = _series_from_is(is_df, "Ebit") or _series_from_is(is_df, "EBIT")
    gross_margin_series = ""
    try:
        gp_s = is_df.loc["Gross Profit"].dropna()
        rev_s = is_df.loc["Total Revenue"].dropna()
        # Align and compute GP/Revenue
        n = min(len(gp_s), len(rev_s))
        if n >= 2:
            gm = (gp_s.iloc[:n].values / np.maximum(rev_s.iloc[:n].values, 1e-9))[::-1]
            gross_margin_series = ";".join([str(float(x)) for x in gm])
    except Exception:
        pass

    market_share_series = ""  # Not available via yfinance
    ocf_series = _series_from_cf(cf_df, "Total Cash From Operating Activities")
    capex_series = _series_from_cf(cf_df, "Capital Expenditures")
    re_series = _series_from_bs(bs_df, "Retained Earnings")
    rnd_series = _series_from_is(is_df, "Research Development")
    gp_series = _series_from_is(is_df, "Gross Profit")

    row = {
        "ticker": ticker,
        "company_name": info.get("shortName") or ticker,
        "sector": sector,
        "tax_rate": tax_rate,
        "interest_expense": interest_expense,
        "total_debt": total_debt,
        "net_income": net_income,
        "shareholders_equity": shareholders_equity,
        "ebit": ebit_val,
        "operating_cash_flow": operating_cf,
        "capital_expenditures": capex,
        "nopat": nopat,
        "market_cap": market_cap,
        "enterprise_value": ev,
        "sales": sales,
        "retained_earnings": retained_earnings,
        "working_capital": working_capital,
        "total_assets": total_assets,
        "insider_ownership_pct": np.nan,    # not available via yfinance (leave blank)
        "insider_net_buy_6m": np.nan,       # not available via yfinance
        "cash_and_equivalents": cash_eq,
        # series
        "revenue_series": revenue_series,
        "ebit_series": ebit_series,
        "gross_margin_series": gross_margin_series,
        "market_share_series": market_share_series,
        "operating_cash_flow_series": ocf_series,
        "capex_series": capex_series,
        "retained_earnings_series": re_series,
        "rnd_series": rnd_series,
        "gross_profit_series": gp_series,
    }
    return row

def _calculate_series_metrics(row: pd.Series) -> pd.Series:
    def ps(col): 
        s = row.get(col)
        return _parse_series(s) if isinstance(s, str) else None

    ocf, capex = ps("operating_cash_flow_series"), ps("capex_series")
    fcf_series = (ocf - capex) if ocf is not None and capex is not None and len(ocf)==len(capex) else None
    row['fcf_stability'] = _stability(fcf_series)
    row['gross_margin_stability'] = _stability(ps("gross_margin_series"))
    rev_cagr = _pct_cagr(ps("revenue_series"))
    gp_trend = _trend_slope(ps("gross_profit_series"))
    row['revenue_growth_quality'] = rev_cagr if (rev_cagr is not None and gp_trend is not None and gp_trend >= 0) else 0.0
    row['market_share_trend'] = _trend_slope(ps("market_share_series"))
    re_s = ps("retained_earnings_series")
    if re_s is not None and len(re_s) >= 2 and row.get('net_income', 0) != 0 and not pd.isna(row.get('net_income', np.nan)):
        row['earnings_reinvestment_rate'] = (re_s[-1]-re_s[-2]) / abs(row['net_income'])
    else:
        row['earnings_reinvestment_rate'] = 0.0
    rnd_s, rev_s = ps("rnd_series"), ps("revenue_series")
    if rnd_s is not None and rev_s is not None and len(rnd_s)==len(rev_s) and rev_s[-1]>0:
        row['rnd_intensity'] = rnd_s[-1] / rev_s[-1]
    else:
        row['rnd_intensity'] = 0.0
    return row

def screen_companies(df: pd.DataFrame) -> pd.DataFrame:
    df_calc = _calculate_raw_metrics(df.copy())
    df_series = df_calc.apply(_calculate_series_metrics, axis=1)
    df_scored = _score_dataframe(df_series)
    df_flagged = _identify_red_flags(df_scored)
    df_flagged['Classification'] = df_flagged['TotalScore'].apply(classify_company)
    out_cols = ["ticker","company_name","sector","FSS","FPS","VS","TotalScore","Classification","Red_Flags"]
    return df_flagged[out_cols].sort_values("TotalScore", ascending=False).reset_index(drop=True)

# ----- UI -----
st.subheader("Fetch tickers")
tickers_text = st.text_input("Enter tickers separated by commas (e.g., AAPL, MSFT, GOOGL)")
go = st.button("Fetch & Rank")

if go and tickers_text.strip():
    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
    rows = []
    with st.spinner("Fetching data..."):
        for tk in tickers:
            try:
                rows.append(fetch_one(tk))
            except Exception as e:
                st.warning(f"{tk}: could not fetch ({e})")
    if len(rows)==0:
        st.error("No data fetched. Check tickers and try again.")
        st.stop()
    df_in = pd.DataFrame(rows)

    # Fill missing required columns with NaN if needed (so the model runs)
    required_cols = [
        "ticker","company_name","sector","tax_rate","interest_expense","total_debt","net_income","shareholders_equity",
        "ebit","operating_cash_flow","capital_expenditures","nopat","market_cap","enterprise_value","sales",
        "retained_earnings","working_capital","total_assets","insider_ownership_pct","insider_net_buy_6m","cash_and_equivalents",
        "revenue_series","ebit_series","gross_margin_series","market_share_series","operating_cash_flow_series","capex_series",
        "retained_earnings_series","rnd_series","gross_profit_series"
    ]
    for c in required_cols:
        if c not in df_in.columns:
            df_in[c] = np.nan

    ranked = screen_companies(df_in)
    st.success(f"Scored {len(ranked)} companies.")
    st.dataframe(ranked, use_container_width=True)
    st.download_button("Download ranked CSV", data=ranked.to_csv(index=False), file_name="ranked_companies_auto.csv", mime="text/csv")

st.info("Tip: If a field isn't available from Yahoo Finance (e.g., insider metrics, market share), it's left blank — the model still runs.")
