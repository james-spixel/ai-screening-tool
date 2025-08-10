
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="AI Company Screening (FSS / FPS / VS)", layout="wide")

# ==============================
# Model helpers (bundled inline)
# ==============================

WEIGHTS_FSS = {"roic": 12, "fcf_stability": 8, "interest_coverage": 7, "gross_margin_stability": 8, "debt_to_equity": 6, "altman_z": 5, "blackford_spread": 4}
WEIGHTS_FPS = {"revenue_growth_quality": 8, "market_share_trend": 7, "rnd_intensity": 6, "earnings_reinvestment_rate": 5, "insider_alignment": 4}
WEIGHTS_VS  = {"fcf_yield": 8, "earnings_yield": 7, "sales_yield": 5}

REQUIRED_COLUMNS = [
    "ticker","company_name","sector","tax_rate","interest_expense","total_debt","net_income","shareholders_equity",
    "ebit","operating_cash_flow","capital_expenditures","nopat","market_cap","enterprise_value","sales",
    "retained_earnings","working_capital","total_assets","insider_ownership_pct","insider_net_buy_6m","cash_and_equivalents"
]
TIME_SERIES_COLUMNS = [
    "revenue_series","ebit_series","gross_margin_series","market_share_series","operating_cash_flow_series",
    "capex_series","retained_earnings_series","rnd_series","gross_profit_series"
]

def _parse_series(s):
    if s is None or (isinstance(s, float) and np.isnan(s)): 
        return None
    try:
        arr = np.array([float(x) for x in str(s).split(";") if str(x).strip()], dtype=float)
        return arr if arr.size > 0 else None
    except Exception:
        return None

def _stability(series):
    if series is None or len(series) < 3: 
        return None
    deltas = np.diff(series) / np.maximum(np.abs(series[:-1]), 1e-9)
    return 1.0 / (1.0 + np.std(deltas))

def _trend_slope(series):
    if series is None or len(series) < 2: 
        return None
    y = np.asarray(series, dtype=float)
    x = np.arange(len(y))
    x_mean = x.mean()
    denom = (x ** 2).sum() - len(x) * (x_mean ** 2)
    if denom == 0: 
        return None
    return ((x * y).sum() - len(x) * x_mean * y.mean()) / denom

def _pct_cagr(series):
    if series is None or len(series) < 2: 
        return None
    start, end = series[0], series[-1]
    if start <= 0 or end <= 0: 
        return None
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

def _calculate_series_metrics(row: pd.Series) -> pd.Series:
    s = {col: _parse_series(row.get(col)) for col in TIME_SERIES_COLUMNS}
    ocf, capex = s.get("operating_cash_flow_series"), s.get("capex_series")
    fcf_series = (ocf - capex) if ocf is not None and capex is not None and len(ocf)==len(capex) else None
    row['fcf_stability'] = _stability(fcf_series)
    row['gross_margin_stability'] = _stability(s.get("gross_margin_series"))
    rev_cagr = _pct_cagr(s.get("revenue_series"))
    gp_trend = _trend_slope(s.get("gross_profit_series"))
    row['revenue_growth_quality'] = rev_cagr if (rev_cagr is not None and gp_trend is not None and gp_trend >= 0) else 0.0
    row['market_share_trend'] = _trend_slope(s.get("market_share_series"))
    re_s = s.get("retained_earnings_series")
    if re_s is not None and len(re_s) >= 2 and row.get('net_income', 0) != 0:
        row['earnings_reinvestment_rate'] = (re_s[-1]-re_s[-2]) / abs(row['net_income'])
    else:
        row['earnings_reinvestment_rate'] = 0.0
    rnd_s, rev_s = s.get("rnd_series"), s.get("revenue_series")
    if rnd_s is not None and rev_s is not None and len(rnd_s)==len(rev_s) and rev_s[-1]>0:
        row['rnd_intensity'] = rnd_s[-1] / rev_s[-1]
    else:
        row['rnd_intensity'] = 0.0
    return row

def _score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    scores = pd.DataFrame(index=df.index)
    def peer_rank_score(metric, weight, higher_is_better=True):
        rank = df.groupby('sector')[metric].rank(pct=True)
        if not higher_is_better: rank = 1.0 - rank
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
        fl = []
        if row.get('altman_z', 3) < 1.8: fl.append("Altman Z Distress")
        if row.get('interest_coverage', 3) < 1.5: fl.append("Low Interest Coverage")
        if row.get('fcf', 1) < 0: fl.append("Negative Free Cash Flow")
        if row.get('revenue_growth_quality', 0.1) < 0: fl.append("Negative Revenue Growth")
        if row.get('debt_to_equity', 1) > 2.5 and row['sector'] != 'Financial Services': fl.append("High D/E")
        if row.get('insider_net_buy_6m', 1) < 0: fl.append("Net Insider Selling")
        flags.append(", ".join(fl) if fl else "None")
    df['Red_Flags'] = flags
    return df

def screen_companies(df: pd.DataFrame) -> pd.DataFrame:
    df_calc = _calculate_raw_metrics(df.copy())
    df_series = df_calc.apply(_calculate_series_metrics, axis=1)
    df_scored = _score_dataframe(df_series)
    df_flagged = _identify_red_flags(df_scored)
    df_flagged['Classification'] = df_flagged['TotalScore'].apply(lambda s:
        "Tier 1: Potential Leader" if s>=75 else ("Tier 2: Strong Contender" if s>=60 else ("Tier 3: Needs Further Review" if s>=45 else "Tier 4: High Risk / Low Score"))
    )
    out_cols = ["ticker","company_name","sector","FSS","FPS","VS","TotalScore","Classification","Red_Flags"]
    return df_flagged[out_cols].sort_values("TotalScore", ascending=False).reset_index(drop=True)

# ==============================
# UI
# ==============================

st.title("AI Company Screening — FSS / FPS / VS")
st.caption("Upload your CSV → Get ranked companies with strengths, valuation, and red flags.")

with st.expander("📋 Required columns (copy these into your CSV)"):
    st.code(", ".join(REQUIRED_COLUMNS + TIME_SERIES_COLUMNS), language="text")

uploaded = st.file_uploader("Upload your CSV", type=["csv"])

def _demo_df():
    return pd.DataFrame({
        'ticker': ['AIPC','STBL','GRTH'],
        'company_name': ['AI Power Corp','Stable Industrial','Growth Bio'],
        'sector': ['Technology','Industrials','Healthcare'],
        'tax_rate': [0.21,0.25,0.18],
        'interest_expense': [50,100,20],
        'total_debt': [2000,5000,1000],
        'net_income': [400,200,50],
        'shareholders_equity': [3000,8000,1200],
        'ebit': [500,350,80],
        'operating_cash_flow': [600,400,100],
        'capital_expenditures': [150,250,120],
        'nopat': [450,300,70],
        'market_cap': [20000,7500,5000],
        'enterprise_value': [21950,12400,5950],
        'sales': [5000,10000,800],
        'retained_earnings': [1800,5200,160],
        'working_capital': [300,1000,-50],
        'total_assets': [5500,14000,2700],
        'insider_ownership_pct': [0.10,0.02,0.15],
        'insider_net_buy_6m': [50000,-10000,100000],
        'cash_and_equivalents': [50,100,50],
        'revenue_series': ["3500;4000;4500;5000;5500","9800;9900;10000;10100;10050","200;300;450;600;800"],
        'ebit_series': ["350;400;450;500;550","340;345;350;355;352","-10;10;30;50;80"],
        'gross_margin_series': ["0.60;0.62;0.63;0.65;0.66","0.30;0.31;0.30;0.31;0.31","0.70;0.68;0.65;0.63;0.61"],
        'market_share_series': ["0.10;0.11;0.12;0.13;0.14","0.40;0.40;0.39;0.39;0.38","0.02;0.03;0.04;0.05;0.06"],
        'operating_cash_flow_series': ["400;450;500;600;700","380;390;400;410;400","50;60;80;90;100"],
        'capex_series': ["100;110;120;130;150","240;245;250;255;250","80;90;100;110;120"],
        'retained_earnings_series': ["1000;1200;1400;1600;1800","5000;5050;5100;5150;5200","50;60;80;110;160"],
        'rnd_series': ["350;400;450;500;550","50;50;50;50;50","80;120;180;240;320"],
        'gross_profit_series': ["2100;2480;2835;3250;3630","2940;3069;3000;3131;3115","140;204;292;378;488"],
    })

col1, col2 = st.columns([1,1])
with col1:
    use_demo = st.button("Load small demo dataset")

df_in = None
if uploaded is not None:
    df_in = pd.read_csv(uploaded)
elif use_demo:
    df_in = _demo_df()

if df_in is not None:
    # Basic validation
    missing = [c for c in REQUIRED_COLUMNS if c not in df_in.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        st.stop()

    with st.spinner("Scoring companies..."):
        ranked = screen_companies(df_in)

    st.success(f"Scored {len(ranked)} companies.")
    # Filters
    filt_col1, filt_col2, filt_col3 = st.columns([1.2,1,1])
    with filt_col1:
        sectors = sorted(ranked['sector'].unique().tolist())
        pick_sectors = st.multiselect("Filter by sector", sectors, default=sectors)
    with filt_col2:
        classes = ranked['Classification'].unique().tolist()
        pick_classes = st.multiselect("Filter by classification", classes, default=classes)
    with filt_col3:
        min_score = st.slider("Minimum TotalScore", 0.0, 100.0, 0.0, 1.0)

    view = ranked[
        ranked['sector'].isin(pick_sectors) &
        ranked['Classification'].isin(pick_classes) &
        (ranked['TotalScore'] >= min_score)
    ].copy()

    st.dataframe(view, use_container_width=True)
    st.download_button("Download ranked CSV", data=view.to_csv(index=False), file_name="ranked_companies.csv", mime="text/csv")

else:
    st.info("Upload a CSV with the required columns, or click **Load small demo dataset**.")
