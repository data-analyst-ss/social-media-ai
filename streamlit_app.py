import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle, os, warnings
import streamlit.components.v1 as components
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Social Media AI — Sarah Silva",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400;1,700&family=Inter:wght@200;300;400;500&family=DM+Mono:wght@300;400&display=swap');

/* ─── RESET ─────────────────────────────────────────── */
html, body, [class*="css"], .stApp {
    background-color: #111416 !important;
    color: #e8e0d6 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 300 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ─── GRAIN TEXTURE ──────────────────────────────────── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 999;
    opacity: 0.4;
}

/* ─── SIDEBAR ────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0d1012 !important;
    border-right: 1px solid rgba(212,139,139,0.08) !important;
}
[data-testid="stSidebar"] * { color: #9a8f86 !important; }
[data-testid="stSidebar"] .stSelectbox > label {
    font-family: 'DM Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    color: #5a5046 !important;
}

/* ─── SELECTBOX ──────────────────────────────────────── */
[data-baseweb="select"] > div {
    background: rgba(13,16,18,0.8) !important;
    border: none !important;
    border-bottom: 1px solid rgba(212,139,139,0.15) !important;
    border-radius: 0 !important;
    color: #c8bfb6 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 300 !important;
    transition: border-color 0.3s !important;
}
[data-baseweb="select"] > div:hover {
    border-bottom-color: rgba(212,139,139,0.4) !important;
}

/* ─── METRICS ────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid rgba(212,139,139,0.1) !important;
    border-radius: 0 !important;
    padding: 24px 8px 20px !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 36px !important;
    font-weight: 200 !important;
    color: #e8e0d6 !important;
    letter-spacing: -1px !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    color: #5a5046 !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 11px !important;
    font-weight: 300 !important;
}

/* ─── TABS ───────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(212,139,139,0.1) !important;
    gap: 0 !important;
    padding: 0 40px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    color: #5a5046 !important;
    background: transparent !important;
    border: none !important;
    padding: 16px 24px !important;
    border-bottom: 1px solid transparent !important;
    transition: all 0.3s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #c8a898 !important; }
.stTabs [aria-selected="true"] {
    color: #D48B8B !important;
    border-bottom: 1px solid #D48B8B !important;
    background: transparent !important;
}

/* ─── BUTTONS ────────────────────────────────────────── */
.stButton > button {
    background: transparent !important;
    color: #D48B8B !important;
    border: 1px solid rgba(212,139,139,0.3) !important;
    border-radius: 0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    padding: 14px 32px !important;
    transition: all 0.4s !important;
}
.stButton > button:hover {
    background: rgba(212,139,139,0.06) !important;
    border-color: rgba(212,139,139,0.6) !important;
    color: #e8b8b8 !important;
}

/* ─── SLIDER ─────────────────────────────────────────── */
.stSlider [data-baseweb="slider"] { color: #D48B8B !important; }
.stSlider label {
    font-family: 'DM Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    color: #5a5046 !important;
}

/* ─── NUMBER INPUT ───────────────────────────────────── */
.stNumberInput input {
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid rgba(212,139,139,0.15) !important;
    border-radius: 0 !important;
    color: #e8e0d6 !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 200 !important;
    font-size: 15px !important;
}
.stNumberInput label {
    font-family: 'DM Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    color: #5a5046 !important;
}

/* ─── DATAFRAME ──────────────────────────────────────── */
[data-testid="stDataFrame"] {
    background: transparent !important;
    border: none !important;
    border-top: 1px solid rgba(212,139,139,0.08) !important;
}

/* ─── DIVIDER ────────────────────────────────────────── */
hr { border-color: rgba(212,139,139,0.08) !important; }

/* ─── SUCCESS/WARNING/ERROR ──────────────────────────── */
.stSuccess { 
    background: rgba(80,100,80,0.08) !important; 
    border: none !important;
    border-left: 2px solid #7a9a7a !important;
    border-radius: 0 !important;
}
.stWarning { 
    background: rgba(197,160,89,0.06) !important; 
    border: none !important;
    border-left: 2px solid #C5A059 !important;
    border-radius: 0 !important;
}
.stError { 
    background: rgba(212,139,139,0.06) !important; 
    border: none !important;
    border-left: 2px solid #D48B8B !important;
    border-radius: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── DATA ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('data/global_ads_performance_dataset.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna()
    df['month'] = df['date'].dt.to_period('M').astype(str)
    return df

df = load_data()

COUNTRY_ISO = {
    'UAE':'ARE','UK':'GBR','USA':'USA',
    'Germany':'DEU','Canada':'CAN','India':'IND','Australia':'AUS'
}

# Luxury layout for charts
LAY = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(13,16,18,0)',
    plot_bgcolor='rgba(13,16,18,0)',
    font=dict(family='Inter', color='#8a8078', size=11),
    title_font=dict(family='Playfair Display', size=16, color='#e8e0d6'),
    margin=dict(l=8, r=8, t=48, b=16),
    legend=dict(
        bgcolor='rgba(13,16,18,0.6)',
        bordercolor='rgba(212,139,139,0.1)',
        borderwidth=1,
        font=dict(size=10, family='DM Mono')
    ),
    xaxis=dict(
        gridcolor='rgba(212,139,139,0.05)',
        linecolor='rgba(212,139,139,0.08)',
        tickfont=dict(size=10, family='DM Mono'),
        zeroline=False
    ),
    yaxis=dict(
        gridcolor='rgba(212,139,139,0.05)',
        linecolor='rgba(212,139,139,0.08)',
        tickfont=dict(size=10, family='DM Mono'),
        zeroline=False
    ),
)

ROSE   = '#D48B8B'
GOLD   = '#C5A059'
SAGE   = '#7a9a7a'
ROSE2  = '#c87878'
MUTED  = '#8a7878'

PLAT_COLORS = {
    'TikTok Ads': '#D48B8B',
    'Meta Ads':   '#C5A059',
    'Google Ads': '#7a9a7a'
}

# ── SIDEBAR ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:32px 20px 24px;border-bottom:1px solid rgba(212,139,139,0.08)'>
        <div style='font-family:DM Mono,monospace;font-size:8px;color:#4a4038;letter-spacing:5px;text-transform:uppercase;margin-bottom:12px'>◈ Social Media AI</div>
        <div style='font-family:Playfair Display,serif;font-size:20px;font-weight:700;color:#e8e0d6;line-height:1.2;margin-bottom:6px'>Intelligence<br>Dashboard</div>
        <div style='font-family:Inter,sans-serif;font-size:11px;color:#5a5046;font-weight:300;margin-top:8px'>by Sarah Silva</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="padding:20px 20px 8px"><div style="font-family:DM Mono,monospace;font-size:8px;color:#4a4038;letter-spacing:5px;text-transform:uppercase;margin-bottom:16px">Filter</div>', unsafe_allow_html=True)

    sel_plat = st.selectbox("Platform", ['All'] + sorted(df['platform'].unique().tolist()))
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    sel_ind  = st.selectbox("Industry",  ['All'] + sorted(df['industry'].unique().tolist()))
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    sel_ctry = st.selectbox("Country",   ['All'] + sorted(df['country'].unique().tolist()))
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    sel_camp = st.selectbox("Campaign Type", ['All'] + sorted(df['campaign_type'].unique().tolist()))
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style='padding:20px;border-top:1px solid rgba(212,139,139,0.08)'>
        <div style='font-family:DM Mono,monospace;font-size:8px;color:#4a4038;letter-spacing:5px;text-transform:uppercase;margin-bottom:14px'>About</div>
        <div style='font-family:Inter,sans-serif;font-size:11px;color:#5a5046;font-weight:300;line-height:2'>
            Kaggle · Real Data · 2024<br>
            TikTok · Meta · Google Ads<br>
            7 International Markets<br>
            XGBoost · Isolation Forest
        </div>
    </div>
    <div style='padding:16px 20px;display:flex;flex-direction:column;gap:8px'>
        <a href='https://linkedin.com/in/sarahgleicesilva' target='_blank'
           style='font-family:DM Mono,monospace;font-size:8px;letter-spacing:3px;color:#D48B8B;text-decoration:none;text-transform:uppercase;padding:10px 0;border-bottom:1px solid rgba(212,139,139,0.12);display:block'>
           LinkedIn ↗
        </a>
        <a href='https://github.com/data-analyst-ss' target='_blank'
           style='font-family:DM Mono,monospace;font-size:8px;letter-spacing:3px;color:#C5A059;text-decoration:none;text-transform:uppercase;padding:10px 0;display:block'>
           GitHub ↗
        </a>
    </div>
    """, unsafe_allow_html=True)

# ── FILTER ────────────────────────────────────────────────────
dff = df.copy()
if sel_plat != 'All': dff = dff[dff['platform'] == sel_plat]
if sel_ind  != 'All': dff = dff[dff['industry']  == sel_ind]
if sel_ctry != 'All': dff = dff[dff['country']   == sel_ctry]
if sel_camp != 'All': dff = dff[dff['campaign_type'] == sel_camp]

is_filtered = any(x != 'All' for x in [sel_plat, sel_ind, sel_ctry, sel_camp])

# ── HEADER ────────────────────────────────────────────────────
best_plat = df.groupby('platform')['ROAS'].mean().idxmax()
best_ind  = df.groupby('industry')['ROAS'].mean().idxmax()

_filter_badge = '<div style="font-family:DM Mono,monospace;font-size:8px;color:#C5A059;letter-spacing:3px;text-transform:uppercase;padding:6px 12px;border:1px solid rgba(197,160,89,0.2)">Filter Active</div>' if is_filtered else ''
components.html(f"""
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;1,400&family=Inter:wght@200;300;400&family=DM+Mono:wght@300;400&display=swap" rel="stylesheet">
<div style='background:#0d1012;border-bottom:1px solid rgba(212,139,139,0.08);padding:48px 48px 36px;position:relative;overflow:hidden;font-family:Inter,sans-serif'>
    <div style='position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(212,139,139,0.3),rgba(197,160,89,0.2),transparent)'></div>
    <div style='font-family:DM Mono,monospace;font-size:8px;color:#4a4038;letter-spacing:6px;text-transform:uppercase;margin-bottom:20px'>
        ◈ &nbsp; Social Media Intelligence &nbsp; · &nbsp; Real Data &nbsp; · &nbsp; Machine Learning
    </div>
    <div style='font-family:Playfair Display,serif;font-size:clamp(32px,4vw,56px);font-weight:400;color:#e8e0d6;line-height:1.05;letter-spacing:-0.5px;margin-bottom:16px'>
        The Art of<br><em style='color:#D48B8B;font-style:italic'>Campaign Intelligence</em>
    </div>
    <div style='display:flex;align-items:center;gap:32px;margin-top:20px;flex-wrap:wrap'>
        <div style='font-family:Inter,sans-serif;font-size:12px;color:#5a5046;font-weight:300'>
            {len(dff):,} campaigns &nbsp;&middot;&nbsp; {dff['country'].nunique()} markets &nbsp;&middot;&nbsp; {dff['platform'].nunique()} platforms
        </div>
        {_filter_badge}
    </div>
</div>
""", height=200, scrolling=False)

# ── KPIs ──────────────────────────────────────────────────────
# LUXURY KPI CARDS — using components.html for guaranteed rendering
avg_roas = dff['ROAS'].mean()
avg_roas_all = df['ROAS'].mean()
roas_delta = ((avg_roas - avg_roas_all) / avg_roas_all * 100)
avg_ctr = dff['CTR'].mean()*100
avg_cpc = dff['CPC'].mean()
avg_cpa = dff['CPA'].mean()
total_rev = dff['revenue'].sum()
best_plat_kpi = df.groupby('platform')['ROAS'].mean().idxmax()
delta_sign = '&#8593;' if roas_delta >= 0 else '&#8595;'

_kpi_html = f"""
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;1,400&family=Inter:wght@200;300;400&family=DM+Mono:wght@300;400&display=swap" rel="stylesheet">
<div style='padding:0 48px 32px;background:#111416'>
<div style='font-family:DM Mono,monospace;font-size:8px;color:#4a4038;letter-spacing:6px;text-transform:uppercase;padding:32px 0 8px'>Performance Overview</div>
<div style='display:grid;grid-template-columns:repeat(5,1fr);gap:0;border-top:1px solid rgba(212,139,139,0.1)'>
    <div style='border-right:1px solid rgba(212,139,139,0.08);padding:28px 24px 24px'>
        <div style='font-family:Playfair Display,serif;font-style:italic;font-size:13px;color:#D48B8B;margin-bottom:10px'>Return on Ad Spend</div>
        <div style='font-family:Inter,sans-serif;font-size:52px;font-weight:200;color:#F8F8F8;letter-spacing:-2px;line-height:1'>{avg_roas:.2f}<span style='font-size:28px'>x</span></div>
        <div style='font-family:DM Mono,monospace;font-size:9px;color:#C5A059;text-transform:uppercase;letter-spacing:2px;margin-top:10px'>{delta_sign} {abs(roas_delta):.1f}% vs overall avg</div>
        <div style='font-family:Inter,sans-serif;font-size:10px;color:#4a4038;margin-top:4px'>For every $1 &rarr; ${avg_roas:.2f} back</div>
    </div>
    <div style='border-right:1px solid rgba(212,139,139,0.08);padding:28px 24px 24px'>
        <div style='font-family:Playfair Display,serif;font-style:italic;font-size:13px;color:#D48B8B;margin-bottom:10px'>Click-Through Rate</div>
        <div style='font-family:Inter,sans-serif;font-size:52px;font-weight:200;color:#F8F8F8;letter-spacing:-2px;line-height:1'>{avg_ctr:.2f}<span style='font-size:28px'>%</span></div>
        <div style='font-family:DM Mono,monospace;font-size:9px;color:#C5A059;text-transform:uppercase;letter-spacing:2px;margin-top:10px'>Best: {dff['CTR'].max()*100:.1f}%</div>
        <div style='font-family:Inter,sans-serif;font-size:10px;color:#4a4038;margin-top:4px'>% of viewers who clicked</div>
    </div>
    <div style='border-right:1px solid rgba(212,139,139,0.08);padding:28px 24px 24px'>
        <div style='font-family:Playfair Display,serif;font-style:italic;font-size:13px;color:#D48B8B;margin-bottom:10px'>Cost per Click</div>
        <div style='font-family:Inter,sans-serif;font-size:52px;font-weight:200;color:#F8F8F8;letter-spacing:-2px;line-height:1'>${avg_cpc:.2f}</div>
        <div style='font-family:DM Mono,monospace;font-size:9px;color:#C5A059;text-transform:uppercase;letter-spacing:2px;margin-top:10px'>Lowest: ${dff['CPC'].min():.2f}</div>
        <div style='font-family:Inter,sans-serif;font-size:10px;color:#4a4038;margin-top:4px'>Average cost per click</div>
    </div>
    <div style='border-right:1px solid rgba(212,139,139,0.08);padding:28px 24px 24px'>
        <div style='font-family:Playfair Display,serif;font-style:italic;font-size:13px;color:#D48B8B;margin-bottom:10px'>Cost per Acquisition</div>
        <div style='font-family:Inter,sans-serif;font-size:52px;font-weight:200;color:#F8F8F8;letter-spacing:-2px;line-height:1'>${avg_cpa:.2f}</div>
        <div style='font-family:DM Mono,monospace;font-size:9px;color:#C5A059;text-transform:uppercase;letter-spacing:2px;margin-top:10px'>Lowest: ${dff['CPA'].min():.2f}</div>
        <div style='font-family:Inter,sans-serif;font-size:10px;color:#4a4038;margin-top:4px'>Cost to acquire one customer</div>
    </div>
    <div style='padding:28px 24px 24px'>
        <div style='font-family:Playfair Display,serif;font-style:italic;font-size:13px;color:#D48B8B;margin-bottom:10px'>Total Revenue</div>
        <div style='font-family:Inter,sans-serif;font-size:52px;font-weight:200;color:#F8F8F8;letter-spacing:-2px;line-height:1'>${total_rev/1e6:.2f}<span style='font-size:28px'>M</span></div>
        <div style='font-family:DM Mono,monospace;font-size:9px;color:#C5A059;text-transform:uppercase;letter-spacing:2px;margin-top:10px'>{len(dff):,} campaigns</div>
        <div style='font-family:Inter,sans-serif;font-size:10px;color:#4a4038;margin-top:4px'>Across all platforms</div>
    </div>
</div>
</div>
"""
components.html(_kpi_html, height=230, scrolling=False)

# ── NARRATIVE INSIGHT ─────────────────────────────────────────
if is_filtered:
    _tail = f'Current filter shows <strong style="color:#C5A059;font-weight:400">{avg_roas:.2f}x</strong> average ROAS &mdash; {abs(roas_delta):.1f}% {"above" if roas_delta >= 0 else "below"} the overall dataset average.'
else:
    _tail = f'Overall average ROAS is <strong style="color:#e8e0d6;font-weight:400">{avg_roas:.2f}x</strong> across all {len(dff):,} campaigns.'
_insight_html = f"""
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;1,400&family=Inter:wght@200;300;400&family=DM+Mono:wght@300;400&display=swap" rel="stylesheet">
<div style='padding:0 48px;margin:8px 0 24px;background:#111416'>
    <div style='background:rgba(13,16,18,0.6);border-left:2px solid #D48B8B;padding:16px 24px;font-family:Inter,sans-serif;font-size:12px;color:#8a8078;font-weight:300;line-height:1.8'>
        <span style='color:#D48B8B;font-family:DM Mono,monospace;font-size:8px;letter-spacing:3px;text-transform:uppercase'>Key Insight &nbsp;&middot;&nbsp; </span>
        <strong style='color:#e8e0d6;font-weight:400'>{best_plat}</strong> leads all platforms in ROAS efficiency.
        <strong style='color:#e8e0d6;font-weight:400'>{best_ind}</strong> is the highest-performing industry vertical.
        {_tail}
    </div>
</div>
"""
components.html(_insight_html, height=90, scrolling=False)

# ── TABS ──────────────────────────────────────────────────────
tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "◎  World Map",
    "◎  Performance",
    "◎  Platform & Industry",
    "◎  Anomaly Detection",
    "◎  ROAS Predictor"
])

def section_title(title, subtitle=""):
    _sub = f'<div style="font-family:Inter,sans-serif;font-size:12px;color:#5a5046;font-weight:300">{subtitle}</div>' if subtitle else ''
    st.markdown(f"""
    <div style='margin:32px 0 24px'>
        <div style='font-family:Playfair Display,serif;font-size:24px;font-weight:700;color:#e8e0d6;margin-bottom:6px'>{title}</div>
        {_sub}
    </div>
    """, unsafe_allow_html=True)

def insight_box(text):
    st.markdown(f"""
    <div style='background:rgba(212,139,139,0.04);border-left:2px solid rgba(212,139,139,0.3);padding:14px 20px;font-family:Inter,sans-serif;font-size:11px;color:#7a7068;font-weight:300;line-height:1.7;margin:12px 0 24px'>
        <span style='color:#D48B8B;font-family:DM Mono,monospace;font-size:8px;letter-spacing:2px;text-transform:uppercase'>Insight &nbsp;· &nbsp;</span>{text}
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 1 — WORLD MAP
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div style='padding:8px 48px 32px'>", unsafe_allow_html=True)
    section_title("Global Campaign Intelligence", "Heatmap · 7 international markets · 2024")

    col_metric, _ = st.columns([2,3])
    with col_metric:
        metric_choice = st.selectbox("Show on map:", [
            "Return on Ad Spend (ROAS)",
            "Click-Through Rate (%)",
            "Cost per Acquisition ($)",
            "Total Revenue ($)",
            "Number of Campaigns"
        ])

    ctry = dff.groupby('country').agg(
        avg_roas=('ROAS','mean'),
        avg_ctr=('CTR','mean'),
        avg_cpa=('CPA','mean'),
        total_revenue=('revenue','sum'),
        campaigns=('country','count')
    ).round(3).reset_index()
    ctry['iso'] = ctry['country'].map(COUNTRY_ISO)

    metric_map = {
        "Return on Ad Spend (ROAS)": ("avg_roas","ROAS (×)","Higher = more revenue per dollar spent"),
        "Click-Through Rate (%)": ("avg_ctr","CTR","Higher = more engaging ads"),
        "Cost per Acquisition ($)": ("avg_cpa","CPA ($)","Lower = more efficient customer acquisition"),
        "Total Revenue ($)": ("total_revenue","Revenue ($)","Total revenue generated"),
        "Number of Campaigns": ("campaigns","Campaigns","Volume of campaigns run"),
    }
    col, label, desc = metric_map[metric_choice]

    fig_map = go.Figure(go.Choropleth(
        locations=ctry['iso'], z=ctry[col], text=ctry['country'],
        colorscale=[
            [0.0, '#0d1012'],
            [0.2, '#1a1014'],
            [0.4, '#3d1f1f'],
            [0.6, '#8B4545'],
            [0.8, '#D48B8B'],
            [1.0, '#f0c8c8'],
        ],
        marker_line_color='rgba(212,139,139,0.15)',
        marker_line_width=0.8,
        colorbar=dict(
            title=dict(text=label, font=dict(color='#8a8078', size=9, family='DM Mono')),
            tickfont=dict(color='#5a5046', size=9, family='DM Mono'),
            bgcolor='rgba(13,16,18,0.8)',
            bordercolor='rgba(212,139,139,0.1)',
            borderwidth=1,
            thickness=10,
            len=0.5,
        ),
        hovertemplate='<b style="font-family:Playfair Display">%{text}</b><br>' + label + ': %{z:.2f}<extra></extra>',
    ))
    fig_map.update_geos(
        showframe=False,
        showcoastlines=True, coastlinecolor='rgba(212,139,139,0.1)',
        showland=True, landcolor='rgba(18,22,24,0.95)',
        showocean=True, oceancolor='rgba(10,13,15,0.98)',
        showlakes=False, showrivers=False,
        showcountries=True, countrycolor='rgba(212,139,139,0.08)',
        bgcolor='rgba(0,0,0,0)',
        projection_type='natural earth',
    )
    fig_map.update_layout(
        paper_bgcolor='rgba(13,16,18,0)',
        geo_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#8a8078'),
        title=dict(
            text=f'<span style="font-family:Playfair Display">{metric_choice}</span>',
            font=dict(size=16, color='#e8e0d6', family='Playfair Display')
        ),
        margin=dict(l=0,r=0,t=48,b=0), height=480,
    )
    st.plotly_chart(fig_map, use_container_width=True)
    insight_box(f"<strong style='color:#e8e0d6'>{desc}</strong>. Darker pink = higher {label}. Countries without color had no campaigns in the current selection. Hover for exact values.")

    # Country sparklines
    c1, c2 = st.columns([1,1])
    with c1:
        section_title("Country Ranking", "Sorted by Return on Ad Spend")
        rank = ctry.sort_values('avg_roas', ascending=False)[['country','avg_roas','avg_ctr','avg_cpa','total_revenue','campaigns']].copy()
        rank.columns = ['Country','ROAS (×)','CTR','CPA ($)','Revenue ($)','Campaigns']
        st.dataframe(rank, use_container_width=True, hide_index=True)

    with c2:
        fig_bar = go.Figure()
        ctry_s = ctry.sort_values('avg_roas')
        for i, row in ctry_s.iterrows():
            alpha = 0.4 + 0.6*(row['avg_roas']-ctry_s['avg_roas'].min())/(ctry_s['avg_roas'].max()-ctry_s['avg_roas'].min()+1e-9)
            fig_bar.add_trace(go.Bar(
                x=[row['avg_roas']], y=[row['country']],
                orientation='h', showlegend=False,
                marker_color=f'rgba(212,139,139,{alpha:.2f})',
                hovertemplate=f"<b>{row['country']}</b><br>ROAS: {row['avg_roas']:.2f}×<extra></extra>"
            ))
        fig_bar.add_vline(x=ctry['avg_roas'].mean(), line_dash='dot',
                         line_color='rgba(197,160,89,0.4)', line_width=1)
        fig_bar.update_layout(
            **{**LAY, 'title': 'ROAS by Country', 'barmode':'overlay',
               'yaxis': dict(tickfont=dict(size=11,family='DM Mono'), gridcolor='rgba(0,0,0,0)'),
               'xaxis': dict(title='Return on Ad Spend (×)', gridcolor='rgba(212,139,139,0.05)')}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Radar
    section_title("Multi-Metric Country Radar", "Normalized performance across all dimensions")
    radar = dff.groupby('country').agg(
        roas=('ROAS','mean'), ctr=('CTR','mean'),
        conversions=('conversions','mean'), revenue=('revenue','mean'),
        campaigns=('country','count')
    ).reset_index()
    for c_n in ['roas','ctr','conversions','revenue','campaigns']:
        mn,mx = radar[c_n].min(),radar[c_n].max()
        radar[c_n+'_n'] = (radar[c_n]-mn)/(mx-mn+1e-9)

    cats = ['ROAS','CTR','Conversions','Revenue','Volume']
    palette = ['#D48B8B','#C5A059','#7a9a7a','#a89ab8','#7aabb8','#b8a87a','#b87a8b']
    fig_r = go.Figure()
    for i, row in radar.iterrows():
        v = [row['roas_n'],row['ctr_n'],row['conversions_n'],row['revenue_n'],row['campaigns_n']]
        v_c = v+[v[0]]; c_c = cats+[cats[0]]
        color = palette[i%len(palette)]
        r,g,b = int(color[1:3],16),int(color[3:5],16),int(color[5:7],16)
        fig_r.add_trace(go.Scatterpolar(
            r=v_c, theta=c_c, fill='toself',
            fillcolor=f'rgba({r},{g},{b},0.06)',
            line=dict(color=color, width=1.5),
            name=row['country'],
        ))
    fig_r.update_layout(
        polar=dict(
            bgcolor='rgba(13,16,18,0)',
            radialaxis=dict(visible=True,range=[0,1],gridcolor='rgba(212,139,139,0.08)',tickfont=dict(color='#4a4038',size=8,family='DM Mono'),linecolor='rgba(212,139,139,0.08)'),
            angularaxis=dict(gridcolor='rgba(212,139,139,0.08)',linecolor='rgba(212,139,139,0.08)',tickfont=dict(color='#8a8078',size=10,family='DM Mono'))
        ),
        paper_bgcolor='rgba(13,16,18,0)',
        font=dict(family='Inter',color='#8a8078'),
        title=dict(text='Country Performance Radar',font=dict(size=16,color='#e8e0d6',family='Playfair Display')),
        legend=dict(bgcolor='rgba(13,16,18,0.6)',bordercolor='rgba(212,139,139,0.1)',borderwidth=1,font=dict(size=10,family='DM Mono')),
        margin=dict(l=40,r=40,t=60,b=40), height=480
    )
    st.plotly_chart(fig_r, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 2 — PERFORMANCE
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div style='padding:8px 48px 32px'>", unsafe_allow_html=True)
    section_title("Revenue & Spend Over Time", "Monthly trend · All platforms")

    monthly = dff.groupby(['month','platform']).agg(
        revenue=('revenue','sum'), spend=('ad_spend','sum'), roas=('ROAS','mean')
    ).reset_index()

    fig_t = go.Figure()
    for plat, color in PLAT_COLORS.items():
        sub = monthly[monthly['platform']==plat]
        if len(sub) == 0: continue
        r,g,b = int(color[1:3],16),int(color[3:5],16),int(color[5:7],16)
        fig_t.add_trace(go.Scatter(
            x=sub['month'], y=sub['revenue'], name=plat,
            line=dict(color=color, width=1.5),
            fill='tozeroy', fillcolor=f'rgba({r},{g},{b},0.04)',
            hovertemplate=f'<b>{plat}</b><br>%{{x}}<br>Revenue: $%{{y:,.0f}}<extra></extra>'
        ))
    fig_t.update_layout(**{**LAY,'title':'Monthly Revenue by Platform ($)'})
    st.plotly_chart(fig_t, use_container_width=True)
    insight_box("Each line shows one platform's monthly revenue. The shaded area shows volume. <strong style='color:#e8e0d6'>Higher lines = more revenue generated</strong>. The gap between platforms reveals which one is scaling fastest.")

    c1,c2 = st.columns(2)
    with c1:
        # Range plot (reimagined box plot)
        section_title("ROAS Range by Platform", "Min · Average · Max — reimagined")
        plat_s = dff.groupby('platform')['ROAS'].agg(['min','mean','max','std']).reset_index()
        fig_range = go.Figure()
        for i, (_, row) in enumerate(plat_s.iterrows()):
            color = PLAT_COLORS.get(row['platform'], ROSE)
            r,g,b = int(color[1:3],16),int(color[3:5],16),int(color[5:7],16)
            # Range bar
            fig_range.add_trace(go.Bar(
                name=row['platform'], x=[row['platform']],
                y=[row['max']-row['min']], base=[row['min']],
                marker_color=f'rgba({r},{g},{b},0.15)',
                marker_line_width=0, showlegend=False,
                hoverinfo='skip'
            ))
            # Mean point
            fig_range.add_trace(go.Scatter(
                x=[row['platform']], y=[row['mean']],
                mode='markers+text',
                marker=dict(color=color, size=12, symbol='circle',
                           line=dict(color='rgba(13,16,18,0.8)', width=2)),
                text=[f"{row['mean']:.1f}×"],
                textposition='top center',
                textfont=dict(size=11, color=color, family='Inter'),
                showlegend=False,
                hovertemplate=f"<b>{row['platform']}</b><br>Avg ROAS: {row['mean']:.2f}×<br>Range: {row['min']:.1f}× – {row['max']:.1f}×<extra></extra>"
            ))
        fig_range.update_layout(**{**LAY,
            'title':'ROAS Range — Min to Max',
            'yaxis':dict(title='Return on Ad Spend (×)', gridcolor='rgba(212,139,139,0.05)', tickfont=dict(size=10,family='DM Mono')),
            'xaxis':dict(gridcolor='rgba(0,0,0,0)', tickfont=dict(size=11,family='DM Mono'))
        })
        st.plotly_chart(fig_range, use_container_width=True)
        insight_box("The <strong style='color:#e8e0d6'>glowing dot</strong> = average ROAS. The bar = the full range from worst to best campaign. Wider bar = more variance in results.")

    with c2:
        section_title("Efficiency Map", "Cost per Click vs Return · Bubble = Budget")
        samp = dff.sample(min(500, len(dff)))
        fig_sc = go.Figure()
        for plat, color in PLAT_COLORS.items():
            sub = samp[samp['platform']==plat]
            r,g,b = int(color[1:3],16),int(color[3:5],16),int(color[5:7],16)
            fig_sc.add_trace(go.Scatter(
                x=sub['CPC'], y=sub['ROAS'],
                mode='markers', name=plat,
                marker=dict(
                    color=f'rgba({r},{g},{b},0.5)',
                    size=sub['ad_spend']/sub['ad_spend'].max()*20+4,
                    line=dict(color=f'rgba({r},{g},{b},0.8)', width=0.5)
                ),
                hovertemplate=f'<b>{plat}</b><br>CPC: $%{{x:.2f}}<br>ROAS: %{{y:.2f}}×<br><extra></extra>'
            ))
        # Quadrant labels
        cpc_m, roas_m = samp['CPC'].median(), samp['ROAS'].median()
        fig_sc.add_hline(y=roas_m, line_dash='dot', line_color='rgba(197,160,89,0.2)', line_width=1)
        fig_sc.add_vline(x=cpc_m, line_dash='dot', line_color='rgba(197,160,89,0.2)', line_width=1)
        fig_sc.add_annotation(x=samp['CPC'].min(), y=samp['ROAS'].max()*0.95,
            text="★ Sweet Spot", font=dict(size=9,color='rgba(197,160,89,0.6)',family='DM Mono'),
            showarrow=False, xanchor='left')
        fig_sc.update_layout(**{**LAY,
            'title':'Cost per Click vs Return on Ad Spend',
            'xaxis':dict(title='Cost per Click ($)', gridcolor='rgba(212,139,139,0.05)', tickfont=dict(size=10,family='DM Mono')),
            'yaxis':dict(title='Return on Ad Spend (×)', gridcolor='rgba(212,139,139,0.05)', tickfont=dict(size=10,family='DM Mono'))
        })
        st.plotly_chart(fig_sc, use_container_width=True)
        insight_box("<strong style='color:#e8e0d6'>Best campaigns: top-left</strong> (low cost, high return). The gold dashed lines split into 4 quadrants. The ★ Sweet Spot is where you want your campaigns to live.")
    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 3 — PLATFORM & INDUSTRY
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div style='padding:8px 48px 32px'>", unsafe_allow_html=True)

    # Platform radar
    section_title("Platform Intelligence", "Multi-metric comparison · Normalized scores")
    plat_r = dff.groupby('platform').agg(
        roas=('ROAS','mean'), ctr=('CTR','mean'), cpc=('CPC','mean'),
        conversions=('conversions','mean'), revenue=('revenue','mean')
    ).reset_index()
    for c_n in ['roas','ctr','conversions','revenue']:
        mn,mx = plat_r[c_n].min(),plat_r[c_n].max()
        plat_r[c_n+'_n'] = (plat_r[c_n]-mn)/(mx-mn+1e-9)
    mn,mx = plat_r['cpc'].min(),plat_r['cpc'].max()
    plat_r['cpc_n'] = 1-(plat_r['cpc']-mn)/(mx-mn+1e-9)

    cats2 = ['ROAS','CTR','CPC Efficiency','Conversions','Revenue']
    fig_r2 = go.Figure()
    for _, row in plat_r.iterrows():
        v = [row['roas_n'],row['ctr_n'],row['cpc_n'],row['conversions_n'],row['revenue_n']]
        v_c = v+[v[0]]; c_c = cats2+[cats2[0]]
        color = PLAT_COLORS.get(row['platform'], ROSE)
        r,g,b = int(color[1:3],16),int(color[3:5],16),int(color[5:7],16)
        fig_r2.add_trace(go.Scatterpolar(
            r=v_c, theta=c_c, fill='toself',
            fillcolor=f'rgba({r},{g},{b},0.08)',
            line=dict(color=color, width=2),
            name=row['platform'],
        ))
    fig_r2.update_layout(
        polar=dict(
            bgcolor='rgba(13,16,18,0)',
            radialaxis=dict(visible=True,range=[0,1],gridcolor='rgba(212,139,139,0.08)',tickfont=dict(color='#4a4038',size=8,family='DM Mono'),linecolor='rgba(212,139,139,0.08)'),
            angularaxis=dict(gridcolor='rgba(212,139,139,0.08)',linecolor='rgba(212,139,139,0.08)',tickfont=dict(color='#8a8078',size=11,family='DM Mono'))
        ),
        paper_bgcolor='rgba(13,16,18,0)',
        font=dict(family='Inter',color='#8a8078'),
        title=dict(text='Platform Performance Radar',font=dict(size=16,color='#e8e0d6',family='Playfair Display')),
        legend=dict(bgcolor='rgba(13,16,18,0.6)',bordercolor='rgba(212,139,139,0.1)',borderwidth=1,font=dict(size=10,family='DM Mono')),
        margin=dict(l=40,r=40,t=60,b=40), height=450
    )
    st.plotly_chart(fig_r2, use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        section_title("Platform Summary")
        pd_df = dff.groupby('platform').agg(
            avg_roas=('ROAS','mean'), avg_ctr=('CTR','mean'),
            total_revenue=('revenue','sum'), campaigns=('platform','count')
        ).round(2).reset_index().sort_values('avg_roas',ascending=False)
        pd_df.columns=['Platform','Avg ROAS','Avg CTR','Revenue ($)','Campaigns']
        st.dataframe(pd_df, use_container_width=True, hide_index=True)

    with c2:
        section_title("Industry Summary")
        id_df = dff.groupby('industry').agg(
            avg_roas=('ROAS','mean'), avg_ctr=('CTR','mean'),
            total_revenue=('revenue','sum'), campaigns=('industry','count')
        ).round(2).reset_index().sort_values('avg_roas',ascending=False)
        id_df.columns=['Industry','Avg ROAS','Avg CTR','Revenue ($)','Campaigns']
        st.dataframe(id_df, use_container_width=True, hide_index=True)

    section_title("Industry Performance", "Average Return on Ad Spend — all verticals")
    ind_chart = dff.groupby('industry')['ROAS'].mean().reset_index().sort_values('ROAS')
    fig_ind = go.Figure()
    for i, row in ind_chart.iterrows():
        alpha = 0.35 + 0.65*(row['ROAS']-ind_chart['ROAS'].min())/(ind_chart['ROAS'].max()-ind_chart['ROAS'].min()+1e-9)
        fig_ind.add_trace(go.Bar(
            x=[row['ROAS']], y=[row['industry']], orientation='h', showlegend=False,
            marker_color=f'rgba(212,139,139,{alpha:.2f})',
            hovertemplate=f"<b>{row['industry']}</b><br>ROAS: {row['ROAS']:.2f}×<extra></extra>"
        ))
    fig_ind.update_layout(**{**LAY,
        'title':'ROAS by Industry Vertical',
        'yaxis':dict(tickfont=dict(size=11,family='DM Mono'),gridcolor='rgba(0,0,0,0)'),
        'xaxis':dict(title='Return on Ad Spend (×)',gridcolor='rgba(212,139,139,0.05)')
    })
    st.plotly_chart(fig_ind, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 4 — ANOMALY DETECTION
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div style='padding:8px 48px 32px'>", unsafe_allow_html=True)
    section_title("AI Anomaly Detection", "Isolation Forest · Unsupervised machine learning")

    st.markdown("""
    <div style='font-family:Inter,sans-serif;font-size:12px;color:#5a5046;font-weight:300;line-height:1.8;max-width:600px;margin-bottom:24px'>
        The model learns what "normal" looks like across all 14 campaign dimensions — 
        then flags anything that deviates significantly. Think of it as an AI auditor 
        reviewing every campaign before you waste budget.
    </div>
    """, unsafe_allow_html=True)

    col_sl, _ = st.columns([2,3])
    with col_sl:
        sensitivity = st.slider("Detection threshold — campaigns flagged (%)", 1, 15, 5)

    feats = ['CTR','CPC','CPA','ROAS','ad_spend','conversions']
    X = dff[feats].copy()
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    iso = IsolationForest(n_estimators=200, contamination=sensitivity/100, random_state=42)
    dff2 = dff.copy()
    dff2['_anom'] = iso.fit_predict(Xs)
    dff2['_score'] = iso.score_samples(Xs)
    dff2['Status'] = dff2['_anom'].map({1:'Normal', -1:'Flagged'})

    n_anom = (dff2['_anom']==-1).sum()
    n_norm = (dff2['_anom']==1).sum()
    wasted = dff2[dff2['_anom']==-1]['ad_spend'].sum()

    a1,a2,a3,a4 = st.columns(4)
    with a1: st.metric("Normal Campaigns", f"{n_norm:,}", help="Performing within expected range")
    with a2: st.metric("Flagged for Review", f"{n_anom:,}", help="AI detected unusual behavior")
    with a3: st.metric("Flag Rate", f"{n_anom/len(dff2)*100:.1f}%")
    with a4: st.metric("Budget at Risk", f"${wasted:,.0f}", help="Total spend in flagged campaigns")

    # Glassmorphism scatter
    fig_a = go.Figure()
    for status, color, size, opacity in [('Normal','#7a9a7a',5,0.4), ('Flagged','#D48B8B',8,0.8)]:
        sub = dff2[dff2['Status']==status]
        r,g,b = int(color[1:3],16),int(color[3:5],16),int(color[5:7],16)
        fig_a.add_trace(go.Scatter(
            x=sub['CPA'], y=sub['ROAS'], mode='markers', name=status,
            marker=dict(
                color=f'rgba({r},{g},{b},{opacity})',
                size=sub['ad_spend']/sub['ad_spend'].max()*size+3,
                line=dict(color=f'rgba({r},{g},{b},{min(opacity+0.2,1)})', width=0.8)
            ),
            hovertemplate=f'<b>{status}</b><br>Cost/Acquisition: $%{{x:.2f}}<br>Return: %{{y:.2f}}×<br><extra></extra>'
        ))

    # Highlight top 3 anomalies with labels
    top3 = dff2[dff2['_anom']==-1].nsmallest(3,'_score')
    for _, row in top3.iterrows():
        fig_a.add_annotation(
            x=row['CPA'], y=row['ROAS'],
            text=f"⚠ {row['platform'][:4]}·{row['country']}",
            font=dict(size=9, color='#C5A059', family='DM Mono'),
            bgcolor='rgba(13,16,18,0.8)',
            bordercolor='rgba(197,160,89,0.3)',
            borderwidth=1, borderpad=4,
            showarrow=True, arrowcolor='rgba(197,160,89,0.3)', arrowwidth=1,
            ax=30, ay=-30
        )
    fig_a.update_layout(**{**LAY,
        'title':'Campaign Health Map — Return vs Acquisition Cost',
        'xaxis':dict(title='Cost per Acquisition ($)',gridcolor='rgba(212,139,139,0.05)',tickfont=dict(size=10,family='DM Mono')),
        'yaxis':dict(title='Return on Ad Spend (×)',gridcolor='rgba(212,139,139,0.05)',tickfont=dict(size=10,family='DM Mono')),
        'height':500
    })
    st.plotly_chart(fig_a, use_container_width=True)
    insight_box("<strong style='color:#7a9a7a'>Green = healthy</strong> campaign — operating within normal parameters. <strong style='color:#D48B8B'>Pink = flagged</strong> — the AI detected unusual cost/return patterns. The <strong style='color:#C5A059'>gold labels</strong> mark the 3 most extreme anomalies — investigate these first.")

    section_title("Flagged Campaigns — Priority Review List")
    worst = dff2[dff2['_anom']==-1].nsmallest(10,'_score')[[
        'platform','industry','country','ROAS','CTR','CPA','ad_spend'
    ]].copy().round(3)
    worst.columns = ['Platform','Industry','Country','Return (ROAS ×)','Click Rate','Cost/Customer ($)','Budget Spent ($)']
    st.dataframe(worst, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 5 — ROAS PREDICTOR
# ════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div style='padding:8px 48px 32px'>", unsafe_allow_html=True)
    section_title("ROAS Predictor", "XGBoost model · Predict return before spending")

    st.markdown("""
    <div style='font-family:Inter,sans-serif;font-size:12px;color:#5a5046;font-weight:300;line-height:1.8;max-width:560px;margin-bottom:32px'>
        Enter your campaign parameters and the model — trained on real 2024 campaign data — 
        will estimate your expected Return on Ad Spend before you commit a single dollar.
    </div>
    """, unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div style="font-family:DM Mono,monospace;font-size:8px;color:#4a4038;letter-spacing:5px;text-transform:uppercase;margin-bottom:16px">Campaign Setup</div>', unsafe_allow_html=True)
        p_plat = st.selectbox("Where will you run the campaign?", ['TikTok Ads','Meta Ads','Google Ads'])
        p_camp = st.selectbox("Campaign format", ['Video','Search','Display','Shopping'])
        p_ind  = st.selectbox("Your industry", ['EdTech','E-commerce','Healthcare','Fintech','SaaS'])
        p_ctry = st.selectbox("Target country", ['UAE','UK','USA','Canada','Germany','India','Australia'])

    with c2:
        st.markdown('<div style="font-family:DM Mono,monospace;font-size:8px;color:#4a4038;letter-spacing:5px;text-transform:uppercase;margin-bottom:16px">Budget & Targets</div>', unsafe_allow_html=True)
        p_imp  = st.number_input("Expected impressions", 1000, 1000000, 50000, 5000)
        p_clk  = st.number_input("Expected clicks", 10, 50000, 2500, 100)
        p_spd  = st.number_input("Ad budget ($)", 100, 100000, 5000, 500)
        p_conv = st.number_input("Expected conversions", 1, 5000, 150, 10)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    if st.button("◈ &nbsp; Run Prediction"):
        if os.path.exists('models/roas_model.pkl'):
            with open('models/roas_model.pkl','rb') as f:
                mdl = pickle.load(f)

            plat_m = {'Google Ads':0,'Meta Ads':1,'TikTok Ads':2}
            camp_m = {'Display':0,'Search':1,'Shopping':2,'Video':3}
            ind_m  = {'E-commerce':0,'EdTech':1,'Fintech':2,'Healthcare':3,'SaaS':4}
            ctry_m = {'Australia':0,'Canada':1,'Germany':2,'India':3,'UAE':4,'UK':5,'USA':6}

            inp = [[p_imp,p_clk,p_clk/p_imp,p_spd/p_clk,p_spd,
                    p_conv,p_spd/p_conv,3,1,1,
                    plat_m.get(p_plat,0),camp_m.get(p_camp,1),
                    ind_m.get(p_ind,0),ctry_m.get(p_ctry,6)]]
            pred = mdl.predict(inp)[0]
            rev  = pred * p_spd
            profit = rev - p_spd
            roi = profit / p_spd * 100
            benchmark = df.groupby('platform')['ROAS'].mean().get(p_plat, df['ROAS'].mean())
            vs_bench = ((pred - benchmark) / benchmark * 100)

            _bc = '#7a9a7a' if vs_bench >= 0 else '#D48B8B'
            _ba = '&#8593; ' if vs_bench >= 0 else '&#8595; '
            _bw = 'above' if vs_bench >= 0 else 'below'
            _pred_html = f"""
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;1,400&family=Inter:wght@200;300;400&family=DM+Mono:wght@300;400&display=swap" rel="stylesheet">
<div style='background:#0d1012;border:1px solid rgba(212,139,139,0.12);padding:48px;margin-top:24px;position:relative;overflow:hidden'>
    <div style='position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(212,139,139,0.4),rgba(197,160,89,0.3),transparent)'></div>
    <div style='font-family:DM Mono,monospace;font-size:8px;color:#4a4038;letter-spacing:6px;text-transform:uppercase;margin-bottom:32px'>
        &#9672; &nbsp; Prediction Result &nbsp; &middot; &nbsp; XGBoost Model &nbsp; &middot; &nbsp; R&sup2; = 59.1%
    </div>
    <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:48px;text-align:left;margin-bottom:40px'>
        <div>
            <div style='font-family:Inter,sans-serif;font-size:64px;font-weight:200;color:#D48B8B;letter-spacing:-3px;line-height:1'>{pred:.2f}<span style='font-size:32px'>x</span></div>
            <div style='font-family:DM Mono,monospace;font-size:8px;color:#4a4038;letter-spacing:4px;text-transform:uppercase;margin-top:12px'>Return on Ad Spend</div>
            <div style='font-family:Inter,sans-serif;font-size:11px;color:#7a7068;margin-top:6px'>For every $1 &rarr; ${pred:.2f} returned</div>
        </div>
        <div>
            <div style='font-family:Inter,sans-serif;font-size:64px;font-weight:200;color:#C5A059;letter-spacing:-3px;line-height:1'>${rev:,.0f}</div>
            <div style='font-family:DM Mono,monospace;font-size:8px;color:#4a4038;letter-spacing:4px;text-transform:uppercase;margin-top:12px'>Expected Revenue</div>
            <div style='font-family:Inter,sans-serif;font-size:11px;color:#7a7068;margin-top:6px'>From ${p_spd:,} invested</div>
        </div>
        <div>
            <div style='font-family:Inter,sans-serif;font-size:64px;font-weight:200;color:#7a9a7a;letter-spacing:-3px;line-height:1'>{roi:.0f}<span style='font-size:32px'>%</span></div>
            <div style='font-family:DM Mono,monospace;font-size:8px;color:#4a4038;letter-spacing:4px;text-transform:uppercase;margin-top:12px'>Return on Investment</div>
            <div style='font-family:Inter,sans-serif;font-size:11px;color:#7a7068;margin-top:6px'>Net profit: ${profit:,.0f}</div>
        </div>
    </div>
    <div style='border-top:1px solid rgba(212,139,139,0.08);padding-top:24px;font-family:Inter,sans-serif;font-size:12px;color:#7a7068;font-weight:300;line-height:1.7'>
        <span style='color:#D48B8B;font-family:DM Mono,monospace;font-size:8px;letter-spacing:3px;text-transform:uppercase'>Benchmark &middot; </span>
        Your predicted ROAS of <strong style='color:#e8e0d6;font-weight:400'>{pred:.2f}x</strong> is
        <strong style='color:{_bc};font-weight:400'>{_ba}{abs(vs_bench):.1f}% {_bw}</strong>
        the {p_plat} average of {benchmark:.2f}x.
    </div>
</div>
            """
            components.html(_pred_html, height=420, scrolling=False)

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            if pred >= 10:
                st.success(f"Excellent efficiency. A {pred:.2f}× ROAS places this campaign in the top tier. Recommended to scale budget.")
            elif pred >= 5:
                st.warning(f"Good performance. A {pred:.2f}× ROAS is profitable. Monitor closely and optimize targeting.")
            else:
                st.error(f"Low efficiency. A {pred:.2f}× ROAS suggests underperformance. Review creative, targeting and budget before launching.")
        else:
            st.error("Model not found. Run `python model.py` first to train the prediction model.")
    st.markdown("</div>", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:48px;border-top:1px solid rgba(212,139,139,0.06);margin-top:24px'>
    <div style='font-family:Playfair Display,serif;font-style:italic;font-size:14px;color:rgba(232,224,214,0.2);margin-bottom:10px;letter-spacing:0.5px'>
        "I don't just analyze data — I engineer intelligence that drives revenue."
    </div>
    <div style='font-family:DM Mono,monospace;font-size:8px;color:rgba(232,224,214,0.12);letter-spacing:3px;text-transform:uppercase'>
        Sarah Silva &nbsp;·&nbsp; Senior Data & AI Engineer &nbsp;·&nbsp; Harvard · TikTok · Globo · WPP
    </div>
</div>
""", unsafe_allow_html=True)
