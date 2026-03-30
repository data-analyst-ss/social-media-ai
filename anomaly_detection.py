import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("SOCIAL MEDIA AI — Anomaly Detection")
print("Isolation Forest — Auto-flag Underperforming Campaigns")
print("=" * 60)

# LOAD DATA
df = pd.read_csv('data/global_ads_performance_dataset.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.dropna()

# FEATURES FOR ANOMALY DETECTION
features = ['CTR', 'CPC', 'CPA', 'ROAS', 'ad_spend', 'conversions']
X = df[features].copy()

# SCALE
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ISOLATION FOREST
print("\n🤖 Training Isolation Forest...")
iso = IsolationForest(
    n_estimators=200,
    contamination=0.05,  # 5% anomalies expected
    random_state=42
)
df['anomaly'] = iso.fit_predict(X_scaled)
df['anomaly_score'] = iso.score_samples(X_scaled)
df['is_anomaly'] = df['anomaly'] == -1
df['status'] = df['is_anomaly'].map({True: '⚠️ Anomaly', False: '✅ Normal'})

# RESULTS
n_anomalies = df['is_anomaly'].sum()
n_normal = (~df['is_anomaly']).sum()
pct = n_anomalies / len(df) * 100

print(f"\n📊 Total campaigns: {len(df)}")
print(f"✅ Normal:    {n_normal} ({100-pct:.1f}%)")
print(f"⚠️  Anomalies: {n_anomalies} ({pct:.1f}%)")

# ANOMALY PROFILE
print("\n--- ANOMALY PROFILE ---")
print(df.groupby('is_anomaly')[features].mean().round(2).to_string())

print("\n--- ANOMALIES BY PLATFORM ---")
print(df[df['is_anomaly']].groupby('platform').size().sort_values(ascending=False).to_string())

print("\n--- ANOMALIES BY INDUSTRY ---")
print(df[df['is_anomaly']].groupby('industry').size().sort_values(ascending=False).to_string())

print("\n--- TOP 10 WORST ANOMALIES (lowest score) ---")
worst = df[df['is_anomaly']].nsmallest(10, 'anomaly_score')[
    ['date', 'platform', 'industry', 'country', 'ROAS', 'CTR', 'CPA', 'ad_spend', 'anomaly_score']
].round(3)
print(worst.to_string(index=False))

# VISUALIZATIONS
lay = dict(
    template='plotly_dark',
    paper_bgcolor='#04070a',
    plot_bgcolor='#04070a',
    font=dict(family='DM Sans, sans-serif', color='#c4b49a'),
    title_font=dict(size=17, color='#f0e6d3'),
)

# CHART 1: ROAS vs CPA — Anomalies highlighted
fig1 = px.scatter(
    df, x='CPA', y='ROAS',
    color='status',
    size='ad_spend',
    hover_data=['platform', 'industry', 'country', 'CTR'],
    title='ROAS vs CPA — Anomaly Detection (Isolation Forest)',
    color_discrete_map={
        '✅ Normal': '#4a7550',
        '⚠️ Anomaly': '#c45a67'
    },
    opacity=0.7
)
fig1.update_layout(**lay)
fig1.show()

# CHART 2: Anomaly score distribution
fig2 = go.Figure()
fig2.add_trace(go.Histogram(
    x=df[~df['is_anomaly']]['anomaly_score'],
    name='Normal', nbinsx=40,
    marker_color='rgba(74,117,80,0.7)',
    hovertemplate='Score: %{x:.3f}<br>Count: %{y}<extra>Normal</extra>'
))
fig2.add_trace(go.Histogram(
    x=df[df['is_anomaly']]['anomaly_score'],
    name='Anomaly', nbinsx=20,
    marker_color='rgba(196,90,103,0.8)',
    hovertemplate='Score: %{x:.3f}<br>Count: %{y}<extra>Anomaly</extra>'
))
fig2.update_layout(
    title='Anomaly Score Distribution',
    xaxis_title='Anomaly Score (lower = more anomalous)',
    yaxis_title='Number of Campaigns',
    barmode='overlay',
    legend=dict(bgcolor='rgba(0,0,0,0.4)'),
    **lay
)
fig2.show()

# CHART 3: Anomalies by platform
anom_plat = df.groupby(['platform', 'is_anomaly']).size().reset_index(name='count')
anom_plat['type'] = anom_plat['is_anomaly'].map({True: '⚠️ Anomaly', False: '✅ Normal'})
fig3 = px.bar(
    anom_plat, x='platform', y='count', color='type',
    title='Normal vs Anomaly Campaigns by Platform',
    barmode='group',
    color_discrete_map={'✅ Normal': '#4a7550', '⚠️ Anomaly': '#c45a67'}
)
fig3.update_layout(**lay)
fig3.show()

# CHART 4: Anomaly rate over time
df['month'] = df['date'].dt.to_period('M').astype(str)
monthly_anom = df.groupby('month').apply(
    lambda x: (x['is_anomaly'].sum() / len(x) * 100)
).reset_index(name='anomaly_rate')
fig4 = go.Figure(go.Scatter(
    x=monthly_anom['month'],
    y=monthly_anom['anomaly_rate'],
    mode='lines+markers',
    line=dict(color='#c45a67', width=2),
    marker=dict(color='#9A3F4A', size=8),
    fill='tozeroy',
    fillcolor='rgba(154,63,74,0.1)',
    hovertemplate='%{x}<br>Anomaly Rate: %{y:.1f}%<extra></extra>'
))
fig4.update_layout(
    title='Monthly Anomaly Rate Over Time',
    xaxis_title='Month',
    yaxis_title='Anomaly Rate (%)',
    **lay
)
fig4.show()

# SAVE FLAGGED CAMPAIGNS
df[df['is_anomaly']].to_csv('outputs/anomalies_detected.csv', index=False)
print(f"\n💾 Anomalies saved → outputs/anomalies_detected.csv")
print(f"\n✅ Anomaly Detection complete!")
print(f"   {n_anomalies} campaigns flagged for review")
print(f"\n🚀 Next: python insights.py — OpenAI AI Insights")
