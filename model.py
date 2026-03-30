import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
import pickle, os, warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("SOCIAL MEDIA AI — XGBoost ROAS Prediction Model")
print("=" * 60)

df = pd.read_csv('data/global_ads_performance_dataset.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.dropna()

df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
df['dayofweek'] = df['date'].dt.dayofweek

le = LabelEncoder()
cat_cols = ['platform', 'campaign_type', 'industry', 'country']
for col in cat_cols:
    df[col + '_enc'] = le.fit_transform(df[col])

features = [
    'impressions', 'clicks', 'CTR', 'CPC', 'ad_spend',
    'conversions', 'CPA', 'month', 'quarter', 'dayofweek',
    'platform_enc', 'campaign_type_enc', 'industry_enc', 'country_enc'
]

feature_labels = {
    'impressions': 'Impressions',
    'clicks': 'Clicks',
    'CTR': 'CTR',
    'CPC': 'CPC',
    'ad_spend': 'Ad Spend',
    'conversions': 'Conversions',
    'CPA': 'CPA',
    'month': 'Month',
    'quarter': 'Quarter',
    'dayofweek': 'Day of Week',
    'platform_enc': 'Platform',
    'campaign_type_enc': 'Campaign Type',
    'industry_enc': 'Industry',
    'country_enc': 'Country'
}

X = df[features]
y = df['ROAS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n📊 Training: {X_train.shape[0]} | Test: {X_test.shape[0]} | Features: {len(features)}")
print("\n🤖 Training XGBoost...")

model = xgb.XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"\n{'=' * 60}")
print("MODEL PERFORMANCE")
print(f"{'=' * 60}")
print(f"✅ R² Score:  {r2:.4f}  ({r2*100:.1f}% variance explained)")
print(f"✅ MAE:       {mae:.4f}")
print(f"✅ MAPE:      {mape:.2f}%")

importance = pd.DataFrame({
    'feature': [feature_labels[f] for f in features],
    'importance': model.feature_importances_
}).sort_values('importance', ascending=True)

print(f"\n📊 Top Features:")
print(importance.sort_values('importance', ascending=False).head(5).to_string(index=False))

os.makedirs('models', exist_ok=True)
with open('models/roas_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\n💾 Model saved → models/roas_model.pkl")

lay = dict(
    template='plotly_dark',
    paper_bgcolor='#04070a',
    plot_bgcolor='rgba(10,16,20,0.8)',
    font=dict(family='DM Sans, sans-serif', color='#c4b49a'),
    title_font=dict(size=18, color='#f0e6d3'),
)

# CHART 1: Actual vs Predicted
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=y_test.values[:150], y=y_pred[:150],
    mode='markers',
    marker=dict(color='#c45a67', size=7, opacity=0.75,
                line=dict(color='#9A3F4A', width=0.5)),
    name='Campaigns',
    hovertemplate='Actual: %{x:.2f}x<br>Predicted: %{y:.2f}x<extra></extra>'
))
fig1.add_trace(go.Scatter(
    x=[y_test.min(), y_test.max()],
    y=[y_test.min(), y_test.max()],
    mode='lines',
    line=dict(color='#BD6809', dash='dash', width=2),
    name='Perfect Prediction'
))
fig1.update_layout(
    title=f'Actual vs Predicted ROAS — XGBoost  |  R² = {r2:.3f}  |  MAE = {mae:.3f}',
    xaxis_title='Actual ROAS',
    yaxis_title='Predicted ROAS',
    legend=dict(bgcolor='rgba(0,0,0,0.4)', bordercolor='rgba(154,63,74,0.2)', borderwidth=1),
    **lay
)
fig1.show()

# CHART 2: Feature Importance
fig2 = go.Figure(go.Bar(
    x=importance['importance'],
    y=importance['feature'],
    orientation='h',
    marker=dict(
        color=importance['importance'],
        colorscale=[[0, '#2F4731'], [0.5, '#9A3F4A'], [1, '#c45a67']],
        showscale=False,
        line=dict(color='rgba(154,63,74,0.3)', width=0.5)
    ),
    hovertemplate='%{y}: %{x:.4f}<extra></extra>'
))
fig2.update_layout(
    title='Feature Importance — What drives ROAS most?',
    xaxis_title='Importance Score',
    yaxis_title='',
    yaxis=dict(tickfont=dict(size=13)),
    margin=dict(l=140, r=40, t=60, b=50),
    **lay
)
fig2.show()

# DEMO PREDICTIONS
print(f"\n{'=' * 60}")
print("ROAS PREDICTIONS — DEMO CAMPAIGNS")
print(f"{'=' * 60}")

def predict_roas(platform, campaign_type, industry, country,
                 impressions, clicks, ad_spend, conversions):
    plat_map = {'Google Ads':0,'Meta Ads':1,'TikTok Ads':2}
    camp_map = {'Display':0,'Search':1,'Shopping':2,'Video':3}
    ind_map  = {'E-commerce':0,'EdTech':1,'Fintech':2,'Healthcare':3,'SaaS':4}
    ctry_map = {'Australia':0,'Canada':1,'Germany':2,'India':3,'UAE':4,'UK':5,'USA':6}
    ctr = clicks/impressions
    cpc = ad_spend/clicks if clicks>0 else 0
    cpa = ad_spend/conversions if conversions>0 else 0
    inp = [[impressions,clicks,ctr,cpc,ad_spend,conversions,cpa,3,1,1,
            plat_map.get(platform,0),camp_map.get(campaign_type,1),
            ind_map.get(industry,0),ctry_map.get(country,6)]]
    return model.predict(inp)[0]

demos = [
    ('TikTok Ads','Video','EdTech','UAE',50000,2500,5000,150),
    ('Google Ads','Search','E-commerce','USA',100000,4000,8000,200),
    ('Meta Ads','Display','Healthcare','UK',80000,3200,6400,180),
    ('TikTok Ads','Video','SaaS','Canada',60000,3000,6000,160),
]
for plat,camp,ind,ctry,imp,clk,spd,conv in demos:
    pred = predict_roas(plat,camp,ind,ctry,imp,clk,spd,conv)
    print(f"  {plat:12} | {camp:8} | {ind:12} | {ctry:10} → ROAS: {pred:.2f}x")

print(f"\n✅ Done! R²={r2*100:.1f}% | MAE={mae:.3f}")
print("🚀 Next: python anomaly_detection.py")
