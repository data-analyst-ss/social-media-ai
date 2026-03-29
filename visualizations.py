import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Generate data
np.random.seed(42)
n = 500
platforms = np.random.choice(['TikTok', 'Meta', 'Google Ads'], n, p=[0.35, 0.35, 0.30])
industries = np.random.choice(['E-commerce', 'Finance', 'Healthcare', 'Entertainment', 'Travel'], n)
dates = pd.date_range('2023-01-01', periods=n, freq='D')

df = pd.DataFrame({
    'date': dates,
    'platform': platforms,
    'industry': industries,
    'impressions': np.random.randint(10000, 500000, n),
    'clicks': np.random.randint(100, 5000, n),
    'spend': np.round(np.random.uniform(100, 10000, n), 2),
    'conversions': np.random.randint(5, 500, n),
    'revenue': np.round(np.random.uniform(500, 50000, n), 2),
})

df['CTR'] = (df['clicks'] / df['impressions'] * 100).round(2)
df['CPC'] = (df['spend'] / df['clicks']).round(2)
df['ROAS'] = (df['revenue'] / df['spend']).round(2)
df['CVR'] = (df['conversions'] / df['clicks'] * 100).round(2)
df['CPA'] = (df['spend'] / df['conversions']).round(2)

# CHART 1 - ROAS by Platform
fig1 = px.box(df, x='platform', y='ROAS', color='platform',
              title='ROAS Distribution by Platform',
              color_discrete_map={'TikTok': '#BD6809', 'Meta': '#9A3F4A', 'Google Ads': '#2F4731'})
fig1.show()

# CHART 2 - Spend vs Revenue
fig2 = px.scatter(df, x='spend', y='revenue', color='platform',
                  size='conversions', hover_data=['CTR', 'ROAS'],
                  title='Spend vs Revenue by Platform',
                  color_discrete_map={'TikTok': '#BD6809', 'Meta': '#9A3F4A', 'Google Ads': '#2F4731'})
fig2.show()

# CHART 3 - Monthly Performance
df['month'] = df['date'].dt.to_period('M').astype(str)
monthly = df.groupby(['month', 'platform']).agg(
    total_spend=('spend', 'sum'),
    total_revenue=('revenue', 'sum'),
    avg_roas=('ROAS', 'mean')
).reset_index()

fig3 = px.line(monthly, x='month', y='avg_roas', color='platform',
               title='Average ROAS Over Time by Platform',
               color_discrete_map={'TikTok': '#BD6809', 'Meta': '#9A3F4A', 'Google Ads': '#2F4731'})
fig3.show()

# CHART 4 - Industry Performance
industry_perf = df.groupby('industry').agg(
    avg_roas=('ROAS', 'mean'),
    total_revenue=('revenue', 'sum'),
    avg_ctr=('CTR', 'mean')
).reset_index().sort_values('avg_roas', ascending=True)

fig4 = px.bar(industry_perf, x='avg_roas', y='industry', orientation='h',
              title='Average ROAS by Industry',
              color='avg_roas', color_continuous_scale=['#2F4731', '#BD6809'])
fig4.show()

print("All charts generated successfully!")
print("\nTop 5 campaigns by ROAS:")
print(df.nlargest(5, 'ROAS')[['date', 'platform', 'industry', 'spend', 'revenue', 'ROAS']])
