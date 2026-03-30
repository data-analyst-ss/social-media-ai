import pandas as pd
import numpy as np
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("SOCIAL MEDIA AI — Real Data Analysis")
print("Global Ads Performance: Google, Meta, TikTok")
print("=" * 60)

df = pd.read_csv('data/global_ads_performance_dataset.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.dropna()

print(f"\n✅ Dataset: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"📅 Period: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"📱 Platforms: {list(df['platform'].unique())}")
print(f"🏭 Industries: {list(df['industry'].unique())}")
print(f"🌍 Countries: {list(df['country'].unique())}")

print("\n--- PLATFORM PERFORMANCE ---")
plat = df.groupby('platform').agg(
    campaigns=('platform','count'),
    avg_roas=('ROAS','mean'),
    avg_ctr=('CTR','mean'),
    avg_cpc=('CPC','mean'),
    total_revenue=('revenue','sum'),
    total_spend=('ad_spend','sum')
).round(2)
print(plat.to_string())

print("\n--- INDUSTRY PERFORMANCE ---")
ind = df.groupby('industry').agg(
    avg_roas=('ROAS','mean'),
    avg_ctr=('CTR','mean'),
    campaigns=('industry','count')
).round(2).sort_values('avg_roas', ascending=False)
print(ind.to_string())

print("\n--- COUNTRY PERFORMANCE ---")
ctry = df.groupby('country').agg(
    avg_roas=('ROAS','mean'),
    total_revenue=('revenue','sum')
).round(2).sort_values('avg_roas', ascending=False)
print(ctry.to_string())

# CHARTS
colors = {'Google Ads':'#2F4731','TikTok Ads':'#9A3F4A','Meta Ads':'#BD6809'}
lay = dict(template='plotly_dark', paper_bgcolor='#04070a', plot_bgcolor='#04070a')

fig1 = px.box(df, x='platform', y='ROAS', color='platform',
              title='ROAS Distribution by Platform — Real Data',
              color_discrete_map=colors)
fig1.update_layout(**lay)
fig1.show()

rev_ind = df.groupby(['industry','platform'])['revenue'].sum().reset_index()
fig2 = px.bar(rev_ind, x='industry', y='revenue', color='platform',
              title='Total Revenue by Industry & Platform — Real Data',
              barmode='group', color_discrete_map=colors)
fig2.update_layout(**lay)
fig2.show()

df['month'] = df['date'].dt.to_period('M').astype(str)
monthly = df.groupby(['month','platform'])['ROAS'].mean().reset_index()
fig3 = px.line(monthly, x='month', y='ROAS', color='platform',
               title='ROAS Trend Over Time — Real Data',
               color_discrete_map={'Google Ads':'#4a7550','TikTok Ads':'#c45a67','Meta Ads':'#e8852a'})
fig3.update_layout(**lay)
fig3.show()

ctry_c = df.groupby('country')['ROAS'].mean().reset_index().sort_values('ROAS')
fig4 = px.bar(ctry_c, x='ROAS', y='country', orientation='h',
              title='Average ROAS by Country — Real Data',
              color='ROAS', color_continuous_scale=['#2F4731','#9A3F4A','#c45a67'])
fig4.update_layout(**lay)
fig4.show()

fig5 = px.scatter(df.sample(min(500,len(df))), x='CTR', y='CPA',
                  color='platform', size='revenue',
                  hover_data=['industry','country','ROAS'],
                  title='CTR vs CPA by Platform — Real Data',
                  color_discrete_map=colors)
fig5.update_layout(**lay)
fig5.show()

print("\n✅ 5 charts generated!")
print(f"\n🏆 Best ROAS platform: {df.groupby('platform')['ROAS'].mean().idxmax()}")
print(f"🏆 Best ROAS industry: {df.groupby('industry')['ROAS'].mean().idxmax()}")
print(f"🏆 Best ROAS country:  {df.groupby('country')['ROAS'].mean().idxmax()}")
print(f"🏆 Best campaign type: {df.groupby('campaign_type')['ROAS'].mean().idxmax()}")
print("\n🚀 Next: python model.py")
