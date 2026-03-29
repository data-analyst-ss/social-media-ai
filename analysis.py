import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
n = 500

platforms = np.random.choice(['TikTok', 'Meta', 'Google Ads'], n, p=[0.35, 0.35, 0.30])
industries = np.random.choice(['E-commerce', 'Finance', 'Healthcare', 'Entertainment', 'Travel'], n)
months = pd.date_range('2023-01-01', periods=n, freq='D')

df = pd.DataFrame({
    'date': months,
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

print("Dataset created!")
print(df.head())
print(df[['CTR','CPC','ROAS','CVR','CPA']].describe().round(2))
