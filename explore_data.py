import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("SOCIAL MEDIA AI INTELLIGENCE DASHBOARD")
print("Real Data Analysis — Global Ads Performance")
print("=" * 60)

# ============================================================
# LOAD REAL DATA
# ============================================================
df = pd.read_csv('data/global_ads_performance_dataset.csv')

print(f"\n✅ Dataset loaded!")
print(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n📋 Columns:")
for col in df.columns:
    print(f"   - {col} ({df[col].dtype})")

print(f"\n🔍 First 3 rows:")
print(df.head(3).to_string())

print(f"\n📈 Basic stats:")
print(df.describe().round(2).to_string())

print(f"\n❓ Missing values:")
print(df.isnull().sum())

print(f"\n🏷️ Unique values per column:")
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"   {col}: {df[col].unique()[:10]}")
