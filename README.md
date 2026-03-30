# 🤖 Social Media AI — Intelligence Dashboard
### ML-Powered Campaign Analytics · TikTok · Meta · Google Ads

<div align="center">

![Python](https://img.shields.io/badge/Python-3.14-9A3F4A?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ROAS_Prediction-BD6809?style=for-the-badge&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-9A3F4A?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-BD6809?style=for-the-badge&logo=openai&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-9A3F4A?style=for-the-badge&logo=plotly&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML_Pipeline-BD6809?style=for-the-badge&logo=scikit-learn&logoColor=white)

[![Live App](https://img.shields.io/badge/LIVE_APP-9A3F4A?style=for-the-badge&logo=streamlit)](https://social-media-ai-ossaocyh4myx8srgsri6up.streamlit.app)

</div>

---

## Overview

A production-ready **AI-powered analytics dashboard** for social media campaign performance across TikTok Ads, Meta Ads, and Google Ads. Built on real Kaggle data with 1,800 campaigns across 10 platforms, 9 industries, and 15 countries, this project combines machine learning, anomaly detection, and generative AI to turn raw ad performance data into actionable business intelligence.

The dashboard runs live 24/7 on Streamlit Cloud — no local setup required.

---

## What the Dashboard Does

| Module | Description |
|--------|-------------|
| **Performance Overview** | KPIs: ROAS, CTR, CPC, CPA, Revenue — filterable by platform, industry, country and campaign type |
| **World Map** | Geographic ROAS distribution across 15 countries |
| **Platform & Industry Analysis** | Side-by-side benchmark comparison — which platform wins by sector |
| **ROAS Predictor** | XGBoost model — predict expected ROAS before launching a campaign |
| **Anomaly Detection** | Isolation Forest — flags campaigns with unusual performance patterns |
| **AI Insights** | GPT-4o-mini generates plain-English strategic insights from filtered data |

---

## Key Results

```
📊 Best ROAS platform:   TikTok Ads (outperforms Meta and Google in this dataset)
📊 Best ROAS industry:   EdTech
📊 Best ROAS country:    UAE
📊 Best campaign type:   Search
📊 Avg ROAS (all):       6.45×  (every $1 spent returns $6.45)
📊 Avg CTR:              3.84%
📊 Avg CPC:              $1.57
📊 XGBoost R²:           59.1%  (ROAS prediction accuracy)
📊 Total campaigns:      1,800 across 10 platforms
```

---

## Tech Stack & Architecture

```
Data Layer
└── Kaggle dataset (global_ads_performance_dataset.csv)
    └── 1,800 campaigns · 10 platforms · 9 industries · 15 countries

Analysis Layer
├── analysis_real.py       — EDA and data profiling
├── visualizations.py      — Plotly interactive charts
├── model.py               — XGBoost ROAS prediction pipeline
├── anomaly_detection.py   — Isolation Forest anomaly flagging
└── insights.py            — OpenAI API integration (GPT-4o-mini)

Presentation Layer
└── streamlit_app.py       — Multi-tab dashboard with filters
    └── Deployed on Streamlit Cloud (live 24/7)
```

### Why each tool was chosen

| Tool | Role | Why |
|------|------|-----|
| **XGBoost** | ROAS prediction | Handles non-linear relationships between campaign features better than linear models · robust to outliers |
| **Isolation Forest** | Anomaly detection | Unsupervised · no labelled anomalies required · fast on tabular data |
| **Plotly** | Visualizations | Interactive charts with hover, filter and zoom · standalone HTML exports |
| **Streamlit** | Dashboard framework | Pure Python · no front-end code · instant deploy on Streamlit Cloud |
| **OpenAI GPT-4o-mini** | AI insights | Converts filtered metrics into plain-English recommendations · low latency · cost-efficient |
| **Pandas + NumPy** | Data pipeline | Industry-standard ETL for tabular data |

---

## ML Models

### ROAS Predictor — XGBoost Regressor

Predicts expected Return on Ad Spend based on campaign configuration before launch.

```
Features used:
  - Platform (TikTok, Meta, Google, LinkedIn, etc.)
  - Industry (EdTech, eCommerce, Healthcare, etc.)
  - Country
  - Campaign Type (Search, Display, Video, Social)
  - Ad Spend
  - Click-Through Rate (CTR)
  - Cost per Click (CPC)

Target variable: ROAS (Revenue / Ad Spend)

Performance:
  R² = 0.591 — explains 59.1% of ROAS variance
  Best for: pre-launch budget allocation decisions
```

### Anomaly Detection — Isolation Forest

Flags campaigns with statistically unusual performance — either underperforming (potential budget waste) or overperforming (patterns worth replicating).

```
Algorithm: Isolation Forest (sklearn)
Contamination: 5% (flags top/bottom 5% of campaigns)
Output: binary flag per campaign (normal / anomaly)
Use case: weekly campaign audit · budget reallocation alerts
```

---

## AI Insights — How It Works

When the user applies filters (e.g., "TikTok Ads · EdTech · UAE"), the dashboard sends the filtered KPIs to GPT-4o-mini with a structured prompt:

```python
prompt = f"""
You are a senior media analyst. Given these campaign metrics:
- Platform: {platform}
- Industry: {industry}
- Avg ROAS: {roas:.2f}x
- Avg CTR: {ctr:.2%}
- Avg CPC: ${cpc:.2f}

Provide 3 concise strategic insights and 2 actionable recommendations.
Focus on what the data implies for budget allocation decisions.
"""
```

The model returns plain-English insights that update dynamically as filters change — turning any data slice into an instant strategic briefing.

---

## Project Structure

```
social-media-ai/
│
├── streamlit_app.py           # Main dashboard (multi-tab Streamlit app)
├── analysis_real.py           # EDA on real Kaggle data
├── visualizations.py          # Plotly chart library
├── model.py                   # XGBoost training and prediction pipeline
├── anomaly_detection.py       # Isolation Forest implementation
├── insights.py                # OpenAI API integration
│
├── data/
│   └── global_ads_performance_dataset.csv   # Source: Kaggle
│
├── models/
│   └── roas_model.pkl         # Trained XGBoost model (serialized)
│
├── outputs/
│   ├── ai_insights.txt        # Sample generated insights
│   └── anomalies_detected.csv # Flagged campaigns
│
└── requirements.txt
```

---

## How to Run Locally

**Requirements:** Python 3.8+ · OpenAI API key

```bash
# 1. Clone the repository
git clone https://github.com/data-analyst-ss/social-media-ai.git
cd social-media-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"
# On Windows: set OPENAI_API_KEY=your-key-here

# 4. Run the dashboard
streamlit run streamlit_app.py

# Dashboard opens at http://localhost:8501
```

**Or just use the live app — no setup required:**
👉 [social-media-ai-ossaocyh4myx8srgsri6up.streamlit.app](https://social-media-ai-ossaocyh4myx8srgsri6up.streamlit.app)

---

## Dataset

**Source:** [Global Ads Performance Dataset — Kaggle](https://www.kaggle.com/)

```
Rows:      1,800 campaigns
Platforms: TikTok Ads · Meta Ads · Google Ads · LinkedIn · Twitter
           Pinterest · Snapchat · YouTube · Reddit · Microsoft Ads
Industries: eCommerce · EdTech · Healthcare · Finance · Travel
            Fashion · Food & Beverage · Gaming · Real Estate · B2B SaaS
Countries:  USA · UK · Germany · France · Brazil · India · UAE
            Australia · Canada · Japan · Mexico · Singapore · + more
Metrics:    Ad Spend · Impressions · Clicks · Conversions · Revenue
            CTR · CPC · CPA · ROAS · Engagement Rate
```

---

## What This Project Demonstrates

**Engineering:**
- End-to-end ML pipeline from raw data to deployed model
- Supervised regression (XGBoost) + unsupervised anomaly detection (Isolation Forest)
- LLM integration via API with structured prompting
- Production deployment on Streamlit Cloud

**Analytics:**
- Multi-dimensional campaign performance benchmarking
- ROAS variance decomposition by platform, industry, country and campaign type
- Anomaly detection for budget protection and opportunity discovery

**Business:**
- Translating model outputs into executive-ready insights
- Building self-service analytics tools that reduce analyst dependency
- Designing for the media buyer's decision workflow, not just data exploration

---

<div align="center">

**Sarah Silva** · Senior Data & AI Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-sarahgleicesilva-9A3F4A?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/sarahgleicesilva)
[![Harvard ML](https://img.shields.io/badge/Harvard_ML_Portfolio-BD6809?style=flat-square&logo=github&logoColor=white)](https://data-analyst-ss.github.io/ml-portfolio-R)
[![Live App](https://img.shields.io/badge/Live_App-9A3F4A?style=flat-square&logo=streamlit&logoColor=white)](https://social-media-ai-ossaocyh4myx8srgsri6up.streamlit.app)

</div>
