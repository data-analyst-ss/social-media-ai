<div align="center">

<!-- TITLE -->
# 🤖 Social Media AI — Intelligence Dashboard
### ML-Powered Campaign Analytics · TikTok · Meta · Google Ads

<br>

<!-- BADGES -->
[![Live App](https://img.shields.io/badge/🚀_LIVE_APP-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://social-media-ai-ossaocyh4myx8srgsri6up.streamlit.app)
[![Made with LangGraph](https://img.shields.io/badge/Made_with-LangGraph-9A3F4A?style=for-the-badge&logo=python&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br>

> **I built an AI that reads 1,800 real ad campaigns and tells you exactly what to do next.**
> XGBoost · Isolation Forest · LangGraph · GPT-4o-mini · Streamlit

</div>

---

## 🎬 See it in action

<!-- Replace with your screenshot or GIF -->
<!-- Tip: record with Loom or ScreenToGif, then drag the file into the repo -->

![Dashboard Demo](assets/demo.gif)

> 📸 To add your own: record the dashboard with [ScreenToGif](https://www.screentogif.com/) (free), save as `demo.gif`, create an `assets/` folder in the repo and upload it there.

---

## ⚡ What this project does

Media buyers drown in data — thousands of campaigns, dozens of metrics, no time to analyze everything manually.

So I built an AI that does it for them.

```
User selects filters (Platform · Industry · Country · Campaign Type)
        ↓
XGBoost predicts ROAS before launch          → R² = 59.1%
Isolation Forest flags anomaly campaigns     → automatic detection
GPT-4o-mini reads the data                  → generates strategic insight
        ↓
Dashboard renders charts + insights + recommendations
```

---

## 📊 Key findings from 1,800 real campaigns

| Metric | Result |
|--------|--------|
| 🏆 Best ROAS platform | **TikTok Ads** |
| 🏆 Best ROAS industry | **EdTech** |
| 🏆 Best ROAS country | **UAE** |
| 📈 Average ROAS (all campaigns) | **6.45×** |
| 📈 Average CTR | **3.84%** |
| 📈 Average CPC | **$1.57** |
| 🤖 XGBoost R² | **59.1%** |
| 📋 Total campaigns analyzed | **1,800** |

> Every $1 spent across these 1,800 campaigns returned **$6.45** on average.

---

## 🧠 Dashboard modules

| Module | What it does |
|--------|-------------|
| **Performance Overview** | KPIs: ROAS · CTR · CPC · CPA · Revenue — filterable by platform, industry, country, campaign type |
| **World Map** | Geographic ROAS distribution across 15 countries |
| **Platform & Industry** | Side-by-side benchmark — which platform wins by sector |
| **ROAS Predictor** | XGBoost model — predict ROAS before launching a campaign |
| **Anomaly Detection** | Isolation Forest — flags campaigns with unusual performance |
| **AI Insights** | GPT-4o-mini generates plain-English strategic insights from filtered data |

---

## 🔧 Tech stack & architecture

```
Data Layer
└── Kaggle — global_ads_performance_dataset.csv
    └── 1,800 campaigns · 10 platforms · 9 industries · 15 countries

AI Pipeline (LangGraph)
├── XGBoost          → ROAS prediction (R²=59.1%)
├── Isolation Forest → Anomaly detection (unsupervised)
├── OpenAI API       → GPT-4o-mini insight generation
└── LangGraph        → Agent orchestration with persistent state

Presentation Layer
└── Streamlit        → Multi-tab dashboard · deployed on Streamlit Cloud
```

### Why each tool was chosen

| Tool | Role | Why |
|------|------|-----|
| **XGBoost** | ROAS prediction | Handles non-linear relationships · robust to outliers |
| **Isolation Forest** | Anomaly detection | Unsupervised · no labelled anomalies required |
| **LangGraph** | Agent orchestration | Stateful · cyclical flows · tool routing |
| **GPT-4o-mini** | Insight generation | Low latency · cost-efficient · strong reasoning |
| **Streamlit** | Dashboard | Pure Python · instant deploy · no front-end code |
| **Plotly** | Visualizations | Interactive charts · hover · zoom · filter |

---

## 🚀 Run locally

**Requirements:** Python 3.8+ · OpenAI API key

```bash
# 1. Clone the repo
git clone https://github.com/data-analyst-ss/social-media-ai.git
cd social-media-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"
# Windows: set OPENAI_API_KEY=your-key-here

# 4. Run the dashboard
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

**Or just use the live app — no setup needed:**

👉 [social-media-ai-ossaocyh4myx8srgsri6up.streamlit.app](https://social-media-ai-ossaocyh4myx8srgsri6up.streamlit.app)

---

## 📁 Project structure

```
social-media-ai/
│
├── streamlit_app.py           # Main dashboard (multi-tab Streamlit app)
├── analysis_real.py           # EDA on real Kaggle data
├── visualizations.py          # Plotly chart library
├── model.py                   # XGBoost training and prediction pipeline
├── anomaly_detection.py       # Isolation Forest implementation
├── insights.py                # OpenAI API + LangGraph integration
│
├── data/
│   └── global_ads_performance_dataset.csv   # Source: Kaggle
│
├── models/
│   └── roas_model.pkl         # Trained XGBoost model (serialized)
│
├── assets/
│   └── demo.gif               # Dashboard demo
│
└── requirements.txt
```

---

## 💡 AI Insights — how it works

When the user applies filters, the dashboard sends the filtered KPIs to GPT-4o-mini via LangGraph:

```python
# Simplified agent flow
prompt = f"""
You are a senior media analyst. Given these campaign metrics:
- Platform: {platform} | Industry: {industry}
- Avg ROAS: {roas:.2f}x | Avg CTR: {ctr:.2%} | Avg CPC: ${cpc:.2f}

Provide 3 concise strategic insights and 2 actionable recommendations.
Focus on budget allocation decisions.
"""
```

The agent returns plain-English insights that update dynamically as filters change — turning any data slice into an instant strategic briefing.

---

## 📦 Dataset

**Source:** [Global Ads Performance Dataset — Kaggle](https://www.kaggle.com/)

```
Campaigns:  1,800
Platforms:  TikTok Ads · Meta Ads · Google Ads · LinkedIn · Twitter
            Pinterest · Snapchat · YouTube · Reddit · Microsoft Ads
Industries: eCommerce · EdTech · Healthcare · Finance · Travel
            Fashion · Food & Beverage · Gaming · Real Estate · B2B SaaS
Countries:  USA · UK · Germany · France · Brazil · India · UAE
            Australia · Canada · Japan · Mexico · Singapore · +more
Metrics:    Ad Spend · Impressions · Clicks · Conversions · Revenue
            CTR · CPC · CPA · ROAS · Engagement Rate
```

---

## 🔗 Related projects

| Project | Description | Link |
|---------|-------------|------|
| 🎓 Harvard ML Portfolio | 12+ ML algorithms · 97.4% accuracy on cancer diagnosis | [Live ↗](https://data-analyst-ss.github.io/ml-portfolio-R) |
| 🛢️ Oil & Gas AI Agent | LangGraph + RAG on drilling data + real-time Slack alerts | Coming soon |

---

<div align="center">

**Built by Sarah Gleice Silva**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-sarahgleicesilva-9A3F4A?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/sarahgleicesilva)
[![Live App](https://img.shields.io/badge/Live_App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://social-media-ai-ossaocyh4myx8srgsri6up.streamlit.app)
[![Portfolio](https://img.shields.io/badge/Harvard_ML_Portfolio-9A3F4A?style=flat-square&logo=github&logoColor=white)](https://data-analyst-ss.github.io/ml-portfolio-R)

*"I don't just analyze data — I engineer intelligence that drives revenue."*

⭐ If this project was useful, consider starring the repo!

</div>
