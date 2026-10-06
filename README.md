# 🚕 Ola AI Mobility Intelligence

A portfolio-grade mobility analytics and data science platform inspired by real-world ride-hailing operations. It uses synthetic mobility data and is **not connected to Ola production systems**.

## What recruiters can explore
- Executive mobility dashboard with KPI storytelling
- Optional animated 3D city traffic model
- Demand vs driver supply intelligence
- Revenue and pricing analytics
- Cancellation intelligence
- Customer experience analytics
- Driver performance analytics
- Cancellation-risk ML model
- Next-hour demand forecasting
- Natural-language → editable safe SQL → results → AI insight
- CSV/Excel data analyst workspace with quality profiling
- Gemini-powered mobility business copilot and executive summary
- Downloadable SQL/results/data outputs

## Stack
Python • Pandas • NumPy • SQL/SQLite • Plotly • Scikit-learn • Streamlit • Gemini • LangChain

## Setup
```powershell
uv sync
copy .env.example .env
# Put your Gemini key in .env
uv run streamlit run app.py
```

## Portfolio note
The live-style metrics and 3D traffic are simulations based on synthetic mobility data. The application does not access live rider, driver, booking, payment or location systems from Ola.
