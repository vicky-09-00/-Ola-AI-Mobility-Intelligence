import re
import time
import os
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from agent.tools import load_rides, build_db, execute_sql, ride_fare_estimate
from agent.agent import ask, generate_sql, generate_insight
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def direct_gemini_sql(question: str, schema: str) -> str:
    """Generate SQL directly with the current Gemini API key.
    This bypasses the older agent retry/fallback path used elsewhere in the app.
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is missing from the project .env file.")

    model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=api_key,
        temperature=0,
        max_retries=0,
    )
    prompt = f"""You are an expert SQLite data analyst.

Schema:
{schema}

User question:
{question}

Return exactly one valid SQLite SELECT query.
Rules: only SELECT; use only the listed tables/columns; no markdown; no explanation.
"""
    response = model.invoke(prompt)
    content = response.content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        content = "\n".join(parts)
    sql = str(content).replace("```sql", "").replace("```SQL", "").replace("```", "").strip()
    if not sql.lower().startswith("select"):
        raise RuntimeError("Gemini returned an invalid SQL response.")
    return sql


def fast_sql_from_question(question: str):
    """Return an instant local SQL template for common mobility questions.
    Complex/ambiguous questions return None and are sent to Gemini.
    """
    q = re.sub(r"\s+", " ", question.lower().strip())

    if ("highest revenue" in q or "most revenue" in q or "maximum revenue" in q or "most money" in q or "highest earning" in q) and "city" in q:
        return "SELECT city, ROUND(SUM(fare), 2) AS revenue\nFROM rides\nGROUP BY city\nORDER BY revenue DESC\nLIMIT 1;"
    if "revenue by city" in q or ("revenue" in q and "city" in q and ("by" in q or "each" in q or "every" in q)):
        return "SELECT city, ROUND(SUM(fare), 2) AS revenue\nFROM rides\nGROUP BY city\nORDER BY revenue DESC;"
    if "cancellation rate" in q and "city" in q:
        return "SELECT city, ROUND(100.0 * SUM(CASE WHEN status = 'Cancelled' THEN 1 ELSE 0 END) / COUNT(*), 2) AS cancellation_rate_pct\nFROM rides\nGROUP BY city\nORDER BY cancellation_rate_pct DESC;"
    if "how many rides were cancelled" in q or ("count" in q and "cancel" in q):
        return "SELECT COUNT(*) AS cancelled_rides\nFROM rides\nWHERE status = 'Cancelled';"
    if "highest average fare" in q or ("average fare" in q and "vehicle" in q):
        return "SELECT vehicle_type, ROUND(AVG(fare), 2) AS average_fare\nFROM rides\nGROUP BY vehicle_type\nORDER BY average_fare DESC\nLIMIT 1;"
    if "average fare by vehicle" in q:
        return "SELECT vehicle_type, ROUND(AVG(fare), 2) AS average_fare\nFROM rides\nGROUP BY vehicle_type\nORDER BY average_fare DESC;"
    if "peak-hour demand by city" in q or ("peak" in q and "demand" in q and "city" in q):
        return "SELECT city, request_hour, COUNT(*) AS ride_requests\nFROM rides\nWHERE request_hour BETWEEN 17 AND 21\nGROUP BY city, request_hour\nORDER BY ride_requests DESC;"
    if "top 10 pickup zones" in q or ("top" in q and "pickup zone" in q and "completed" in q):
        return "SELECT pickup_zone, COUNT(*) AS completed_rides\nFROM rides\nWHERE status = 'Completed'\nGROUP BY pickup_zone\nORDER BY completed_rides DESC\nLIMIT 10;"
    if "total rides" in q or "how many rides" in q:
        return "SELECT COUNT(*) AS total_rides\nFROM rides;"
    if "total revenue" in q:
        return "SELECT ROUND(SUM(fare), 2) AS total_revenue\nFROM rides;"
    if "average rating" in q:
        return "SELECT ROUND(AVG(rating), 2) AS average_rating\nFROM rides\nWHERE rating IS NOT NULL;"
    if "average eta" in q or "avg eta" in q:
        return "SELECT ROUND(AVG(estimated_eta_min), 2) AS average_eta_minutes\nFROM rides;"

    # Extra offline patterns so SQL Analyst remains usable even when Gemini
    # is unavailable, rate-limited, or the API key has a temporary issue.
    if "revenue" in q and "vehicle" in q:
        return "SELECT vehicle_type, ROUND(SUM(fare), 2) AS revenue\nFROM rides\nGROUP BY vehicle_type\nORDER BY revenue DESC;"
    if "revenue" in q:
        return "SELECT ROUND(SUM(fare), 2) AS total_revenue\nFROM rides\nWHERE status = 'Completed';"
    if ("completed" in q or "complete" in q) and "rides" in q:
        return "SELECT COUNT(*) AS completed_rides\nFROM rides\nWHERE status = 'Completed';"
    if "cancel" in q and "vehicle" in q:
        return "SELECT vehicle_type, ROUND(100.0 * SUM(CASE WHEN status = 'Cancelled' THEN 1 ELSE 0 END) / COUNT(*), 2) AS cancellation_rate_pct\nFROM rides\nGROUP BY vehicle_type\nORDER BY cancellation_rate_pct DESC;"
    if "cancel" in q:
        return "SELECT COUNT(*) AS cancelled_rides, ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM rides), 2) AS cancellation_rate_pct\nFROM rides\nWHERE status = 'Cancelled';"
    if "rating" in q and "vehicle" in q:
        return "SELECT vehicle_type, ROUND(AVG(rating), 2) AS average_rating\nFROM rides\nWHERE rating IS NOT NULL\nGROUP BY vehicle_type\nORDER BY average_rating DESC;"
    if "rating" in q and "city" in q:
        return "SELECT city, ROUND(AVG(rating), 2) AS average_rating\nFROM rides\nWHERE rating IS NOT NULL\nGROUP BY city\nORDER BY average_rating DESC;"
    if "distance" in q and "vehicle" in q:
        return "SELECT vehicle_type, ROUND(AVG(distance_km), 2) AS average_distance_km\nFROM rides\nGROUP BY vehicle_type\nORDER BY average_distance_km DESC;"
    if "eta" in q and "city" in q:
        return "SELECT city, ROUND(AVG(estimated_eta_min), 2) AS average_eta_minutes\nFROM rides\nGROUP BY city\nORDER BY average_eta_minutes DESC;"
    if "surge" in q and "city" in q:
        return "SELECT city, ROUND(AVG(surge_multiplier), 2) AS average_surge\nFROM rides\nGROUP BY city\nORDER BY average_surge DESC;"
    if "payment" in q:
        return "SELECT payment_method, COUNT(*) AS rides\nFROM rides\nGROUP BY payment_method\nORDER BY rides DESC;"
    if "top" in q and "city" in q:
        return "SELECT city, COUNT(*) AS rides\nFROM rides\nGROUP BY city\nORDER BY rides DESC\nLIMIT 10;"
    if "top" in q and "vehicle" in q:
        return "SELECT vehicle_type, COUNT(*) AS rides\nFROM rides\nGROUP BY vehicle_type\nORDER BY rides DESC\nLIMIT 10;"
    return None

def fallback_sql_from_question(question: str):
    """Broader offline fallback for SQL Analyst when the LLM cannot be reached."""
    q = re.sub(r"\s+", " ", question.lower().strip())
    if "highest" in q and "revenue" in q and "city" in q:
        return "SELECT city, ROUND(SUM(fare), 2) AS revenue FROM rides WHERE status = 'Completed' GROUP BY city ORDER BY revenue DESC LIMIT 1;"
    if "highest" in q and "fare" in q and "vehicle" in q:
        return "SELECT vehicle_type, ROUND(AVG(fare), 2) AS average_fare FROM rides GROUP BY vehicle_type ORDER BY average_fare DESC LIMIT 1;"
    if "cancellation rate" in q and "city" in q:
        return "SELECT city, ROUND(100.0 * SUM(CASE WHEN status = 'Cancelled' THEN 1 ELSE 0 END) / COUNT(*), 2) AS cancellation_rate_pct FROM rides GROUP BY city ORDER BY cancellation_rate_pct DESC;"
    if "cancel" in q:
        return "SELECT COUNT(*) AS cancelled_rides, ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM rides), 2) AS cancellation_rate_pct FROM rides WHERE status = 'Cancelled';"
    if "revenue" in q and "city" in q:
        return "SELECT city, ROUND(SUM(fare), 2) AS revenue FROM rides WHERE status = 'Completed' GROUP BY city ORDER BY revenue DESC;"
    if "revenue" in q:
        return "SELECT ROUND(SUM(fare), 2) AS total_revenue FROM rides WHERE status = 'Completed';"
    if "average fare" in q and "vehicle" in q:
        return "SELECT vehicle_type, ROUND(AVG(fare), 2) AS average_fare FROM rides GROUP BY vehicle_type ORDER BY average_fare DESC;"
    if "how many rides" in q or "total rides" in q:
        return "SELECT COUNT(*) AS total_rides FROM rides;"
    if "completed" in q and "rides" in q:
        return "SELECT COUNT(*) AS completed_rides FROM rides WHERE status = 'Completed';"
    if "peak" in q and "demand" in q:
        return "SELECT city, request_hour, COUNT(*) AS ride_requests FROM rides WHERE request_hour BETWEEN 17 AND 21 GROUP BY city, request_hour ORDER BY ride_requests DESC;"
    if "top" in q and "pickup" in q:
        return "SELECT pickup_zone, COUNT(*) AS rides FROM rides GROUP BY pickup_zone ORDER BY rides DESC LIMIT 10;"
    return None

st.set_page_config(
    page_title="Ola AI Mobility Intelligence",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# PREMIUM UI
# -----------------------------------------------------------------------------
st.markdown(r"""
<style>
:root{--bg:#101722;--panel:#172130;--panel2:#202c3d;--line:rgba(93,173,255,.18);--text:#f4f7fb;--muted:#a9b6c5;--green:#4ea1ff;--green2:#9ac7ff;--blue:#f3b84b;--danger:#ff6b7a;}
header[data-testid="stHeader"]{height:0!important;min-height:0!important;background:transparent!important;box-shadow:none!important}
header[data-testid="stHeader"]>div{display:none!important}[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}
html,body,.stApp{background:radial-gradient(circle at 12% 0%,rgba(22,214,161,.10),transparent 25%),radial-gradient(circle at 88% 12%,rgba(55,132,255,.07),transparent 23%),linear-gradient(135deg,#040b15 0%,#071321 52%,#06101c 100%)!important;color:var(--text)!important}
.main .block-container{max-width:1540px!important;padding:.25rem 1.65rem 2.4rem!important}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#06111e 0%,#071725 62%,#06111d 100%)!important;border-right:1px solid rgba(130,240,214,.12)!important}
section[data-testid="stSidebar"]>div:first-child{padding-top:.7rem!important}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,section[data-testid="stSidebar"] .stCaption,section[data-testid="stSidebar"] label{color:#9eb4b7!important}
.sidebar-brand{padding:8px 6px 18px}.brand-title{color:#f1fbf8;font-size:21px;font-weight:850;letter-spacing:-.5px}.brand-sub{color:#688187;font-size:11px;margin-top:3px}.sidebar-section{color:#6f898e!important;text-transform:uppercase;letter-spacing:1.4px;font-size:10px;font-weight:800;margin:9px 5px 7px}
section[data-testid="stSidebar"] .stRadio>div{gap:4px!important}section[data-testid="stSidebar"] .stRadio label{min-height:38px!important;padding:7px 10px!important;margin:0!important;border:1px solid transparent!important;border-radius:12px!important;background:transparent!important;color:#a9bec0!important;transition:all .18s ease!important}section[data-testid="stSidebar"] .stRadio label:hover{background:rgba(22,214,161,.08)!important;border-color:rgba(22,214,161,.10)!important;transform:translateX(2px)}section[data-testid="stSidebar"] .stRadio label p{color:#a9bec0!important;font-size:12px!important;font-weight:600!important}section[data-testid="stSidebar"] .stRadio label:has(input:checked){background:linear-gradient(90deg,rgba(22,214,161,.18),rgba(22,214,161,.055))!important;border-color:rgba(22,214,161,.24)!important;box-shadow:inset 3px 0 0 var(--green),0 7px 22px rgba(0,0,0,.14)!important}section[data-testid="stSidebar"] .stRadio label:has(input:checked) p{color:#72efd0!important;font-weight:800!important}section[data-testid="stSidebar"] .stRadio [data-testid="stWidgetLabel"]{display:none!important}
.stack{display:flex;flex-wrap:wrap;gap:7px;margin-top:8px}.pill{padding:6px 8px;border-radius:9px;background:rgba(255,255,255,.035);border:1px solid rgba(255,255,255,.06);color:#a5b8ba;font-size:10px;font-weight:700}.status-box{margin-top:16px;padding:13px;border-radius:14px;background:linear-gradient(135deg,rgba(22,214,161,.11),rgba(22,214,161,.035));border:1px solid rgba(22,214,161,.18)}.dot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#19e1a7;box-shadow:0 0 12px #19e1a7;margin-right:7px}.status-title{color:#74efd2;font-weight:800;font-size:12px}.status-sub{color:#6f8d91;font-size:10px;margin-top:5px}
.topbar{height:58px;margin-bottom:10px;padding:0 14px;display:flex;align-items:center;justify-content:space-between;border:1px solid rgba(255,255,255,.065);border-radius:16px;background:rgba(7,19,31,.78);backdrop-filter:blur(16px)}.top-left{display:flex;align-items:center;gap:11px}.logo{width:32px;height:32px;border-radius:50%;display:grid;place-items:center;background:radial-gradient(circle at 35% 30%,#77ffe1 0 15%,#16d6a1 17% 42%,#087d69 45% 62%,#06273a 65%);box-shadow:0 0 22px rgba(22,214,161,.22);font-size:15px}.top-title{font-size:15px;font-weight:850;color:#eefaf7}.premium{font-size:10px;font-weight:850;color:#6ef0cf;border:1px solid rgba(22,214,161,.45);padding:5px 9px;border-radius:99px;background:rgba(22,214,161,.08)}.top-sub{font-size:11px;color:#657d83}.top-right{font-size:11px;color:#82989c}
.hero{padding:30px 32px 31px;margin:12px 0 14px;border:1px solid rgba(126,240,207,.15);border-radius:22px;background:radial-gradient(circle at 88% 18%,rgba(22,214,161,.13),transparent 24%),radial-gradient(circle at 15% 110%,rgba(85,167,255,.08),transparent 30%),linear-gradient(135deg,rgba(5,17,29,.98),rgba(8,27,42,.88));box-shadow:0 20px 65px rgba(0,0,0,.28),inset 0 1px 0 rgba(255,255,255,.025);position:relative;overflow:hidden}.hero:before{content:"";position:absolute;inset:0;background-image:linear-gradient(rgba(126,240,207,.026) 1px,transparent 1px),linear-gradient(90deg,rgba(126,240,207,.026) 1px,transparent 1px);background-size:38px 38px;mask-image:linear-gradient(90deg,transparent,black 15%,black 85%,transparent);pointer-events:none}.hero>*{position:relative;z-index:1}.badge{display:inline-block;padding:7px 11px;border-radius:999px;background:rgba(22,214,161,.10);border:1px solid rgba(22,214,161,.18);color:#78efd1;font-size:10px;font-weight:850;letter-spacing:.4px}.hero h1{font-size:42px!important;letter-spacing:-1.8px;line-height:1.05;margin:11px 0 8px!important;color:#f2fbf9!important}.accent{color:#18d9a3}.muted{color:#8ba4a7;font-size:13px;line-height:1.55}.mobility-brief{position:relative;display:flex;align-items:center;justify-content:space-between;gap:24px;min-height:82px;margin:0 0 12px;padding:14px 20px;border:1px solid rgba(126,240,207,.12);border-radius:18px;background:linear-gradient(100deg,rgba(7,22,36,.97),rgba(8,27,42,.86),rgba(5,17,29,.96));box-shadow:0 14px 40px rgba(0,0,0,.20),inset 0 1px 0 rgba(255,255,255,.025);overflow:hidden}.mobility-brief:before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:linear-gradient(180deg,#16d6a1,#55a7ff)}.mobility-brief:after{content:"";position:absolute;width:320px;height:150px;right:2%;top:-75px;background:radial-gradient(circle,rgba(22,214,161,.10),transparent 68%);pointer-events:none}.brief-main{position:relative;z-index:2;min-width:270px}.brief-kicker{color:#65e7c7;font-size:9px;font-weight:850;letter-spacing:1.35px;text-transform:uppercase}.brief-title{margin-top:3px;color:#edf9f6;font-size:14px;font-weight:800;letter-spacing:-.2px}.brief-sub{margin-top:3px;color:#718b91;font-size:10px}.brief-metrics{position:relative;z-index:2;display:flex;align-items:center;justify-content:flex-end;gap:4px;flex:1}.brief-metric{min-width:120px;padding:8px 13px;border-left:1px solid rgba(255,255,255,.07)}.brief-label{color:#708a90;font-size:8px;font-weight:800;letter-spacing:.9px;text-transform:uppercase}.brief-value{margin-top:3px;color:#dff8f2;font-size:14px;font-weight:850}.brief-value span{color:#55dcb9;font-size:10px;font-weight:700}@media(max-width:900px){.mobility-brief{align-items:flex-start;flex-direction:column;gap:10px}.brief-metrics{width:100%;justify-content:flex-start;overflow-x:auto}.brief-metric{min-width:108px}}
.card{background:linear-gradient(145deg,rgba(255,255,255,.055),rgba(255,255,255,.022));border:1px solid rgba(255,255,255,.075);border-radius:17px;padding:17px;box-shadow:0 10px 35px rgba(0,0,0,.14)}.kpi{font-size:28px;font-weight:850;color:#7deed0;line-height:1.1;margin-top:5px}.klabel{font-size:10px;letter-spacing:1.1px;color:#86a1a4;font-weight:850}.trend-up{color:#6ef0cf;font-size:11px;font-weight:800}.trend-down{color:#ff8a98;font-size:11px;font-weight:800}
h1,h2,h3,h4,h5,h6{color:#eaf7f4!important}.stMarkdown,.stMarkdown p,.stMarkdown li,.stMarkdown span,.stMarkdown strong,.stMarkdown em{color:#dcefeb!important}.stMarkdown a{color:#72efd0!important}.stCaption,.stCaption p{color:#7f989d!important}[data-testid="stWidgetLabel"] label,[data-testid="stWidgetLabel"] p,[data-testid="stWidgetLabel"] span,label{color:#b9cfcd!important}
.stTextInput input,.stTextArea textarea,.stNumberInput input,[data-baseweb="select"]>div,[data-baseweb="input"]>div,[data-baseweb="textarea"]>div{background:#091a29!important;color:#ecfaf6!important;border:1px solid rgba(126,240,207,.16)!important;border-radius:11px!important;caret-color:#20dca6!important}.stTextInput input::placeholder,.stTextArea textarea::placeholder,.stNumberInput input::placeholder{color:#637d83!important}.stTextInput input:focus,.stTextArea textarea:focus,.stNumberInput input:focus{border-color:rgba(22,214,161,.55)!important;box-shadow:0 0 0 2px rgba(22,214,161,.08)!important}[data-baseweb="select"] *{color:#eaf8f5!important}[data-baseweb="popover"],[data-baseweb="menu"]{background:#0a1a29!important}[data-baseweb="menu"] li{color:#d9eeea!important}
[data-testid="stFileUploader"] section{background:linear-gradient(145deg,#091a29,#081522)!important;border:1px dashed rgba(22,214,161,.32)!important;border-radius:15px!important;padding:16px!important}[data-testid="stFileUploader"] section *{color:#b8cfcd!important}[data-testid="stFileUploader"] button{background:#102738!important;color:#dff8f2!important;border:1px solid rgba(22,214,161,.25)!important}
.stButton>button,.stDownloadButton>button{border-radius:11px!important;border:1px solid rgba(22,214,161,.22)!important;background:linear-gradient(135deg,#0d2a32,#0b202d)!important;color:#dff9f3!important;font-weight:750!important}.stButton>button:hover,.stDownloadButton>button:hover{border-color:rgba(22,214,161,.62)!important;background:linear-gradient(135deg,#103a38,#0b2835)!important;transform:translateY(-1px);box-shadow:0 8px 24px rgba(0,0,0,.22)!important}.stButton>button[kind="primary"]{background:linear-gradient(135deg,#0abf91,#07866f)!important;color:white!important;border-color:#14d6a4!important}
[data-testid="stChatInput"]{background:transparent!important}[data-testid="stChatInput"]>div{background:#091a29!important;border:1px solid rgba(126,240,207,.20)!important;border-radius:15px!important;box-shadow:0 8px 28px rgba(0,0,0,.20)!important}[data-testid="stChatInput"] textarea{background:#091a29!important;color:#effaf7!important}[data-testid="stChatInput"] textarea::placeholder{color:#6f898e!important}[data-testid="stChatMessage"]{background:rgba(9,24,38,.55)!important;border:1px solid rgba(255,255,255,.045)!important;border-radius:14px!important;margin-bottom:8px!important}
[data-testid="stAlert"]{background:#091a29!important;border:1px solid rgba(126,240,207,.13)!important;color:#d8efea!important}[data-testid="stAlert"] p,[data-testid="stAlert"] div,[data-testid="stAlert"] span{color:#dcefeb!important}[data-testid="stExpander"]{background:#091725!important;border:1px solid rgba(126,240,207,.10)!important;border-radius:14px!important}[data-testid="stExpander"] summary,[data-testid="stExpander"] summary span{color:#dcefeb!important}
.codebox{background:#071622;border:1px solid rgba(22,214,161,.12);border-radius:14px;padding:13px}.section-title{font-size:19px;font-weight:850;color:#eefaf7;margin:5px 0}.section-sub{font-size:12px;color:#7f999d;margin-bottom:12px}.insight{background:linear-gradient(135deg,rgba(22,214,161,.10),rgba(85,167,255,.05));border:1px solid rgba(22,214,161,.16);border-radius:15px;padding:15px}.warning-card{background:rgba(255,107,122,.07);border:1px solid rgba(255,107,122,.18);border-radius:15px;padding:15px}.mini{font-size:10px;color:#7d989c;text-transform:uppercase;letter-spacing:1px;font-weight:800}.big-number{font-size:31px;font-weight:900;color:#7deed0}.tag{display:inline-block;padding:5px 8px;border-radius:999px;background:rgba(22,214,161,.08);color:#7deed0;border:1px solid rgba(22,214,161,.16);font-size:10px;font-weight:800}
@media(max-width:900px){.main .block-container{padding:.2rem .85rem 2rem!important}.hero{padding:19px;border-radius:18px}.hero h1{font-size:32px!important}.top-sub{display:none}}
@media(max-width:900px){.main .block-container{padding:.2rem .85rem 2rem!important}.hero{padding:19px;border-radius:18px}.hero h1{font-size:32px!important}.top-sub{display:none}}

/* Visual refresh: graphite/slate palette + visible native sidebar toggle. */
html,body,.stApp{background:radial-gradient(circle at 10% 0%,rgba(78,161,255,.12),transparent 26%),radial-gradient(circle at 92% 10%,rgba(243,184,75,.09),transparent 24%),linear-gradient(135deg,#0f1722 0%,#172130 52%,#111a27 100%)!important;color:#f4f7fb!important}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#111a28 0%,#182334 58%,#111a27 100%)!important;border-right:1px solid rgba(143,177,211,.20)!important}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,section[data-testid="stSidebar"] .stCaption,section[data-testid="stSidebar"] label{color:#c4cfdb!important}
.brand-title{color:#ffffff!important}.brand-sub{color:#91a1b4!important}.sidebar-section{color:#7fa0c1!important}
section[data-testid="stSidebar"] .stRadio label{color:#c5d0dc!important}section[data-testid="stSidebar"] .stRadio label p{color:#c5d0dc!important}section[data-testid="stSidebar"] .stRadio label:hover{background:rgba(78,161,255,.10)!important;border-color:rgba(78,161,255,.22)!important}section[data-testid="stSidebar"] .stRadio label:has(input:checked){background:linear-gradient(90deg,rgba(78,161,255,.20),rgba(78,161,255,.06))!important;border-color:rgba(78,161,255,.34)!important;box-shadow:inset 3px 0 0 #4ea1ff,0 7px 22px rgba(0,0,0,.16)!important}section[data-testid="stSidebar"] .stRadio label:has(input:checked) p{color:#9ac7ff!important}
.pill{background:rgba(255,255,255,.055)!important;border-color:rgba(157,178,204,.14)!important;color:#c7d2df!important}.status-box{background:linear-gradient(135deg,rgba(78,161,255,.12),rgba(78,161,255,.035))!important;border-color:rgba(78,161,255,.22)!important}.status-title{color:#9ac7ff!important}.status-sub{color:#8fa2b8!important}.dot{background:#4ea1ff!important;box-shadow:0 0 12px rgba(78,161,255,.75)!important}
.topbar{background:rgba(20,30,45,.90)!important;border-color:rgba(157,178,204,.13)!important}.top-title{color:#f7f9fc!important}.premium{color:#ffd27a!important;border-color:rgba(243,184,75,.45)!important;background:rgba(243,184,75,.09)!important}.top-sub,.top-right{color:#91a1b4!important}.logo{background:radial-gradient(circle at 35% 30%,#d8ebff 0 15%,#4ea1ff 17% 42%,#245ca6 45% 62%,#162c4a 65%)!important;box-shadow:0 0 22px rgba(78,161,255,.28)!important}
.hero{border-color:rgba(116,169,222,.18)!important;background:radial-gradient(circle at 88% 18%,rgba(78,161,255,.14),transparent 24%),radial-gradient(circle at 15% 110%,rgba(243,184,75,.07),transparent 30%),linear-gradient(135deg,rgba(18,28,42,.98),rgba(29,42,59,.92))!important}.hero:before{background-image:linear-gradient(rgba(145,184,225,.025) 1px,transparent 1px),linear-gradient(90deg,rgba(145,184,225,.025) 1px,transparent 1px)!important}.badge{background:rgba(78,161,255,.10)!important;border-color:rgba(78,161,255,.25)!important;color:#9ac7ff!important}.accent{color:#f3c35f!important}.muted{color:#aebdcd!important}
.mobility-brief{border-color:rgba(116,169,222,.16)!important;background:linear-gradient(100deg,rgba(21,32,47,.98),rgba(29,43,61,.94),rgba(17,27,41,.98))!important}.mobility-brief:before{background:linear-gradient(180deg,#4ea1ff,#f3b84b)!important}.brief-kicker{color:#9ac7ff!important}.brief-title{color:#f1f5f9!important}.brief-sub,.brief-label{color:#8fa2b8!important}.brief-value{color:#edf4fb!important}.brief-value span{color:#ffd27a!important}

/* Sidebar toggle: make Streamlit's native collapse/expand control unmistakable and usable. */
header[data-testid="stHeader"]{height:0!important;min-height:0!important;background:transparent!important;box-shadow:none!important;overflow:visible!important}
header[data-testid="stHeader"]>div{display:block!important;overflow:visible!important}
[data-testid="stSidebarCollapseButton"],[data-testid="stSidebarCollapsedControl"],button[data-testid="stSidebarCollapseButton"],button[data-testid="stSidebarCollapsedControl"]{display:flex!important;align-items:center!important;justify-content:center!important;position:fixed!important;top:12px!important;left:12px!important;z-index:2147483647!important;width:42px!important;height:42px!important;min-width:42px!important;min-height:42px!important;padding:0!important;margin:0!important;border-radius:12px!important;background:rgba(29,47,70,.97)!important;border:1px solid rgba(154,199,255,.55)!important;box-shadow:0 8px 26px rgba(0,0,0,.38),inset 0 1px 0 rgba(255,255,255,.08)!important;color:#ffffff!important;opacity:1!important;visibility:visible!important;pointer-events:auto!important}
[data-testid="stSidebarCollapseButton"] svg,[data-testid="stSidebarCollapsedControl"] svg,button[data-testid="stSidebarCollapseButton"] svg,button[data-testid="stSidebarCollapsedControl"] svg{display:block!important;color:#ffffff!important;stroke:#ffffff!important;fill:none!important;width:22px!important;height:22px!important;opacity:1!important;visibility:visible!important}
[data-testid="stSidebarCollapseButton"]:hover,[data-testid="stSidebarCollapsedControl"]:hover,button[data-testid="stSidebarCollapseButton"]:hover,button[data-testid="stSidebarCollapsedControl"]:hover{background:#315b84!important;border-color:#f3c35f!important;box-shadow:0 10px 30px rgba(0,0,0,.42),0 0 0 3px rgba(243,195,95,.10)!important;transform:scale(1.04)!important}
[data-testid="stSidebarCollapseButton"]:focus-visible,[data-testid="stSidebarCollapsedControl"]:focus-visible,button[data-testid="stSidebarCollapseButton"]:focus-visible,button[data-testid="stSidebarCollapsedControl"]:focus-visible{outline:2px solid #f3c35f!important;outline-offset:2px!important}
/* Keep the main content from sitting under the toggle. */
.main .block-container{padding-top:3.8rem!important}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DATA + STATE
# -----------------------------------------------------------------------------
@st.cache_data

def get_data():
    d = load_rides().copy()
    d['ride_date'] = pd.to_datetime(d['ride_date'])
    return d

df = get_data()
vehicles = sorted(df['vehicle_type'].dropna().astype(str).unique().tolist())
if 'con' not in st.session_state:
    st.session_state.con = build_db(df)
if 'history' not in st.session_state: st.session_state.history=[]
if 'messages' not in st.session_state: st.session_state.messages=[]
if 'sql_editor' not in st.session_state: st.session_state.sql_editor=''
if 'sql_question' not in st.session_state: st.session_state.sql_question=''
if 'sql_result' not in st.session_state: st.session_state.sql_result=None
if 'simulation_tick' not in st.session_state: st.session_state.simulation_tick=0

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def money(x): return f"₹{x/100000:.2f}L" if abs(x)>=100000 else f"₹{x:,.0f}"
def safe_rate(n,d): return (n/d*100) if d else 0

def kpi_card(label,value,trend=None,trend_class='trend-up'):
    extra=f'<div class="{trend_class}">{trend}</div>' if trend else ''
    return f'<div class="card"><div class="klabel">{label}</div><div class="kpi">{value}</div>{extra}</div>'

def render_kpis(items):
    cols=st.columns(len(items))
    for c,(label,value,*rest) in zip(cols,items):
        with c: st.markdown(kpi_card(label,value,*rest),unsafe_allow_html=True)

def demand_table(data):
    g=data.groupby(['city','request_hour']).size().reset_index(name='requests')
    supply=data.groupby(['city','request_hour']).driver_available.mean().reset_index()
    g=g.merge(supply,on=['city','request_hour'])
    g['supply_gap']=np.maximum(g['requests']/max(data['request_hour'].nunique(),1)-g['driver_available']/10,0)
    return g

def mobility_summary(data):
    total=len(data); completed=(data.status=='Completed').sum(); cancelled=(data.status=='Cancelled').sum()
    return {'rides':total,'revenue':data.loc[data.status=='Completed','fare'].sum(),'avg_fare':data.fare.mean(),'cancel_rate':safe_rate(cancelled,total),'rating':data.rating.mean(),'eta':data.estimated_eta_min.mean()}

# -----------------------------------------------------------------------------
# 3D SCENE
# -----------------------------------------------------------------------------
scene_html = r"""<!doctype html><html><head><style>html,body{margin:0;height:100%;overflow:hidden;background:#050b14}#scene{width:100%;height:430px;border-radius:22px;overflow:hidden;position:relative;border:1px solid rgba(126,240,207,.14);box-shadow:0 18px 55px rgba(0,0,0,.25)}.badge3,.scene-live{position:absolute;z-index:20;color:#9ff5dc;font:800 11px Arial;letter-spacing:1px;background:rgba(3,10,20,.78);padding:8px 11px;border:1px solid rgba(159,245,220,.2);border-radius:99px;backdrop-filter:blur(8px)}.badge3{left:20px;top:16px}.scene-live{right:18px;bottom:16px;letter-spacing:0;color:#8ff5d9}.scene-live i{display:inline-block;width:7px;height:7px;border-radius:50%;background:#19e1a7;box-shadow:0 0 10px #19e1a7;margin-right:6px}canvas{display:block}</style></head><body><div id="scene"><div class="badge3">LIVE MOBILITY SIMULATION • 3D CITY</div><div class="scene-live"><i></i>Traffic simulation running</div></div><script type="module">
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
const root=document.getElementById('scene');const scene=new THREE.Scene();scene.background=new THREE.Color(0x07101d);scene.fog=new THREE.Fog(0x07101d,48,155);
const camera=new THREE.PerspectiveCamera(47,root.clientWidth/root.clientHeight,.1,280);camera.position.set(8.5,8.5,25);camera.lookAt(0,1.2,0);
const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setSize(root.clientWidth,root.clientHeight);renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.2;root.appendChild(renderer.domElement);
scene.add(new THREE.HemisphereLight(0x9bded0,0x101a2a,1.8));const sun=new THREE.DirectionalLight(0xbaffec,2.6);sun.position.set(12,24,14);scene.add(sun);const glow=new THREE.PointLight(0x16d6a1,2.2,50);glow.position.set(0,4,8);scene.add(glow);
const ROAD_W=24,ROAD_L=160;const road=new THREE.Mesh(new THREE.PlaneGeometry(ROAD_W,ROAD_L),new THREE.MeshStandardMaterial({color:0x0c1522,roughness:.82,metalness:.08}));road.rotation.x=-Math.PI/2;scene.add(road);
for(const x of [-13,13]){const s=new THREE.Mesh(new THREE.BoxGeometry(1.8,.08,ROAD_L),new THREE.MeshStandardMaterial({color:0x101e2b}));s.position.set(x,.04,0);scene.add(s)}
for(const x of [-11.5,11.5]){const e=new THREE.Mesh(new THREE.BoxGeometry(.14,.045,ROAD_L),new THREE.MeshBasicMaterial({color:0x3d6473}));e.position.set(x,.055,0);scene.add(e)}
const lm=new THREE.MeshBasicMaterial({color:0x9bb7c2});for(let z=-76;z<80;z+=6){for(const x of [-4.5,0,4.5]){const l=new THREE.Mesh(new THREE.BoxGeometry(.11,.035,3.2),lm);l.position.set(x,.06,z);scene.add(l)}}
for(let i=0;i<30;i++){const h=3+Math.random()*13,w=2+Math.random()*4,d=2+Math.random()*4,side=Math.random()<.5?-1:1;const b=new THREE.Mesh(new THREE.BoxGeometry(w,h,d),new THREE.MeshStandardMaterial({color:0x162235,roughness:.9}));b.position.set(side*(19+Math.random()*17),h/2,(Math.random()-.5)*125);scene.add(b)}
function lamp(x,z){const p=new THREE.Mesh(new THREE.CylinderGeometry(.035,.05,3.3,10),new THREE.MeshStandardMaterial({color:0x263847}));p.position.set(x,1.65,z);scene.add(p);const q=new THREE.Mesh(new THREE.SphereGeometry(.1,12,12),new THREE.MeshBasicMaterial({color:0xbaffef}));q.position.set(x,3.25,z);scene.add(q)}for(let z=-60;z<70;z+=15){lamp(-9.5,z);lamp(9.5,z+7)}
function car(color){const g=new THREE.Group();const body=new THREE.Mesh(new THREE.BoxGeometry(1.7,.68,3.45),new THREE.MeshStandardMaterial({color,metalness:.35,roughness:.22,emissive:color,emissiveIntensity:.12}));body.position.y=.7;g.add(body);const roof=new THREE.Mesh(new THREE.BoxGeometry(1.2,.56,1.65),new THREE.MeshStandardMaterial({color:0x102a38,metalness:.15,roughness:.22}));roof.position.set(0,1.18,.05);g.add(roof);const glass=new THREE.Mesh(new THREE.BoxGeometry(1.02,.28,.08),new THREE.MeshBasicMaterial({color:0x75e6d2,transparent:true,opacity:.8}));glass.position.set(0,1.2,-.74);glass.rotation.x=-.15;g.add(glass);for(const x of [-.82,.82])for(const z of [-1.1,1.1]){const w=new THREE.Mesh(new THREE.CylinderGeometry(.31,.31,.21,24),new THREE.MeshStandardMaterial({color:0x020508}));w.rotation.z=Math.PI/2;w.position.set(x,.38,z);g.add(w)}for(const x of [-.45,.45]){const h=new THREE.Mesh(new THREE.BoxGeometry(.27,.13,.1),new THREE.MeshBasicMaterial({color:0xeaffff}));h.position.set(x,.76,-1.76);g.add(h);const r=new THREE.Mesh(new THREE.BoxGeometry(.25,.12,.08),new THREE.MeshBasicMaterial({color:0xff3655}));r.position.set(x,.72,1.76);g.add(r)}const pl=new THREE.PointLight(color,1.1,6);pl.position.set(0,.3,0);g.add(pl);return g}
const cars=[],colors=[0x35d6a2,0x50a5ff,0xf3c76b,0xff6b6b,0xb58cff,0x35d6a2,0x55e6d0,0x70a8ff,0xff9c66,0x7ae6ff];for(let i=0;i<10;i++){const c=car(colors[i%colors.length]);const lane=i%2===0?-4.5:4.5;c.position.set(lane,.08,-66+i*14);c.rotation.y=Math.PI;c.scale.setScalar(.95+Math.random()*.08);c.renderOrder=20;c.traverse(o=>{if(o.isMesh)o.renderOrder=20});scene.add(c);cars.push({mesh:c,speed:.25+Math.random()*.12})}
let t=0;function animate(){requestAnimationFrame(animate);t+=.012;cars.forEach(o=>{o.mesh.position.z+=o.speed;if(o.mesh.position.z>78)o.mesh.position.z=-78});camera.position.x=8.5+Math.sin(t*.18)*.7;camera.lookAt(0,1.2,0);renderer.render(scene,camera)}animate();window.addEventListener('resize',()=>{camera.aspect=root.clientWidth/root.clientHeight;camera.updateProjectionMatrix();renderer.setSize(root.clientWidth,root.clientHeight)})</script></body></html>"""

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-brand"><div class="brand-title">🚕 Ola AI Mobility</div><div class="brand-sub">Mobility Intelligence Platform</div></div><div class="sidebar-section">Analytics Workspace</div>',unsafe_allow_html=True)
    page_label=st.radio('Workspace',[
        '🏠  Executive Dashboard','🚕  Ride Assistant','🚦  Mobility Monitor','📈  Demand & Supply','💰  Revenue & Pricing','❌  Cancellation Intelligence','⭐  Customer Experience','👨‍✈️  Driver Analytics','🤖  Predictive Analytics','⌘  SQL Analyst','📂  Data Analyst','📊  Make Dashboard','🧠  AI Business Copilot','🕘  SQL History'],label_visibility='collapsed')
    page=page_label.split('  ',1)[1]
    st.markdown('<div class="sidebar-section">Stack</div><div class="stack"><span class="pill">🐍 Python</span><span class="pill">⚡ Streamlit</span><span class="pill">✦ Gemini</span><span class="pill">🔗 LangChain</span><span class="pill">🗄 SQL</span><span class="pill">◫ Pandas</span><span class="pill">ML</span></div><div class="status-box"><div class="status-title"><span class="dot"></span>System Online</div><div class="status-sub">Synthetic mobility data • AI ready • Portfolio simulation</div></div>',unsafe_allow_html=True)
    st.caption(f"{len(df):,} rides • {df.city.nunique()} cities • 60-day demo")
    st.checkbox('Show 3D mobility model',value=True,key='show_3d')
    if st.button('Clear SQL history',use_container_width=True):
        st.session_state.history=[];st.rerun()

# -----------------------------------------------------------------------------
# TOPBAR + HERO
# -----------------------------------------------------------------------------
st.markdown('<div class="topbar"><div class="top-left"><div class="logo">✦</div><div class="top-title">Ola AI Mobility Intelligence</div><div class="premium">DATA SCIENCE DEMO</div><div class="top-sub">Analytics • Prediction • AI Decision Support</div></div><div class="top-right">● Simulation Environment&nbsp;&nbsp; • &nbsp;&nbsp;LLM Ready</div></div>',unsafe_allow_html=True)

# Mobility Operations Brief — visual-only section; existing analytics remain unchanged.
_completed_pct = (df['status'].eq('Completed').mean() * 100) if 'status' in df.columns and len(df) else 0
_peak_hour = int(df['request_hour'].value_counts().idxmax()) if 'request_hour' in df.columns and len(df) else 0
_avg_surge = float(df['surge_multiplier'].mean()) if 'surge_multiplier' in df.columns and len(df) else 1.0
_city_count = int(df['city'].nunique()) if 'city' in df.columns else 0

st.markdown(f'''<div class="mobility-brief"><div class="brief-main"><div class="brief-kicker">MOBILITY OPERATIONS BRIEF</div><div class="brief-title">Network snapshot & operating signals</div><div class="brief-sub">A decision-ready view of demand, service quality and pricing pressure</div></div><div class="brief-metrics"><div class="brief-metric"><div class="brief-label">Network</div><div class="brief-value">{_city_count} <span>cities</span></div></div><div class="brief-metric"><div class="brief-label">Completion</div><div class="brief-value">{_completed_pct:.1f}<span>%</span></div></div><div class="brief-metric"><div class="brief-label">Peak Hour</div><div class="brief-value">{_peak_hour:02d}:00</div></div><div class="brief-metric"><div class="brief-label">Avg Surge</div><div class="brief-value">{_avg_surge:.2f}<span>x</span></div></div></div></div>''',unsafe_allow_html=True)

st.markdown('<div class="hero"><span class="badge">✦ OLA-INSPIRED MOBILITY ANALYTICS</span><h1>Turn ride data into <span class="accent">business decisions.</span></h1><div class="muted">Demand & supply intelligence • Revenue & pricing • Cancellation risk • Customer experience • Predictive analytics • AI-powered recommendations</div></div>',unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# EXECUTIVE DASHBOARD
# -----------------------------------------------------------------------------
if page=='Executive Dashboard':
    s=mobility_summary(df)
    render_kpis([('TOTAL RIDES',f"{s['rides']:,}"),('REVENUE',money(s['revenue'])),('AVG FARE',f"₹{s['avg_fare']:,.0f}"),('CANCELLATION',f"{s['cancel_rate']:.1f}%"),('AVG RATING',f"{s['rating']:.2f}/5"),('AVG ETA',f"{s['eta']:.1f} min")])
    c1,c2=st.columns([1.6,1])
    with c1:
        st.markdown('<div class="section-title">Mobility command center</div><div class="section-sub">Use the 3D switch in the sidebar to change between the live city simulation and analytical view.</div>',unsafe_allow_html=True)
        if st.session_state.show_3d:
            components.html(scene_html,height=430,scrolling=False)
        else:
            city=df.groupby('city',as_index=False).agg(rides=('ride_id','count'),revenue=('fare','sum'),eta=('estimated_eta_min','mean')).sort_values('rides',ascending=False)
            fig=px.bar(city,x='city',y='rides',color='eta',template='plotly_dark',title='Ride demand by city',labels={'rides':'Requests','eta':'Avg ETA'})
            fig.update_layout(height=430,margin=dict(l=20,r=20,t=55,b=20),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig,use_container_width=True)
    with c2:
        city=df.groupby('city',as_index=False).agg(revenue=('fare','sum'),rides=('ride_id','count'),cancel=('status',lambda x:(x=='Cancelled').mean()*100)).sort_values('revenue',ascending=False)
        st.plotly_chart(px.bar(city,x='revenue',y='city',orientation='h',template='plotly_dark',title='Revenue by city'),use_container_width=True)
    l,r=st.columns(2)
    with l:
        daily=df.groupby('ride_date',as_index=False).agg(rides=('ride_id','count'),revenue=('fare','sum'))
        st.plotly_chart(px.line(daily,x='ride_date',y='rides',template='plotly_dark',title='Ride demand trend'),use_container_width=True)
    with r:
        v=df.groupby('vehicle_type',as_index=False).agg(rides=('ride_id','count'),revenue=('fare','sum'))
        st.plotly_chart(px.bar(v,x='vehicle_type',y='revenue',template='plotly_dark',title='Revenue by vehicle type'),use_container_width=True)
    top_city=city.iloc[0]
    st.markdown(f'<div class="insight"><b>✦ Executive Insight</b><br>{top_city.city} currently leads revenue at <b>{money(top_city.revenue)}</b>. Use Demand & Supply and Cancellation Intelligence to identify whether growth is constrained by driver availability or rider behaviour.</div>',unsafe_allow_html=True)
    snapshot=(f"Ola AI Mobility Intelligence — Executive Snapshot\n\nTotal rides: {s['rides']:,}\nRevenue: {money(s['revenue'])}\nAverage fare: ₹{s['avg_fare']:,.0f}\nCancellation rate: {s['cancel_rate']:.1f}%\nAverage rating: {s['rating']:.2f}/5\nAverage ETA: {s['eta']:.1f} min\nTop revenue city: {top_city.city}\n\nPortfolio simulation using synthetic mobility data.")
    st.download_button('⬇ Download executive snapshot',snapshot,'executive_snapshot.txt','text/plain',use_container_width=True)
    st.caption('Portfolio simulation using synthetic mobility data; not connected to Ola production systems.')

# -----------------------------------------------------------------------------
# RIDE ASSISTANT
# -----------------------------------------------------------------------------
elif page=='Ride Assistant':
    st.markdown('<div class="section-title">🚕 Ride Assistant</div><div class="section-sub">A portfolio-safe ride planning demo using the same mobility pricing logic as the analytics layer.</div>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: city=st.selectbox('Pickup city',sorted(df.city.unique()),key='fare_city')
    with c2: vehicle=st.selectbox('Vehicle type',['Ola Auto','Ola Mini','Ola Prime','Ola Prime SUV'],key='fare_vehicle')
    with c3: distance=st.number_input('Distance (km)',1.0,100.0,8.0,.5,key='fare_distance')
    if st.button('Estimate fare',type='primary'):
        fare=ride_fare_estimate(distance,vehicle)
        st.success(f'Estimated {vehicle} fare in {city}: ₹{fare:,.0f}')
    render_kpis([('SELECTED CITY',city),('VEHICLE',vehicle),('DISTANCE',f'{distance:.1f} km'),('ESTIMATED FARE',f'₹{ride_fare_estimate(distance,vehicle):,.0f}')])
    st.markdown('<div class="insight"><b>How this supports analytics</b><br>The estimator is intentionally a demo workflow. Pricing decisions should be validated against revenue, surge, distance, cancellation and customer-experience analytics rather than treated as a live booking quote.</div>',unsafe_allow_html=True)
    st.info('Portfolio demo only — no live Ola booking, driver, payment or location system is connected.')

# -----------------------------------------------------------------------------
# MOBILITY MONITOR
# -----------------------------------------------------------------------------
elif page=='Mobility Monitor':
    st.markdown('<div class="section-title">🚦 Live Mobility Monitor</div><div class="section-sub">A real-time-style simulation of ride requests, active rides, driver supply and operational events.</div>',unsafe_allow_html=True)
    if st.button('↻ Refresh simulation',type='primary'):
        st.session_state.simulation_tick+=1
    tick=st.session_state.simulation_tick
    now=pd.Timestamp.now()
    rng=np.random.default_rng(100+tick)
    active=int(1100+rng.integers(-80,120)); requests=int(320+rng.integers(-35,55)); drivers=int(310+rng.integers(-25,35)); eta=float(np.clip(6.1+rng.normal(0,.5),3.8,9.5)); cancel=float(np.clip(8.2+rng.normal(0,1.1),4,15))
    render_kpis([('ACTIVE RIDES',f'{active:,}', '● LIVE'),('REQUESTS / MIN',f'{requests:,}','▲ demand'),('AVAILABLE DRIVERS',f'{drivers:,}','● supply'),('AVG ETA',f'{eta:.1f} min'),('CANCELLATION',f'{cancel:.1f}%')])
    if st.session_state.show_3d:
        components.html(scene_html,height=430,scrolling=False)
    else:
        st.info('3D model is off. Enable “Show 3D mobility model” in the sidebar to return to the animated city view.')
    events=[('11:59:42','Ride accepted','Bengaluru • Whitefield'),('11:59:39','Driver assigned','Gurugram • Cyber City'),('11:59:35','Demand spike','Delhi • Connaught Place'),('11:59:31','Ride completed','Mumbai • BKC'),('11:59:28','Cancellation risk','Noida • Sector 18')]
    ev=pd.DataFrame(events,columns=['Time','Event','Context'])
    st.markdown('<div class="section-title">Live operational events</div>',unsafe_allow_html=True)
    st.dataframe(ev,use_container_width=True,hide_index=True)

# -----------------------------------------------------------------------------
# DEMAND & SUPPLY
# -----------------------------------------------------------------------------
elif page=='Demand & Supply':
    st.markdown('<div class="section-title">📈 Demand & Supply Intelligence</div><div class="section-sub">Find where rider demand is likely to outpace driver availability.</div>',unsafe_allow_html=True)
    city=st.selectbox('City',sorted(df.city.unique()),key='ds_city'); hour=st.slider('Hour of day',0,23,18,key='ds_hour')
    subset=df[df.city==city]
    byhour=subset.groupby('request_hour').agg(requests=('ride_id','count'),drivers=('driver_available','mean'),eta=('estimated_eta_min','mean'),cancel=('status',lambda x:(x=='Cancelled').mean()*100)).reset_index()
    peak=byhour.iloc[(byhour['requests']-byhour['drivers']/10).abs().argmax()]
    selected=byhour[byhour.request_hour==hour]
    if selected.empty: selected=byhour.iloc[[0]]
    row=selected.iloc[0]; gap=max(float(row.requests-row.drivers/10),0)
    render_kpis([('REQUESTS',f"{int(row.requests):,}"),('DRIVER SUPPLY',f"{int(row.drivers):,}"),('SUPPLY GAP',f"{gap:,.0f}"),('AVG ETA',f"{row.eta:.1f} min"),('CANCEL RISK',f"{row.cancel:.1f}%")])
    l,r=st.columns(2)
    with l:
        fig=go.Figure();fig.add_trace(go.Scatter(x=byhour.request_hour,y=byhour.requests,mode='lines+markers',name='Requests'));fig.add_trace(go.Scatter(x=byhour.request_hour,y=byhour.drivers/10,mode='lines+markers',name='Driver capacity (scaled)'));fig.update_layout(template='plotly_dark',title=f'Demand vs supply — {city}',xaxis_title='Hour',yaxis_title='Volume',height=380,paper_bgcolor='rgba(0,0,0,0)');st.plotly_chart(fig,use_container_width=True)
    with r:
        heat=subset.groupby(['request_hour','pickup_zone']).size().reset_index(name='requests')
        st.plotly_chart(px.density_heatmap(heat,x='request_hour',y='pickup_zone',z='requests',template='plotly_dark',title='Demand heatmap by zone/hour'),use_container_width=True)
    if gap>20:
        st.markdown(f'<div class="warning-card"><b>⚠ Supply gap detected</b><br>{city} at {hour:02d}:00 shows an estimated capacity gap of <b>{gap:.0f}</b> units. Increasing driver availability around high-demand zones could reduce ETA and cancellation pressure.</div>',unsafe_allow_html=True)
    else:
        st.markdown('<div class="insight"><b>✓ Balanced window</b><br>Current demand is relatively close to estimated driver capacity. Monitor adjacent peak hours for early supply pressure.</div>',unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# REVENUE & PRICING
# -----------------------------------------------------------------------------
elif page=='Revenue & Pricing':
    st.markdown('<div class="section-title">💰 Revenue & Pricing Intelligence</div><div class="section-sub">Understand which cities, vehicles and trip profiles drive monetization.</div>',unsafe_allow_html=True)
    completed=df[df.status=='Completed']; rev=completed.fare.sum()
    render_kpis([('COMPLETED REVENUE',money(rev)),('AVG FARE',f'₹{completed.fare.mean():.0f}'),('REVENUE / KM',f'₹{(completed.fare/completed.distance_km).mean():.1f}'),('AVG SURGE',f'{completed.surge_multiplier.mean():.2f}x')])
    l,r=st.columns(2)
    with l:
        g=completed.groupby('city',as_index=False).fare.sum().sort_values('fare',ascending=False);st.plotly_chart(px.bar(g,x='city',y='fare',template='plotly_dark',title='Revenue by city'),use_container_width=True)
    with r:
        g=completed.groupby('vehicle_type',as_index=False).agg(revenue=('fare','sum'),avg_fare=('fare','mean'));st.plotly_chart(px.bar(g,x='vehicle_type',y='revenue',template='plotly_dark',title='Revenue by vehicle'),use_container_width=True)
    st.plotly_chart(px.scatter(completed.sample(min(1600,len(completed)),random_state=42),x='distance_km',y='fare',color='vehicle_type',size='surge_multiplier',template='plotly_dark',title='Fare vs distance — pricing behaviour'),use_container_width=True)
    top=completed.groupby('city').fare.sum().idxmax();st.markdown(f'<div class="insight"><b>✦ Pricing insight</b><br><b>{top}</b> is the current top revenue market. Fare and distance should be interpreted alongside cancellation and ETA metrics before making pricing decisions.</div>',unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CANCELLATION
# -----------------------------------------------------------------------------
elif page=='Cancellation Intelligence':
    st.markdown('<div class="section-title">❌ Cancellation Intelligence</div><div class="section-sub">Identify operational conditions associated with ride cancellations.</div>',unsafe_allow_html=True)
    cancel=df[df.status=='Cancelled']; render_kpis([('CANCELLED RIDES',f'{len(cancel):,}'),('CANCELLATION RATE',f'{safe_rate(len(cancel),len(df)):.1f}%'),('AVG ETA — CANCELLED',f'{cancel.estimated_eta_min.mean():.1f} min'),('AVG SURGE — CANCELLED',f'{cancel.surge_multiplier.mean():.2f}x')])
    l,r=st.columns(2)
    with l:
        g=df.groupby('request_hour').status.apply(lambda x:(x=='Cancelled').mean()*100).reset_index(name='cancel_rate');st.plotly_chart(px.line(g,x='request_hour',y='cancel_rate',markers=True,template='plotly_dark',title='Cancellation rate by hour'),use_container_width=True)
    with r:
        g=df.groupby('city').status.apply(lambda x:(x=='Cancelled').mean()*100).reset_index(name='cancel_rate').sort_values('cancel_rate',ascending=False);st.plotly_chart(px.bar(g,x='city',y='cancel_rate',template='plotly_dark',title='Cancellation rate by city'),use_container_width=True)
    g=df.groupby('vehicle_type').status.apply(lambda x:(x=='Cancelled').mean()*100).reset_index(name='cancel_rate');st.plotly_chart(px.bar(g,x='vehicle_type',y='cancel_rate',template='plotly_dark',title='Cancellation by vehicle type'),use_container_width=True)
    worst=g.sort_values('cancel_rate',ascending=False).iloc[0];st.markdown(f'<div class="warning-card"><b>Operational signal</b><br>{worst.vehicle_type} has the highest cancellation rate in this simulation at <b>{worst.cancel_rate:.1f}%</b>. Investigate supply, ETA, surge and pickup-zone conditions before operational changes.</div>',unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CUSTOMER EXPERIENCE
# -----------------------------------------------------------------------------
elif page=='Customer Experience':
    st.markdown('<div class="section-title">⭐ Customer Experience Analytics</div><div class="section-sub">Connect ETA, fare, rating and cancellation behaviour to rider experience.</div>',unsafe_allow_html=True)
    rated=df.dropna(subset=['rating']);render_kpis([('AVG RATING',f'{rated.rating.mean():.2f}/5'),('5★ SHARE',f'{safe_rate((rated.rating>=4.8).sum(),len(rated)):.1f}%'),('AVG ETA',f'{df.estimated_eta_min.mean():.1f} min'),('AVG FARE',f'₹{df.fare.mean():.0f}')])
    l,r=st.columns(2)
    with l: st.plotly_chart(px.histogram(rated,x='rating',nbins=14,template='plotly_dark',title='Rating distribution'),use_container_width=True)
    with r: st.plotly_chart(px.scatter(rated.sample(min(1600,len(rated)),random_state=7),x='estimated_eta_min',y='rating',color='status',template='plotly_dark',title='ETA vs rating'),use_container_width=True)
    corr=rated[['rating','estimated_eta_min','fare','distance_km','surge_multiplier']].corr()['rating'].drop('rating').sort_values();st.dataframe(corr.rename('Correlation with rating').to_frame(),use_container_width=True)
    st.markdown('<div class="insight"><b>✦ Experience lens</b><br>Use ETA, cancellation and surge together with rating. A single KPI can hide the operational driver behind a poor customer experience.</div>',unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DRIVER ANALYTICS
# -----------------------------------------------------------------------------
elif page=='Driver Analytics':
    st.markdown('<div class="section-title">👨‍✈️ Driver Performance Analytics</div><div class="section-sub">Operational view of driver activity without exposing personal identities.</div>',unsafe_allow_html=True)
    g=df.groupby('driver_id').agg(rides=('ride_id','count'),completed=('status',lambda x:(x=='Completed').sum()),avg_rating=('rating','mean'),avg_eta=('estimated_eta_min','mean'),revenue=('fare','sum')).reset_index()
    g['completion_rate']=g.completed/g.rides*100
    render_kpis([('DRIVERS',f'{len(g):,}'),('AVG RIDES / DRIVER',f'{g.rides.mean():.1f}'),('AVG COMPLETION',f'{g.completion_rate.mean():.1f}%'),('AVG DRIVER RATING',f'{g.avg_rating.mean():.2f}')])
    l,r=st.columns(2)
    with l: st.plotly_chart(px.histogram(g,x='rides',nbins=25,template='plotly_dark',title='Ride distribution per driver'),use_container_width=True)
    with r: st.plotly_chart(px.scatter(g.sample(min(1200,len(g)),random_state=1),x='rides',y='completion_rate',size='revenue',template='plotly_dark',title='Driver activity vs completion'),use_container_width=True)
    top=g.sort_values(['completion_rate','rides'],ascending=False).head(15);st.dataframe(top[['driver_id','rides','completed','completion_rate','avg_rating','avg_eta','revenue']],use_container_width=True,hide_index=True)

# -----------------------------------------------------------------------------
# PREDICTIVE ANALYTICS
# -----------------------------------------------------------------------------
elif page=='Predictive Analytics':
    st.markdown('<div class="section-title">🤖 Predictive Mobility Analytics</div><div class="section-sub">Portfolio-grade machine learning layer: cancellation risk and next-hour demand estimation.</div>',unsafe_allow_html=True)
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.metrics import accuracy_score, mean_absolute_error
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        ml_ok=True
    except Exception:
        ml_ok=False
    if not ml_ok:
        st.error('scikit-learn is required for predictive analytics. Run: uv sync')
    else:
        tab1,tab2=st.tabs(['⚠ Cancellation Risk','📈 Demand Forecast'])
        with tab1:
            st.markdown('#### Predict cancellation probability for a hypothetical ride')
            a,b,c=st.columns(3)
            with a: city=st.selectbox('City',sorted(df.city.unique()),key='ml_city'); vehicle=st.selectbox('Vehicle',vehicles,key='ml_vehicle')
            with b: dist=st.number_input('Distance (km)',1.0,60.0,8.0,key='ml_dist'); hour=st.slider('Pickup hour',0,23,18,key='ml_hour')
            with c: eta=st.number_input('Estimated ETA (min)',2.0,25.0,8.0,key='ml_eta'); surge=st.number_input('Surge multiplier',0.9,2.0,1.2,.05,key='ml_surge')
            X=df[['city','vehicle_type','distance_km','request_hour','estimated_eta_min','surge_multiplier']].copy();y=(df.status=='Cancelled').astype(int)
            cat=['city','vehicle_type'];num=['distance_km','request_hour','estimated_eta_min','surge_multiplier'];pre=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),cat)],remainder='passthrough');model=Pipeline([('pre',pre),('rf',RandomForestClassifier(n_estimators=140,max_depth=8,random_state=42,class_weight='balanced'))]);Xt,Xv,yt,yv=train_test_split(X,y,test_size=.25,random_state=42,stratify=y);model.fit(Xt,yt);acc=accuracy_score(yv,model.predict(Xv));sample=pd.DataFrame([{'city':city,'vehicle_type':vehicle,'distance_km':dist,'request_hour':hour,'estimated_eta_min':eta,'surge_multiplier':surge}]);prob=float(model.predict_proba(sample)[0,1]);render_kpis([('CANCELLATION PROBABILITY',f'{prob*100:.1f}%'),('MODEL VALIDATION ACCURACY',f'{acc*100:.1f}%')])
            if prob>=.6: st.markdown(f'<div class="warning-card"><b>HIGH RISK</b><br>Predicted cancellation probability is <b>{prob*100:.1f}%</b>. Review driver supply, pickup distance and peak-hour conditions.</div>',unsafe_allow_html=True)
            else: st.markdown(f'<div class="insight"><b>LOWER RISK</b><br>Predicted cancellation probability is <b>{prob*100:.1f}%</b> under the selected conditions.</div>',unsafe_allow_html=True)
        with tab2:
            st.markdown('#### Estimate demand for the next operating hour')
            city=st.selectbox('Forecast city',sorted(df.city.unique()),key='fc_city'); hour=st.slider('Forecast hour',0,23,19,key='fc_hour')
            hourly=df.groupby(['ride_date','city','request_hour']).size().reset_index(name='requests');hourly['date']=pd.to_datetime(hourly.ride_date);hourly['dow']=hourly.date.dt.dayofweek
            train=hourly[hourly.city==city].copy();
            if len(train)>=30:
                X=train[['request_hour','dow']];y=train.requests
                reg=RandomForestRegressor(n_estimators=160,max_depth=10,random_state=42);reg.fit(X,y);pred=max(0,float(reg.predict(pd.DataFrame([{'request_hour':hour,'dow':pd.Timestamp.now().dayofweek}]))[0]));mae=mean_absolute_error(y,reg.predict(X));render_kpis([('PREDICTED REQUESTS',f'{pred:,.0f}'),('TRAINING MAE',f'{mae:.1f}'),('FORECAST HORIZON','NEXT HOUR')]);st.plotly_chart(px.line(train.tail(120),x='date',y='requests',color='request_hour',template='plotly_dark',title=f'Historical demand — {city}'),use_container_width=True)
                st.markdown(f'<div class="insight"><b>✦ Forecast recommendation</b><br>Plan capacity for approximately <b>{pred:,.0f} requests</b> in the selected hour. Forecast is a portfolio model, not a production demand forecast.</div>',unsafe_allow_html=True)
            else: st.info('Not enough historical observations for this city.')

# -----------------------------------------------------------------------------
# SQL ANALYST
# -----------------------------------------------------------------------------
elif page=='SQL Analyst':
    st.markdown('<div class="section-title">⌘ SQL Analyst</div><div class="section-sub">Ask a business question, review/edit the generated SQL, execute safely, and turn results into business recommendations.</div>',unsafe_allow_html=True)
    examples=['Which city has the highest revenue?','What is the cancellation rate by city?','Which vehicle type has the highest average fare?','Show peak-hour demand by city.','Find the top 10 pickup zones by completed rides.']
    ex=st.selectbox('Business question example',['Custom']+examples);q=st.text_input('Business question',value='' if ex=='Custom' else ex,placeholder='e.g. Which city has the highest cancellation rate during peak hours?')
    if st.button('✦ Generate SQL',type='primary') and q:
        schema="""rides(ride_id INTEGER, city TEXT, vehicle_type TEXT, distance_km REAL, fare REAL, status TEXT, payment_method TEXT, rating REAL, ride_date TEXT, request_hour INTEGER, driver_available INTEGER, estimated_eta_min REAL, surge_multiplier REAL, pickup_zone TEXT, driver_id TEXT, customer_id TEXT)"""
        generated=fast_sql_from_question(q)
        if generated is not None:
            st.session_state.sql_question=q
            st.session_state.sql_editor=generated
            st.session_state.sql_generation_source='⚡ Instant local SQL'
        else:
            try:
                st.session_state.sql_generation_source='✦ Gemini 3.5 Flash Lite'
                with st.spinner('Generating SQL with Gemini...'):
                    generated=direct_gemini_sql(q,schema)
                st.session_state.sql_question=q
                st.session_state.sql_editor=generated
            except Exception as e:
                fallback = fallback_sql_from_question(q)
                st.session_state.sql_question=q
                if fallback:
                    st.session_state.sql_editor=fallback
                    st.session_state.sql_generation_source='⚡ Offline SQL fallback'
                    st.warning(f'Gemini could not generate this query, so an offline SQL template was used. Details: {e}')
                else:
                    st.session_state.sql_generation_source='⚠ Gemini connection issue'
                    st.error(f'Gemini connection failed: {e}')
    source_label=st.session_state.get('sql_generation_source','')
    if source_label:
        st.caption(source_label + ' • simple mobility questions are answered without an LLM request')
    st.markdown('<div class="mini">SQL QUERY — EDITABLE</div>',unsafe_allow_html=True)
    edited=st.text_area('SQL Query',key='sql_editor',height=170,label_visibility='collapsed',placeholder='Type or edit a SELECT query here...')
    c1,c2=st.columns(2)
    with c1: run=st.button('▶ Execute SQL',type='primary',use_container_width=True)
    with c2:
        if st.button('↺ Clear SQL',use_container_width=True): st.session_state.sql_editor='';st.session_state.sql_result=None;st.rerun()
    if run and edited.strip():
        unsafe=re.search(r'\b(insert|update|delete|drop|alter|attach|pragma|create|replace|vacuum|reindex|detach)\b',edited,re.I)
        if unsafe or not edited.strip().lower().startswith('select'): st.error('Blocked unsafe SQL. Only SELECT queries are allowed.')
        else:
            try:
                result=execute_sql(st.session_state.con,edited.strip());st.session_state.sql_result=result;st.session_state.history.insert(0,{'time':datetime.now().strftime('%H:%M:%S'),'question':st.session_state.sql_question or 'Manual SQL','sql':edited.strip()});st.session_state.history=st.session_state.history[:50]
            except Exception as e: st.error(str(e))
    if st.session_state.sql_result is not None:
        result=st.session_state.sql_result;st.markdown('<div class="section-title">Query Result</div>',unsafe_allow_html=True);st.dataframe(result,use_container_width=True,height=280)
        if st.session_state.sql_question:
            with st.expander('✦ AI Business Insight',expanded=False):
                st.caption('Optional Gemini analysis — click only when you want an AI interpretation of the result.')
                if st.button('Generate AI Business Insight', key='generate_sql_insight', use_container_width=True):
                    try:
                        with st.spinner('Generating business insight...'):
                            insight=generate_insight(st.session_state.sql_question,result.to_string(index=False))
                        st.write(insight)
                    except Exception as e: st.warning(f'AI insight unavailable: {e}')
        a,b=st.columns(2)
        with a: st.download_button('⬇ Download SQL result CSV',result.to_csv(index=False).encode(),'sql_result.csv','text/csv',use_container_width=True)
        with b: st.download_button('⬇ Download SQL query',edited.strip().encode(),'query.sql','text/plain',use_container_width=True)
    with st.expander('🧩 SQL Schema Explorer'):
        st.code("""rides\n├── ride_id INTEGER\n├── city TEXT\n├── vehicle_type TEXT\n├── distance_km REAL\n├── fare REAL\n├── status TEXT\n├── payment_method TEXT\n├── rating REAL\n├── ride_date TEXT\n├── request_hour INTEGER\n├── driver_available INTEGER\n├── estimated_eta_min REAL\n├── surge_multiplier REAL\n├── pickup_zone TEXT\n├── driver_id TEXT\n└── customer_id TEXT""",language='text')

# -----------------------------------------------------------------------------
# DATA ANALYST
# -----------------------------------------------------------------------------
elif page=='Data Analyst':
    st.markdown('<div class="section-title">📂 Data Analyst Workspace</div><div class="section-sub">Upload CSV/Excel, profile the data, inspect quality, visualize patterns and export results.</div>',unsafe_allow_html=True)
    uploaded=st.file_uploader('Upload CSV or Excel',type=['csv','xlsx'],help='CSV and XLSX supported')
    work=df.copy();source='Synthetic mobility demo dataset'
    if uploaded:
        try:
            work=pd.read_csv(uploaded) if uploaded.name.lower().endswith('.csv') else pd.read_excel(uploaded);source=uploaded.name;st.success(f'Loaded {source} • {len(work):,} rows • {work.shape[1]} columns')
        except Exception as e: st.error(f'Could not read file: {e}')
    render_kpis([('ROWS',f'{len(work):,}'),('COLUMNS',f'{work.shape[1]:,}'),('MISSING VALUES',f'{int(work.isna().sum().sum()):,}'),('DUPLICATES',f'{int(work.duplicated().sum()):,}')])
    tabs=st.tabs(['Preview','Quality','Visualization','Profile'])
    with tabs[0]: st.dataframe(work.head(250),use_container_width=True,height=370)
    with tabs[1]:
        q=pd.DataFrame({'column':work.columns,'dtype':[str(x) for x in work.dtypes],'missing':work.isna().sum().values,'missing_%':(work.isna().mean()*100).round(2).values,'unique':[work[c].nunique(dropna=True) for c in work.columns]});st.dataframe(q,use_container_width=True,hide_index=True)
    with tabs[2]:
        numeric=work.select_dtypes(include='number').columns.tolist();
        if numeric:
            metric=st.selectbox('Numeric field',numeric);st.plotly_chart(px.histogram(work,x=metric,template='plotly_dark',title=f'{metric} distribution'),use_container_width=True)
        catcols=work.select_dtypes(exclude='number').columns.tolist();
        if catcols:
            cat=st.selectbox('Categorical field',catcols);counts=work[cat].astype(str).value_counts().head(15).reset_index();counts.columns=[cat,'count'];st.plotly_chart(px.bar(counts,x=cat,y='count',template='plotly_dark',title=f'Top {cat} values'),use_container_width=True)
    with tabs[3]:
        st.write(work.describe(include='all').transpose().fillna('').head(40))
    st.download_button('⬇ Download current dataset',work.to_csv(index=False).encode(),'current_dataset.csv','text/csv',use_container_width=True)

# -----------------------------------------------------------------------------
# MAKE DASHBOARD — AUTO BI WORKSPACE
# -----------------------------------------------------------------------------
elif page=='Make Dashboard':
    st.markdown('<div class="section-title">📊 Make Dashboard</div><div class="section-sub">Build a Power BI-style interactive dashboard instantly from the demo mobility dataset or your own CSV / Excel file.</div>',unsafe_allow_html=True)

    uploaded_dash = st.file_uploader(
        "Upload CSV or Excel to build a dashboard",
        type=["csv", "xlsx"],
        key="make_dashboard_upload",
        help="Upload a business dataset. The dashboard automatically detects numeric, categorical and date fields."
    )

    dashboard_df = df.copy()
    dashboard_source = "Ola Mobility Demo Dataset"

    if uploaded_dash is not None:
        try:
            if uploaded_dash.name.lower().endswith('.csv'):
                dashboard_df = pd.read_csv(uploaded_dash)
            else:
                dashboard_df = pd.read_excel(uploaded_dash)
            dashboard_source = uploaded_dash.name
            st.success(f"Dashboard source loaded: {uploaded_dash.name} • {len(dashboard_df):,} rows • {dashboard_df.shape[1]} columns")
        except Exception as e:
            st.error(f"Could not read the uploaded file: {e}")
            dashboard_df = df.copy()
            dashboard_source = "Ola Mobility Demo Dataset"

    if dashboard_df.empty or dashboard_df.shape[1] == 0:
        st.warning("No usable data found. Upload a CSV or Excel file with at least one column.")
    else:
        # Normalize obvious date columns so time-series visuals work automatically.
        for col in dashboard_df.columns:
            if dashboard_df[col].dtype == 'object':
                sample = dashboard_df[col].dropna().astype(str).head(100)
                if len(sample) >= 5:
                    parsed = pd.to_datetime(sample, errors='coerce')
                    if parsed.notna().mean() >= 0.8:
                        dashboard_df[col] = pd.to_datetime(dashboard_df[col], errors='coerce')

        numeric_cols = dashboard_df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = dashboard_df.select_dtypes(include=['object','category','bool']).columns.tolist()
        date_cols = dashboard_df.select_dtypes(include=['datetime','datetimetz']).columns.tolist()

        # Smart metric selection: prefer business-style columns where possible.
        preferred_metrics = ['fare','revenue','sales','amount','price','profit','distance_km','rating','rides','orders','quantity']
        metric = next((c for key in preferred_metrics for c in numeric_cols if key in str(c).lower()), numeric_cols[0] if numeric_cols else None)

        total_rows = len(dashboard_df)
        missing = int(dashboard_df.isna().sum().sum())
        unique_values = int(dashboard_df.nunique(dropna=True).sum())
        metric_value = float(dashboard_df[metric].sum()) if metric else float(total_rows)

        st.markdown(f'<div class="sql-result-card"><b>📌 {dashboard_source}</b><br><span class="muted">Auto-generated Power BI-style analytics workspace • {total_rows:,} records • {dashboard_df.shape[1]} fields</span></div>',unsafe_allow_html=True)

        k1,k2,k3,k4=st.columns(4)
        with k1:
            st.markdown(f'<div class="card"><div class="klabel">RECORDS</div><div class="kpi">{total_rows:,}</div></div>',unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="card"><div class="klabel">TOTAL {str(metric).upper() if metric else "VALUE"}</div><div class="kpi">{metric_value:,.0f}</div></div>',unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="card"><div class="klabel">MISSING VALUES</div><div class="kpi">{missing:,}</div></div>',unsafe_allow_html=True)
        with k4:
            st.markdown(f'<div class="card"><div class="klabel">FIELDS</div><div class="kpi">{dashboard_df.shape[1]:,}</div></div>',unsafe_allow_html=True)

        # Interactive filters.
        f1,f2=st.columns(2)
        filtered = dashboard_df.copy()
        with f1:
            if categorical_cols:
                filter_col = st.selectbox('Filter dimension', categorical_cols, key='dashboard_filter_col')
                vals = sorted(filtered[filter_col].dropna().astype(str).unique().tolist())
                selected = st.multiselect('Select values', vals, default=vals[:12], key='dashboard_filter_values')
                if selected:
                    filtered = filtered[filtered[filter_col].astype(str).isin(selected)]
            else:
                st.info('No categorical field detected for a dimension filter.')
        with f2:
            if date_cols:
                date_col = st.selectbox('Date field', date_cols, key='dashboard_date_col')
                valid_dates = filtered[date_col].dropna()
                if not valid_dates.empty:
                    min_d,max_d=valid_dates.min().date(),valid_dates.max().date()
                    date_range=st.date_input('Date range',value=(min_d,max_d),key='dashboard_date_range')
                    if isinstance(date_range,tuple) and len(date_range)==2:
                        filtered=filtered[(filtered[date_col].dt.date>=date_range[0])&(filtered[date_col].dt.date<=date_range[1])]
            else:
                st.info('No date field detected. Upload a dataset with a date column for time filtering.')

        st.markdown(f'<div class="section-title">Executive visuals</div><div class="section-sub">Showing {len(filtered):,} filtered records.</div>',unsafe_allow_html=True)

        v1,v2=st.columns(2)
        with v1:
            if date_cols and metric:
                dcol=date_cols[0]
                trend=filtered.dropna(subset=[dcol]).groupby(filtered[dcol].dt.date,as_index=False)[metric].sum()
                trend.columns=['date',metric]
                fig=px.line(trend,x='date',y=metric,markers=True,title=f'{metric} trend',template='plotly_dark')
            elif categorical_cols and metric:
                ccol=categorical_cols[0]
                top=filtered.groupby(ccol,as_index=False)[metric].sum().sort_values(metric,ascending=False).head(12)
                fig=px.bar(top,x=ccol,y=metric,title=f'{metric} by {ccol}',template='plotly_dark')
            elif metric:
                fig=px.histogram(filtered,x=metric,title=f'{metric} distribution',template='plotly_dark')
            else:
                fig=px.histogram(filtered,title='Record distribution',template='plotly_dark')
            fig.update_layout(height=360,margin=dict(l=20,r=20,t=55,b=20))
            st.plotly_chart(fig,use_container_width=True)

        with v2:
            if categorical_cols:
                ccol2=categorical_cols[min(1,len(categorical_cols)-1)]
                counts=filtered[ccol2].astype(str).value_counts().head(12).reset_index()
                counts.columns=[ccol2,'records']
                fig2=px.pie(counts,names=ccol2,values='records',hole=.45,title=f'Record mix by {ccol2}',template='plotly_dark')
            elif numeric_cols:
                fig2=px.scatter(filtered,x=numeric_cols[0],y=numeric_cols[1] if len(numeric_cols)>1 else numeric_cols[0],title='Numeric relationship',template='plotly_dark')
            else:
                fig2=px.bar(x=['Records'],y=[len(filtered)],title='Dataset size',template='plotly_dark')
            fig2.update_layout(height=360,margin=dict(l=20,r=20,t=55,b=20))
            st.plotly_chart(fig2,use_container_width=True)

        # Optional field controls for an analyst who wants to change the visuals.
        with st.expander('⚙️ Customize dashboard fields'):
            c1,c2=st.columns(2)
            with c1:
                if numeric_cols:
                    chosen_metric=st.selectbox('Primary numeric metric',numeric_cols,index=numeric_cols.index(metric) if metric in numeric_cols else 0,key='dashboard_metric')
                    st.metric('Selected metric',chosen_metric)
                else:
                    st.info('No numeric columns detected.')
            with c2:
                if categorical_cols:
                    chosen_dimension=st.selectbox('Primary dimension',categorical_cols,key='dashboard_dimension')
                    st.metric('Selected dimension',chosen_dimension)
                else:
                    st.info('No categorical columns detected.')

        st.markdown('<div class="download-card"><b>⬇️ Export dashboard data</b><br><span class="muted">Download the filtered dataset used by this dashboard.</span></div>',unsafe_allow_html=True)
        st.download_button('Download filtered CSV',filtered.to_csv(index=False).encode(),'dashboard_filtered_data.csv','text/csv',use_container_width=True)

        st.info('💡 This is a Power BI-style interactive dashboard generated inside Streamlit. Creating a real .pbix report in Microsoft Power BI requires Power BI Service/API authentication and a workspace connection.')

# -----------------------------------------------------------------------------
# AI COPILOT
# -----------------------------------------------------------------------------
elif page=='AI Business Copilot':
    st.markdown('<div class="section-title">🧠 AI Business Copilot</div><div class="section-sub">A mobility analyst assistant for business questions, KPI explanations and recommended actions.</div>',unsafe_allow_html=True)
    prompts=['Why is cancellation increasing?','Which city should receive more driver supply?','What vehicle type is strongest for revenue?','Give me 3 executive recommendations from this dataset.','Explain the most important mobility KPI.']
    cols=st.columns(5)
    for c,p in zip(cols,prompts):
        if c.button(p.split('?')[0][:20]+'…',use_container_width=True): st.session_state.messages.append({'role':'user','content':p});st.rerun()
    if not st.session_state.messages:
        st.markdown('<div class="card"><div class="klabel">AI DECISION SUPPORT</div><div style="font-size:18px;font-weight:800;margin-top:6px">Your mobility business copilot is ready.</div><div class="muted" style="margin-top:6px">Ask why a KPI moved, where supply should increase, or what action a business team should take.</div></div>',unsafe_allow_html=True)
    for m in st.session_state.messages:
        with st.chat_message(m['role']): st.markdown(m['content'])
    prompt=st.chat_input('Ask about demand, revenue, cancellations, drivers or customer experience...')
    if prompt: st.session_state.messages.append({'role':'user','content':prompt})
    if st.session_state.messages and st.session_state.messages[-1]['role']=='user':
        p=st.session_state.messages[-1]['content']
        with st.chat_message('assistant'):
            context=f"""You are a professional mobility business analyst for an Ola-inspired portfolio simulation. Never claim access to real Ola production data. Dataset KPIs: {mobility_summary(df)}. Use business language. User question: {p}"""
            try:
                with st.spinner('Analyzing mobility context...'): ans=ask(context)
            except Exception as e: ans=f'Gemini error: {e}'
            st.markdown(ans);st.session_state.messages.append({'role':'assistant','content':ans})
    st.markdown('<div class="mini" style="margin-top:18px">EXECUTIVE AI SUMMARY</div>',unsafe_allow_html=True)
    if st.button('✦ Generate executive insights'):
        s=mobility_summary(df);prompt=f"""Create a concise executive summary for a ride-hailing business using these KPIs: {s}. Mention demand/supply, revenue, cancellations, ETA and customer experience. Give 4 insights and 3 recommended actions. Clearly label it as a portfolio simulation and do not invent real company facts."""
        try: st.markdown('<div class="insight">'+ask(prompt).replace('\n','<br>')+'</div>',unsafe_allow_html=True)
        except Exception as e: st.error(f'Gemini error: {e}')

# -----------------------------------------------------------------------------
# HISTORY
# -----------------------------------------------------------------------------
elif page=='SQL History':
    st.markdown('<div class="section-title">🕘 SQL Query History</div><div class="section-sub">Review and download the latest generated or manually executed queries.</div>',unsafe_allow_html=True)
    if not st.session_state.history: st.info('No SQL queries yet. Run a query from SQL Analyst.')
    for item in st.session_state.history:
        with st.expander(f"{item['time']} • {item['question']}"): st.code(item['sql'],language='sql')
    if st.session_state.history: st.download_button('⬇ Download query history',pd.DataFrame(st.session_state.history).to_csv(index=False).encode(),'sql_query_history.csv','text/csv',use_container_width=True)

st.markdown('---')
st.caption('Ola AI Mobility Intelligence • Portfolio simulation using synthetic mobility data • No live Ola production backend, booking, payment or driver systems are connected.')





# FINAL SIDEBAR TOGGLE FIX - OPEN AND CLOSED
st.markdown('\n<style>\n\n/* =========================================================\n   SIDEBAR TOGGLE ? FINAL FIX\n   CLOSED + OPEN BOTH CLEARLY VISIBLE\n   ========================================================= */\n\n/* Keep header/control layer visible */\nheader[data-testid="stHeader"] {\n    display: block !important;\n    visibility: visible !important;\n    overflow: visible !important;\n    position: relative !important;\n    z-index: 999999 !important;\n}\n\n/* =========================================================\n   OPEN SIDEBAR BUTTON\n   ========================================================= */\n\nbutton[data-testid="stSidebarCollapseButton"] {\n    display: flex !important;\n    visibility: visible !important;\n    opacity: 1 !important;\n\n    position: fixed !important;\n    top: 10px !important;\n    left: 7px !important;\n\n    width: 40px !important;\n    height: 40px !important;\n    min-width: 40px !important;\n    min-height: 40px !important;\n\n    align-items: center !important;\n    justify-content: center !important;\n\n    padding: 0 !important;\n    margin: 0 !important;\n\n    background: #ffffff !important;\n    border: 2px solid #3478b9 !important;\n    border-radius: 11px !important;\n\n    color: #1261a0 !important;\n\n    box-shadow: 0 5px 16px rgba(0,0,0,.30) !important;\n\n    z-index: 2147483647 !important;\n}\n\n/* Open arrow */\nbutton[data-testid="stSidebarCollapseButton"] svg {\n    display: none !important;\n}\n\nbutton[data-testid="stSidebarCollapseButton"]::after {\n    content: "?" !important;\n\n    display: block !important;\n\n    color: #1261a0 !important;\n\n    font-family: Arial, sans-serif !important;\n    font-size: 34px !important;\n    font-weight: 700 !important;\n\n    line-height: 35px !important;\n\n    position: absolute !important;\n    inset: 0 !important;\n\n    text-align: center !important;\n}\n\n\n/* =========================================================\n   CLOSED SIDEBAR\n   IMPORTANT:\n   Target the COLLAPSED CONTROL itself,\n   not only its inner button.\n   ========================================================= */\n\n[data-testid="stSidebarCollapsedControl"] {\n\n    display: flex !important;\n    visibility: visible !important;\n    opacity: 1 !important;\n\n    position: fixed !important;\n    top: 10px !important;\n    left: 7px !important;\n\n    width: 40px !important;\n    height: 40px !important;\n\n    min-width: 40px !important;\n    min-height: 40px !important;\n\n    align-items: center !important;\n    justify-content: center !important;\n\n    padding: 0 !important;\n    margin: 0 !important;\n\n    background: #ffffff !important;\n\n    border: 2px solid #3478b9 !important;\n\n    border-radius: 11px !important;\n\n    box-shadow:\n        0 5px 16px rgba(0,0,0,.30),\n        0 0 0 1px rgba(52,120,185,.10) !important;\n\n    z-index: 2147483647 !important;\n\n    cursor: pointer !important;\n}\n\n/* Closed state inner button */\n[data-testid="stSidebarCollapsedControl"] button {\n\n    display: flex !important;\n    visibility: visible !important;\n    opacity: 1 !important;\n\n    width: 100% !important;\n    height: 100% !important;\n\n    min-width: 100% !important;\n    min-height: 100% !important;\n\n    padding: 0 !important;\n    margin: 0 !important;\n\n    background: transparent !important;\n\n    border: none !important;\n\n    color: #1261a0 !important;\n\n    align-items: center !important;\n    justify-content: center !important;\n\n    box-shadow: none !important;\n}\n\n/* Hide Streamlit\'s faint original icon */\n[data-testid="stSidebarCollapsedControl"] svg {\n    display: none !important;\n}\n\n/* CLOSED = RIGHT ARROW */\n[data-testid="stSidebarCollapsedControl"]::after {\n\n    content: "?" !important;\n\n    display: block !important;\n\n    color: #1261a0 !important;\n\n    font-family: Arial, sans-serif !important;\n    font-size: 34px !important;\n    font-weight: 700 !important;\n\n    line-height: 35px !important;\n\n    position: absolute !important;\n    inset: 0 !important;\n\n    text-align: center !important;\n\n    pointer-events: none !important;\n}\n\n\n/* =========================================================\n   HOVER ? BOTH STATES\n   ========================================================= */\n\nbutton[data-testid="stSidebarCollapseButton"]:hover,\n[data-testid="stSidebarCollapsedControl"]:hover {\n\n    background: #ffffff !important;\n\n    border-color: #1261a0 !important;\n\n    box-shadow:\n        0 7px 20px rgba(0,0,0,.35),\n        0 0 0 3px rgba(52,120,185,.15) !important;\n\n    transform: scale(1.05) !important;\n}\n\n</style>\n', unsafe_allow_html=True)
