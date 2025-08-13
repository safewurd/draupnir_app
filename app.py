# app.py
import os
import sqlite3
import streamlit as st

# ---- Use a single canonical DB path everywhere ----
os.makedirs("data", exist_ok=True)
DB_PATH = os.path.join("data", "draupnir.db")

# ---- Import modules (then force their DB paths) ----
from draupnir_core import settings as settings_mod
from draupnir_core import portfolio as portfolio_mod
from draupnir_core import summary as summary_mod
from draupnir_core import forecast as forecast_mod
from draupnir_core import forecast_engine as forecast_engine_mod
from draupnir_core import trades as trades_mod

# Make every tab/engine use the same DB file
settings_mod.DB_PATH = DB_PATH
portfolio_mod.DB_PATH = DB_PATH
summary_mod.DB_PATH = DB_PATH
forecast_mod.DB_PATH = DB_PATH
trades_mod.DB_PATH = DB_PATH
forecast_engine_mod.DB_PATH_DEFAULT = DB_PATH

# Pull tab functions
from draupnir_core.summary import summary_tab
from draupnir_core.portfolio import portfolio_tab
from draupnir_core.forecast import forecast_tab
from draupnir_core.trades import trades_tab
from draupnir_core.settings import settings_tab

# Ensure base tables exist on first run
try:
    settings_mod.create_settings_tables()
except Exception as ex:
    st.error(f"Failed to initialize base tables: {ex}")

# Set page config
st.set_page_config(
    page_title="Draupnir",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title & Navigation
st.markdown("# 🧠 Draupnir Portfolio Management")
st.markdown("Welcome to your private wealth analysis and projection system.")
st.caption(f"DB: `{os.path.abspath(DB_PATH)}`")

# (Optional) quick peek at tables; comment out later if you like
try:
    with sqlite3.connect(DB_PATH) as _conn:
        _tbls = [r[0] for r in _conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        ).fetchall()]
    if _tbls:
        st.caption("Tables: " + ", ".join(_tbls))
except Exception:
    pass

# Tabs
tabs = st.tabs(["📈 Summary", "📁 Portfolio", "🔮 Forecast", "📄 Trade Blotter", "⚙️ Settings"])

with tabs[0]:
    summary_tab()

with tabs[1]:
    portfolio_tab()

with tabs[2]:
    forecast_tab()

with tabs[3]:
    trades_tab()

with tabs[4]:
    settings_tab()
