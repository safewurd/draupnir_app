# app.py
import os
import sqlite3
from pathlib import Path
import streamlit as st

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Draupnir",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Canonical DB path (per machine) ----------------
# Uses draupnir_core/db_config.py to resolve either a local file
# or a dropbox://... path specific to this machine.
from draupnir_core.db_config import get_db_path

os.makedirs("data", exist_ok=True)
DB_PATH = get_db_path()  # absolute path for THIS machine

# ---------------- Core modules ----------------
from draupnir_core import settings as settings_mod
from draupnir_core import portfolio as portfolio_mod
from draupnir_core import summary as summary_mod
from draupnir_core import forecast as forecast_mod
from draupnir_core import forecast_engine as forecast_engine_mod
from draupnir_core import trades as trades_mod

# Force every module to use the same DB file
settings_mod.DB_PATH = DB_PATH
portfolio_mod.DB_PATH = DB_PATH
summary_mod.DB_PATH = DB_PATH
forecast_mod.DB_PATH = DB_PATH
trades_mod.DB_PATH = DB_PATH
if hasattr(forecast_engine_mod, "DB_PATH_DEFAULT"):
    forecast_engine_mod.DB_PATH_DEFAULT = DB_PATH

# Pull tab functions AFTER setting DB_PATH
from draupnir_core.summary import summary_tab
from draupnir_core.portfolio import portfolio_tab
from draupnir_core.forecast import forecast_tab
from draupnir_core.trades import trades_tab
from draupnir_core.settings import settings_tab

# ---------------- One-time base table init ----------------
try:
    settings_mod.create_settings_tables()
except Exception as ex:
    st.error(f"Failed to initialize base tables: {ex}")

# ---------------- Header ----------------
st.markdown("# 🧠 Draupnir Portfolio Management")
st.markdown("Welcome to your private portfolio management and forecasting system.")
st.caption(f"DB: `{DB_PATH}`")

# ---------------- Tabs ----------------
tabs = st.tabs(["Summary", "Portfolio", "Forecast", "Trade Blotter", "Settings"])

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
