# app.py
import os
import sqlite3
import streamlit as st

# ---------------- Page config (do this first) ----------------
st.set_page_config(
    page_title="Draupnir",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Canonical DB path ----------------
os.makedirs("data", exist_ok=True)
DB_PATH = os.path.join("data", "draupnir.db")

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
# Some engines keep a default path constant
if hasattr(forecast_engine_mod, "DB_PATH_DEFAULT"):
    forecast_engine_mod.DB_PATH_DEFAULT = DB_PATH

# Pull tab functions (after setting DB_PATH on modules)
from draupnir_core.summary import summary_tab
from draupnir_core.portfolio import portfolio_tab
from draupnir_core.forecast import forecast_tab
from draupnir_core.trades import trades_tab
from draupnir_core.settings import settings_tab

# DB snapshot utilities (export/import CSVs + schema)
from draupnir_core.db_sync import export_snapshot, import_snapshot  # <-- you created this

# ---------------- One-time base table init ----------------
try:
    settings_mod.create_settings_tables()
except Exception as ex:
    st.error(f"Failed to initialize base tables: {ex}")

# ---------------- Header ----------------
st.markdown("# 🧠 Draupnir Portfolio Management")
st.markdown("Welcome to your private wealth analysis and projection system.")
st.caption(f"DB: `{os.path.abspath(DB_PATH)}`")

# Quick peek at tables (safe, optional)
try:
    with sqlite3.connect(DB_PATH) as _conn:
        _tbls = [r[0] for r in _conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        ).fetchall()]
    if _tbls:
        st.caption("Tables: " + ", ".join(_tbls))
except Exception:
    pass

# ---------------- Tabs ----------------
tabs = st.tabs([
    "📈 Summary",
    "📁 Portfolio",
    "🔮 Forecast",
    "📄 Trade Blotter",
    "⚙️ Settings",
    "🔁 Data Sync",
])

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

# ---------------- Data Sync tab ----------------
with tabs[5]:
    st.subheader("Data Sync (DB snapshot export/import)")
    st.caption("Exports schema + tables to `data/seed/` (tracked by git), and can rebuild `data/draupnir.db` from that snapshot.")

    left, right = st.columns(2)

    with left:
        if st.button("📤 Export snapshot to data/seed"):
            try:
                meta = export_snapshot(DB_PATH)  # writes schema.sql + *.csv + manifest.json
                st.success(f"Exported {len(meta.get('tables', []))} tables to `data/seed/`")
                with st.expander("Export details"):
                    st.json(meta)
                st.info("Next step: `git add data/seed/* && git commit && git push` to share the data.")
            except Exception as e:
                st.error(f"Export failed: {e}")

    with right:
        if st.button("📥 Import snapshot from data/seed"):
            try:
                res = import_snapshot(DB_PATH)  # recreates tables and loads CSVs
                st.success(f"Imported snapshot into `{os.path.abspath(DB_PATH)}`")
                with st.expander("Import details"):
                    st.json(res)
                st.warning("If the views still look empty, click 'Rerun' or refresh the page.")
            except Exception as e:
                st.error(f"Import failed: {e}")
