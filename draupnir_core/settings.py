import streamlit as st
import sqlite3
import pandas as pd
import os  # for showing absolute paths and ensuring data dir
from pathlib import Path

# NEW: per-machine DB config helper
from draupnir_core.db_config import get_db_path, set_db_path_local, clear_db_path_local

# ---- Unified DB path (now resolved per machine) ----
os.makedirs("data", exist_ok=True)
DB_PATH = get_db_path()  # <-- absolute path for THIS machine

# ---------- Default Option Lists ----------
BASE_CURRENCY_OPTIONS = ["CAD", "USD"]
MARKET_DATA_PROVIDER_OPTIONS = ["yahoo", "alpha_vantage", "polygon"]

# ---------- Table Management ----------
def create_settings_tables():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Primary key-value store used by this app
    c.execute("""
        CREATE TABLE IF NOT EXISTS global_settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    # Compatibility key-value table (some modules may read from 'settings')
    c.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS institutions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS tax_treatments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        )
    """)

    # Ensure portfolios table exists (columns used by the app)
    c.execute("""
        CREATE TABLE IF NOT EXISTS portfolios (
            portfolio_id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_owner TEXT,
            institution TEXT,
            tax_treatment TEXT,
            account_number TEXT,
            portfolio_name TEXT
        )
    """)

    # Populate defaults if missing
    c.executemany("INSERT OR IGNORE INTO institutions (name) VALUES (?)", [
        ("RBC",), ("RBCDI",), ("Sunlife",), ("RBC Insurance",)
    ])
    c.executemany("INSERT OR IGNORE INTO tax_treatments (name) VALUES (?)", [
        ("Taxable",), ("TFSA",), ("RRSP",), ("RESP",), ("RRIF",)
    ])

    conn.commit()
    conn.close()

# ---------- Settings DAO ----------
def get_settings():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT key, value FROM global_settings", conn)
    conn.close()
    return dict(zip(df["key"], df["value"]))

def set_setting(key, value):
    """
    Write to global_settings and mirror into settings (compat).
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("REPLACE INTO global_settings (key, value) VALUES (?, ?)", (key, value))
    c.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value)
    )
    conn.commit()
    conn.close()

def get_setting_value(key: str) -> str | None:
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute("SELECT value FROM global_settings WHERE key=? LIMIT 1;", (key,)).fetchone()
        if row and row[0] is not None:
            return str(row[0])
        row2 = conn.execute("SELECT value FROM settings WHERE key=? LIMIT 1;", (key,)).fetchone()
        if row2 and row2[0] is not None:
            return str(row2[0])
        return None
    finally:
        conn.close()

def get_dropdown_list(table):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT name FROM {table}", conn)
    conn.close()
    return sorted(df["name"].tolist())

def add_dropdown_option(table, name):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(f"INSERT OR IGNORE INTO {table} (name) VALUES (?)", (name,))
    conn.commit()
    conn.close()

def load_portfolios_df():
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(
            "SELECT portfolio_id, portfolio_name, portfolio_owner, institution, tax_treatment, account_number "
            "FROM portfolios ORDER BY portfolio_name;",
            conn
        )
    finally:
        conn.close()

def portfolio_exists(institution: str, account_number: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT 1 FROM portfolios WHERE institution=? AND account_number=? LIMIT 1;",
            (institution, account_number)
        ).fetchone()
        return row is not None
    finally:
        conn.close()

def insert_portfolio(owner: str, institution: str, tax_treatment: str, account_number: str, set_default: bool) -> int:
    """
    Compose portfolio_name as 'Owner - Institution - Tax Treatment - Account Number'
    using only non-empty parts, then insert. Optionally set as default_portfolio_id.
    Returns new portfolio_id.
    """
    parts = [
        owner.strip(),
        str(institution).strip(),
        str(tax_treatment).strip(),
        account_number.strip()
    ]
    parts = [p for p in parts if p]  # drop empties
    portfolio_name = " - ".join(parts)

    conn = sqlite3.connect(DB_PATH)
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO portfolios (portfolio_owner, institution, tax_treatment, account_number, portfolio_name) "
                "VALUES (?, ?, ?, ?, ?);",
                (owner.strip(), str(institution).strip(), str(tax_treatment).strip(), account_number.strip(), portfolio_name)
            )
            new_id = int(cur.lastrowid)

            if set_default:
                conn.execute(
                    "INSERT INTO global_settings (key, value) VALUES ('default_portfolio_id', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value;",
                    (str(new_id),)
                )
                conn.execute(
                    "INSERT INTO settings (key, value) VALUES ('default_portfolio_id', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value;",
                    (str(new_id),)
                )
        return new_id
    finally:
        conn.close()

# ---------- UI Tab ----------
def settings_tab():
    create_settings_tables()
    settings = get_settings()

    # --- NEW: Database Location (this machine only) ---
    st.markdown("### 📦 Database Location (this machine)")
    st.caption("Set where this computer should read/write the SQLite DB. Not committed to Git.")
    st.text_input("Current DB path", value=str(DB_PATH), disabled=True)

    colA, colB = st.columns(2)
    with colA:
        new_path = st.text_input(
            "Set DB file path (absolute path or dropbox://Apps/draupnir/data/draupnir.db)",
            placeholder=r"N:\Dropbox\Apps\draupnir\data\draupnir.db  or  dropbox://Apps/draupnir/data/draupnir.db"
        )
        copy_first = st.checkbox("If target is new, copy current DB there once", value=True)
        if st.button("💾 Use This Path (This Machine Only)"):
            try:
                resolved = set_db_path_local(new_path.strip(), copy_if_needed=copy_first)
                st.success(f"DB path saved for this machine:\n{resolved}")
                st.info("Restart app or reload page to apply.")
            except Exception as e:
                st.error(f"Failed to set DB path: {e}")

    with colB:
        if st.button("↩️ Switch Back to Local (data/draupnir.db)"):
            try:
                clear_db_path_local()
                st.success("Switched to local DB: data/draupnir.db")
                st.info("Restart app or reload page to apply.")
            except Exception as e:
                st.error(f"Failed to clear local DB override: {e}")

    st.divider()

    # --- Editable Global Settings (trimmed per your request) ---
    st.markdown("### 🌍 Projection Defaults")

    base_currency = st.selectbox(
        "Base Currency", BASE_CURRENCY_OPTIONS,
        index=BASE_CURRENCY_OPTIONS.index(settings.get("base_currency", "CAD"))
    )

    market_data_provider = st.selectbox(
        "Market Data Provider", MARKET_DATA_PROVIDER_OPTIONS,
        index=MARKET_DATA_PROVIDER_OPTIONS.index(settings.get("market_data_provider", "yahoo"))
    )

    # --- Forecast output directory ---
    st.markdown("### 📁 Output")
    forecast_output_dir = st.text_input(
        "Forecast Output Directory",
        value=settings.get("forecast_output_dir", ""),
        placeholder=r"C:\Users\you\Documents\Draupnir\Forecasts (or /Users/you/Draupnir\Forecasts)"
    )
    if forecast_output_dir:
        st.caption(f"Will save forecast Excel files to: `{os.path.abspath(forecast_output_dir)}`")

    if st.button("💾 Save Settings"):
        set_setting("base_currency", base_currency)
        set_setting("market_data_provider", market_data_provider)
        set_setting("forecast_output_dir", forecast_output_dir.strip())
        st.success("✅ Settings saved.")

    st.markdown("---")

    # ---- Create New Portfolio ----
    st.markdown("### 🆕 Create New Portfolio")

    institutions = get_dropdown_list("institutions")
    taxes = get_dropdown_list("tax_treatments")

    col1, col2 = st.columns(2)
    with col1:
        owner = st.text_input("Portfolio Owner", placeholder="e.g., John Doe")
        institution = st.selectbox("Institution", options=institutions) if institutions else st.text_input("Institution")
        tax_treatment = st.selectbox("Tax Treatment", options=taxes) if taxes else st.text_input("Tax Treatment")
    with col2:
        account_number = st.text_input("Account Number", placeholder="e.g., 123-456-789")
        set_default = st.checkbox("Set as default portfolio", value=False)

    if st.button("➕ Create Portfolio", type="primary"):
        # Validation
        errs = []
        if not owner.strip():
            errs.append("Owner is required.")
        if not institution or not str(institution).strip():
            errs.append("Institution is required.")
        if not tax_treatment or not str(tax_treatment).strip():
            errs.append("Tax Treatment is required.")
        if not account_number.strip():
            errs.append("Account Number is required.")

        if not errs and portfolio_exists(str(institution).strip(), account_number.strip()):
            errs.append("A portfolio with this Institution and Account Number already exists.")

        if errs:
            for e in errs:
                st.error(e)
        else:
            try:
                pid = insert_portfolio(
                    owner=owner.strip(),
                    institution=str(institution).strip(),
                    tax_treatment=str(tax_treatment).strip(),
                    account_number=account_number.strip(),
                    set_default=set_default
                )
                st.success(f"✅ Portfolio created (ID {pid}).")
                st.rerun()
            except Exception as ex:
                st.error(f"Failed to create portfolio: {ex}")

    st.markdown("---")
    st.markdown("### 📌 Default Portfolio")

    pf_df = load_portfolios_df()
    if pf_df.empty:
        st.info("No portfolios found. Add portfolios to the database first.")
    else:
        options = pf_df["portfolio_name"].tolist()
        ids = pf_df["portfolio_id"].tolist()

        current_default_id = get_setting_value("default_portfolio_id")
        pre_idx = 0
        if current_default_id and str(current_default_id).isdigit():
            try:
                pre_idx = ids.index(int(current_default_id))
            except ValueError:
                pre_idx = 0

        sel_name = st.selectbox("Default portfolio (shown first in Portfolio tab)", options=options, index=pre_idx)
        sel_id = ids[options.index(sel_name)] if options else None

        if st.button("💾 Save Default Portfolio"):
            if sel_id is not None:
                set_setting("default_portfolio_id", str(sel_id))
                st.success(f"✅ Default portfolio saved: {sel_name} (ID {sel_id})")
            else:
                st.error("Could not resolve selected portfolio.")

    st.markdown("---")
    st.markdown("### 🏦 Manage Institutions")

    institution_list = get_dropdown_list("institutions")
    st.selectbox("Existing Institutions", institution_list)

    new_institution = st.text_input("Add New Institution")
    if st.button("➕ Add Institution") and new_institution:
        add_dropdown_option("institutions", new_institution.strip())
        st.success(f"✅ '{new_institution}' added.")
        st.rerun()

    st.markdown("---")
    st.markdown("### 🧾 Manage Tax Treatments")

    tax_list = get_dropdown_list("tax_treatments")
    st.selectbox("Existing Tax Treatments", tax_list)

    new_tax = st.text_input("Add New Tax Treatment")
    if st.button("➕ Add Tax Treatment") and new_tax:
        add_dropdown_option("tax_treatments", new_tax.strip())
        st.success(f"✅ '{new_tax}' added.")
        st.rerun()
