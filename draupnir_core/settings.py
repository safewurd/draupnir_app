import streamlit as st
import sqlite3
import pandas as pd
import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import math

import yfinance as yf  # for yields/prices

# per-machine DB config helper
from draupnir_core.db_config import get_db_path, set_db_path_local, clear_db_path_local

# ---- Unified DB path (now resolved per machine) ----
os.makedirs("data", exist_ok=True)
DB_PATH = get_db_path()  # absolute path for THIS machine

# ---------- Default Option Lists ----------
BASE_CURRENCY_OPTIONS = ["CAD", "USD"]
MARKET_DATA_PROVIDER_OPTIONS = ["yahoo", "alpha_vantage", "polygon"]

# ---------- Column utilities / migrations ----------
def _columns_in(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
        return {r[1] for r in rows}
    except Exception:
        return set()

def _ensure_portfolios_columns(conn: sqlite3.Connection) -> None:
    """
    Add new portfolio-level yield and reinvest columns if missing.
    Stored as:
      - interest_yield, div_eligible_yield, div_noneligible_yield : REAL (e.g., 2.0 means 2%/yr)
      - reinvest_interest, reinvest_dividends : INTEGER (1=true, 0=false), defaults 1
    """
    cols = _columns_in(conn, "portfolios")
    add_sql = []
    if "interest_yield" not in cols:
        add_sql.append("ALTER TABLE portfolios ADD COLUMN interest_yield REAL;")
    if "div_eligible_yield" not in cols:
        add_sql.append("ALTER TABLE portfolios ADD COLUMN div_eligible_yield REAL;")
    if "div_noneligible_yield" not in cols:
        add_sql.append("ALTER TABLE portfolios ADD COLUMN div_noneligible_yield REAL;")
    if "reinvest_interest" not in cols:
        add_sql.append("ALTER TABLE portfolios ADD COLUMN reinvest_interest INTEGER DEFAULT 1;")
    if "reinvest_dividends" not in cols:
        add_sql.append("ALTER TABLE portfolios ADD COLUMN reinvest_dividends INTEGER DEFAULT 1;")
    for sql in add_sql:
        conn.execute(sql)
    if add_sql:
        conn.commit()

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

    # Auto-migrate portfolios to include yields & reinvest flags
    _ensure_portfolios_columns(conn)

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

def load_portfolios_df(include_dist_cols: bool = True):
    cols = [
        "portfolio_id","portfolio_name","portfolio_owner",
        "institution","tax_treatment","account_number"
    ]
    if include_dist_cols:
        cols += [
            "interest_yield","div_eligible_yield","div_noneligible_yield",
            "reinvest_interest","reinvest_dividends"
        ]
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(
            f"SELECT {', '.join(cols)} FROM portfolios ORDER BY portfolio_name;",
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

def insert_portfolio(
    owner: str,
    institution: str,
    tax_treatment: str,
    account_number: str,
    set_default: bool,
    interest_yield: float | None = None,
    div_eligible_yield: float | None = None,
    div_noneligible_yield: float | None = None,
    reinvest_interest: bool = True,
    reinvest_dividends: bool = True,
) -> int:
    """
    Compose portfolio_name as 'Owner - Institution - Tax Treatment - Account Number'
    using only non-empty parts, then insert. Optionally set as default_portfolio_id.
    Annualized yields are stored as whole percentages (e.g., 2.0 = 2%/yr).
    Reinvent flags stored as 1/0.
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
                "INSERT INTO portfolios (portfolio_owner, institution, tax_treatment, account_number, portfolio_name, "
                "                         interest_yield, div_eligible_yield, div_noneligible_yield, "
                "                         reinvest_interest, reinvest_dividends) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                (
                    owner.strip(),
                    str(institution).strip(),
                    str(tax_treatment).strip(),
                    account_number.strip(),
                    portfolio_name,
                    (None if interest_yield is None else float(interest_yield)),
                    (None if div_eligible_yield is None else float(div_eligible_yield)),
                    (None if div_noneligible_yield is None else float(div_noneligible_yield)),
                    (1 if reinvest_interest else 0),
                    (1 if reinvest_dividends else 0),
                )
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

def update_portfolio_distribution_settings(df: pd.DataFrame) -> None:
    """
    Persist edited yields and reinvest flags for each row in the provided DataFrame.
    Expects columns: portfolio_id, interest_yield, div_eligible_yield, div_noneligible_yield,
                     reinvest_interest (bool), reinvest_dividends (bool)
    """
    if df.empty:
        return
    conn = sqlite3.connect(DB_PATH)
    try:
        with conn:
            rows = []
            for r in df.itertuples(index=False):
                pid = int(getattr(r, "portfolio_id"))
                iy  = getattr(r, "interest_yield", None)
                de  = getattr(r, "div_eligible_yield", None)
                dne = getattr(r, "div_noneligible_yield", None)
                ri  = getattr(r, "reinvest_interest", False)
                rd  = getattr(r, "reinvest_dividends", False)
                rows.append((
                    float(iy) if iy is not None else None,
                    float(de) if de is not None else None,
                    float(dne) if dne is not None else None,
                    1 if bool(ri) else 0,
                    1 if bool(rd) else 0,
                    pid
                ))
            conn.executemany("""
                UPDATE portfolios
                SET interest_yield = ?,
                    div_eligible_yield = ?,
                    div_noneligible_yield = ?,
                    reinvest_interest = ?,
                    reinvest_dividends = ?
                WHERE portfolio_id = ?;
            """, rows)
    finally:
        conn.close()

# ---------- Yahoo helpers for weighted-average yields ----------
CAD_SUFFIXES = (".TO", ".V", ".NE", ".CN")

def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def _load_trades() -> pd.DataFrame:
    conn = _connect(DB_PATH)
    try:
        try:
            df = pd.read_sql("""
                SELECT trade_id, portfolio_id, portfolio_name, ticker, currency, action, quantity, price,
                       commission, yahoo_symbol, trade_date
                FROM trades
            """, conn)
        except Exception:
            return pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        return df

    # Normalize
    for c in ["ticker","currency","action","yahoo_symbol"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    for c in ["quantity","price","commission"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Signed qty
    df["action_up"] = df["action"].str.upper()
    df["signed_qty"] = df.apply(lambda r: r["quantity"] * (1.0 if r["action_up"] == "BUY" else (-1.0 if r["action_up"] == "SELL" else 0.0)), axis=1)
    return df

@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_last_close(sym: str) -> Optional[float]:
    try:
        if not sym:
            return None
        hist = yf.Ticker(sym).history(period="1d", auto_adjust=False, actions=False, raise_errors=False)
        if hist.empty or "Close" not in hist.columns:
            return None
        ser = hist["Close"].dropna()
        return float(ser.iloc[-1]) if not ser.empty else None
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=86400)
def _fetch_info(sym: str) -> Dict:
    try:
        if not sym:
            return {}
        t = yf.Ticker(sym)
        return dict(t.info or {})
    except Exception:
        return {}

def _classify_symbol_for_yields(info: Dict, symbol: str) -> Tuple[bool, bool]:
    """
    Returns (is_fixed_income_like, is_canadian_dividend)
    - fixed income heuristic:
        - info['yield'] present OR
        - 'Bond'/'Treasury'/'Fixed Income'/'Aggregate' in shortName/longName/category
    - canadian dividend: symbol ends with Canadian suffix
    """
    name_fields = []
    for key in ("shortName","longName","category"):
        v = info.get(key)
        if isinstance(v, str):
            name_fields.append(v.lower())
    name_text = " ".join(name_fields)

    fixed_income_like = False
    if info.get("yield") is not None:
        fixed_income_like = True
    else:
        for kw in ("bond", "treasury", "fixed income", "aggregate"):
            if kw in name_text:
                fixed_income_like = True
                break

    is_cad_listed = False
    s = (symbol or "").upper()
    for suf in CAD_SUFFIXES:
        if s.endswith(suf.upper()):
            is_cad_listed = True
            break

    return fixed_income_like, is_cad_listed

# ---- NEW: normalize Yahoo values to DECIMAL (0.027 for 2.7%) ----
def _normalize_to_decimal(val):
    """
    Normalize a yield value to DECIMAL form (0.027 for 2.7%).
    Yahoo fields:
      - info['dividendYield'] is usually decimal already (0.027)
      - info['yield'] (on funds) is often PERCENT (2.7), but not always
    Heuristic:
      - Convert to float; if > 1.0, assume percent and divide by 100
      - If still > 0.5 (50%), divide by 100 again (defensive)
    """
    if val is None:
        return None
    try:
        y = float(val)
    except Exception:
        return None
    if y > 1.0:
        y = y / 100.0
    if y > 0.5:  # >50% is almost certainly percent mislabeled as decimal
        y = y / 100.0
    if y < 0:
        return None
    return y

def _div_yield_from_info(info: Dict) -> Optional[float]:
    # Usually DECIMAL already (e.g., 0.041 = 4.1%)
    return _normalize_to_decimal(info.get("dividendYield", None))

def _fund_yield_from_info(info: Dict) -> Optional[float]:
    # Often PERCENT on funds (e.g., 2.7 = 2.7%), normalize it
    return _normalize_to_decimal(info.get("yield", None))

def _compute_weighted_yields_for_all_portfolios() -> Tuple[pd.DataFrame, Dict[int, List[str]]]:
    """
    Returns:
      df with columns [portfolio_id, interest_yield, div_eligible_yield, div_noneligible_yield] (percent values)
      missing: dict {portfolio_id: [symbols_without_yield_info]}
    """
    trades = _load_trades()
    if trades.empty:
        return pd.DataFrame(), {}

    # Aggregate positions by portfolio_id + symbol
    sym = trades["yahoo_symbol"].fillna("").str.strip()
    trades = trades.assign(symbol=sym.where(sym != "", trades["ticker"].fillna("").str.strip()))
    grp = trades.groupby(["portfolio_id","symbol"], dropna=False)["signed_qty"].sum().reset_index()
    grp = grp[grp["signed_qty"].ne(0.0)]

    if grp.empty:
        return pd.DataFrame(), {}

    # Fetch prices and infos
    grp["price"] = grp["symbol"].apply(_fetch_last_close)
    grp["mkt_value"] = grp["signed_qty"] * grp["price"].fillna(0.0)

    # Get yahoo info fields we care about
    infos = {symb: _fetch_info(symb) for symb in grp["symbol"].unique().tolist()}
    grp["info"] = grp["symbol"].map(infos)

    # Pull raw yields (normalized to DECIMAL)
    grp["div_yield_dec"] = grp["info"].apply(_div_yield_from_info)
    grp["fund_yield_dec"] = grp["info"].apply(_fund_yield_from_info)

    # Classify per security
    grp[["is_fixed_income_like","is_cad_listed"]] = grp.apply(
        lambda r: pd.Series(_classify_symbol_for_yields(r["info"], r["symbol"])), axis=1
    )

    # Initialize buckets (DECIMAL, not percent yet)
    grp["interest_component"] = 0.0
    grp["eligible_div_component"] = 0.0
    grp["noneligible_div_component"] = 0.0

    # Heuristics:
    # - If fixed-income-like:
    #     use fund_yield_dec when available; else if div_yield_dec exists, treat it as interest (bond ETFs sometimes show dividends)
    # - Else (equity-like):
    #     use div_yield_dec as dividend; split eligible vs non-eligible by exchange suffix (CAD-listed → eligible)
    def _assign_components(row):
        mv = float(row["mkt_value"] or 0.0)
        if mv <= 0 or (math.isfinite(mv) is False):
            return 0.0, 0.0, 0.0

        divy = row["div_yield_dec"]
        fundy = row["fund_yield_dec"]
        fixed_like = bool(row["is_fixed_income_like"])
        is_cad = bool(row["is_cad_listed"])

        interest = 0.0
        elig = 0.0
        nonelig = 0.0

        if fixed_like:
            y = fundy if (fundy is not None) else (divy if (divy is not None) else None)
            if y is not None:
                interest = y * mv
        else:
            if divy is not None:
                if is_cad:
                    elig = divy * mv
                else:
                    nonelig = divy * mv

        return interest, elig, nonelig

    comps = grp.apply(_assign_components, axis=1, result_type="expand")
    grp["interest_amt"] = comps[0]
    grp["eligible_div_amt"] = comps[1]
    grp["noneligible_div_amt"] = comps[2]

    # Sum by portfolio
    agg = grp.groupby("portfolio_id", as_index=False).agg(
        mkt_value=("mkt_value","sum"),
        interest_amt=("interest_amt","sum"),
        eligible_div_amt=("eligible_div_amt","sum"),
        noneligible_div_amt=("noneligible_div_amt","sum"),
    )

    # Weighted average yields (convert to PERCENT values)
    def _pct(x): return (x * 100.0)
    agg["interest_yield"] = agg.apply(lambda r: _pct(r["interest_amt"] / r["mkt_value"]) if r["mkt_value"] else 0.0, axis=1)
    agg["div_eligible_yield"] = agg.apply(lambda r: _pct(r["eligible_div_amt"] / r["mkt_value"]) if r["mkt_value"] else 0.0, axis=1)
    agg["div_noneligible_yield"] = agg.apply(lambda r: _pct(r["noneligible_div_amt"] / r["mkt_value"]) if r["mkt_value"] else 0.0, axis=1)

    # Find missing symbols (no price or no yield signal at all)
    missing: Dict[int, List[str]] = {}
    for pid, g in grp.groupby("portfolio_id"):
        miss_syms: List[str] = []
        for _, r in g.iterrows():
            # consider missing if neither fund_yield nor div_yield available
            if r["price"] is None:
                miss_syms.append(str(r["symbol"]))
                continue
            if (r["div_yield_dec"] is None) and (r["fund_yield_dec"] is None):
                miss_syms.append(str(r["symbol"]))
        if miss_syms:
            missing[int(pid)] = sorted(list(set([s for s in miss_syms if s])))

    out = agg[["portfolio_id","interest_yield","div_eligible_yield","div_noneligible_yield"]].copy()
    return out, missing

# ---------- UI Tab ----------
def settings_tab():
    create_settings_tables()
    settings = get_settings()

    # --- Database Location (this machine only) ---
    st.markdown("### 📦 Database Location (this machine)")
    st.caption("Set where this computer should read/write the SQLite DB. Not committed to Git.")
    st.text_input("Current DB path", value=str(DB_PATH), disabled=True)

    colA, colB = st.columns(2)
    with colA:
        new_path = st.text_input(
            "Set DB file path (absolute path or dropbox://Apps/draupnir/data/draupnir.db)",
            placeholder=r"N:\Dropbox\Apps\draupnir\data\draupnir.db  or  dropbox://Apps\draupnir\data\draupnir.db"
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

    # --- Projection Defaults ---
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
        placeholder=r"C:\\Users\\you\\Documents\\Draupnir\\Forecasts (or /Users/you/Draupnir/Forecasts)"
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

    st.markdown("#### Distribution & Reinvestment Details (optional)")
    cA, cB, cC = st.columns(3)
    with cA:
        interest_yield = st.number_input("Interest Yield (%/yr)", min_value=0.0, step=0.1, value=0.0, format="%.1f")
    with cB:
        div_eligible_yield = st.number_input("Eligible Dividend Yield (%/yr)", min_value=0.0, step=0.1, value=0.0, format="%.1f")
    with cC:
        div_noneligible_yield = st.number_input("Non-eligible Dividend Yield (%/yr)", min_value=0.0, step=0.1, value=0.0, format="%.1f")

    cD, cE = st.columns(2)
    with cD:
        reinvest_interest = st.checkbox("Reinvest interest", value=True)
    with cE:
        reinvest_dividends = st.checkbox("Reinvest dividends", value=True)

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
                    set_default=set_default,
                    interest_yield=float(interest_yield) if interest_yield is not None else None,
                    div_eligible_yield=float(div_eligible_yield) if div_eligible_yield is not None else None,
                    div_noneligible_yield=float(div_noneligible_yield) if div_noneligible_yield is not None else None,
                    reinvest_interest=bool(reinvest_interest),
                    reinvest_dividends=bool(reinvest_dividends),
                )
                st.success(f"✅ Portfolio created (ID {pid}).")
                st.rerun()
            except Exception as ex:
                st.error(f"Failed to create portfolio: {ex}")

    st.markdown("---")

    # ---- Edit existing portfolio distribution settings (INLINE EDITOR) ----
    st.markdown("### ✏️ Edit Portfolio Distribution & Reinvestment")
    pf_df = load_portfolios_df(include_dist_cols=True)
    if pf_df.empty:
        st.info("No portfolios found. Add one above first.")
    else:
        # Normalize types for editing
        for c in ["interest_yield","div_eligible_yield","div_noneligible_yield"]:
            if c in pf_df.columns:
                pf_df[c] = pd.to_numeric(pf_df[c], errors="coerce")
        for c in ["reinvest_interest","reinvest_dividends"]:
            if c in pf_df.columns:
                pf_df[c] = pf_df[c].map(lambda x: bool(int(x)) if pd.notna(x) else True)

        # Fix optional typo if schema has 'tax_treatment'
        if "tax_treatment" in pf_df.columns and "tax_treatement" not in pf_df.columns:
            pf_df["tax_treatement"] = pf_df["tax_treatment"]

        show_cols = [
            "portfolio_id","portfolio_name","institution","tax_treatement","account_number",
            "interest_yield","div_eligible_yield","div_noneligible_yield",
            "reinvest_interest","reinvest_dividends"
        ]
        show_cols = [c for c in show_cols if c in pf_df.columns]

        # Use a separate session_state buffer for the editor's data
        buffer_key = "portfolio_dist_editor_df"
        editor_key = "portfolio_dist_editor"

        if buffer_key not in st.session_state:
            st.session_state[buffer_key] = pf_df[show_cols].copy()

        # Render editor from the buffer (do not mutate editor_key in session_state directly)
        edited = st.data_editor(
            st.session_state[buffer_key],
            use_container_width=True,
            num_rows="fixed",
            key=editor_key
        )

        col_btn1, col_btn2 = st.columns([2, 1])
        with col_btn1:
            if st.button("📡 Fetch Yahoo Yields → Populate Table", type="secondary", key="btn_fetch_yahoo"):
                with st.spinner("Fetching prices & yields from Yahoo and computing weighted averages…"):
                    auto_df, missing = _compute_weighted_yields_for_all_portfolios()

                if auto_df.empty:
                    st.warning("No holdings found or unable to compute yields. You can edit values manually below.")
                else:
                    # Merge computed yields into the BUFFER (not the widget key)
                    merged = st.session_state[buffer_key].merge(
                        auto_df,
                        on="portfolio_id",
                        how="left",
                        suffixes=("", "_auto")
                    )
                    for k in ["interest_yield","div_eligible_yield","div_noneligible_yield"]:
                        k_auto = f"{k}_auto"
                        if k_auto in merged.columns:
                            merged[k] = merged[k_auto].where(merged[k_auto].notna(), merged[k])
                            merged.drop(columns=[k_auto], inplace=True, errors="ignore")

                    st.session_state[buffer_key] = merged  # update the buffer
                    # Show prompts for manual entry if some symbols had no yield info
                    if missing:
                        msgs = []
                        for pid, syms in missing.items():
                            if not syms:
                                continue
                            name = merged.loc[merged["portfolio_id"] == pid, "portfolio_name"].head(1).astype(str).tolist()
                            name_str = name[0] if name else f"ID {pid}"
                            msgs.append(f"- **{name_str}**: {', '.join(syms)}")
                        if msgs:
                            st.warning(
                                "Some holdings lacked yield data from Yahoo. "
                                "Please review and adjust the table if needed:\n\n" + "\n".join(msgs)
                            )
                        else:
                            st.info("Fetched yields applied. Review and adjust if needed, then click **Save Changes**.")
                    st.rerun()  # re-render editor with new buffer data

        with col_btn2:
            if st.button("💾 Save Changes", type="primary", key="btn_save_portfolio_dist"):
                try:
                    # Persist modifications returned by the editor
                    update_portfolio_distribution_settings(edited if isinstance(edited, pd.DataFrame) else st.session_state[buffer_key])
                    # Keep buffer in sync with latest saved edits
                    st.session_state[buffer_key] = (edited if isinstance(edited, pd.DataFrame) else st.session_state[buffer_key])
                    st.success("✅ Portfolio settings updated.")
                except Exception as ex:
                    st.error(f"Failed to update portfolios: {ex}")

    st.markdown("---")

    # ---- Default Portfolio ----
    st.markdown("### 📌 Default Portfolio")

    pf_df2 = load_portfolios_df(include_dist_cols=False)
    if pf_df2.empty:
        st.info("No portfolios found. Add portfolios to the database first.")
    else:
        options = pf_df2["portfolio_name"].tolist()
        ids = pf_df2["portfolio_id"].tolist()

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

    # ---- Manage Institutions ----
    st.markdown("### 🏦 Manage Institutions")

    institution_list = get_dropdown_list("institutions")
    st.selectbox("Existing Institutions", institution_list)

    new_institution = st.text_input("Add New Institution")
    if st.button("➕ Add New Institution") and new_institution:
        add_dropdown_option("institutions", new_institution.strip())
        st.success(f"✅ '{new_institution}' added.")
        st.rerun()

    st.markdown("---")

    # ---- Manage Tax Treatments ----
    st.markdown("### 🧾 Manage Tax Treatments")

    tax_list = get_dropdown_list("tax_treatments")
    st.selectbox("Existing Tax Treatments", tax_list)

    new_tax = st.text_input("Add New Tax Treatment")
    if st.button("➕ Add New Tax Treatment") and new_tax:
        add_dropdown_option("tax_treatments", new_tax.strip())
        st.success(f"✅ '{new_tax}' added.")
        st.rerun()
