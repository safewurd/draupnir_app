# draupnir_core/trades.py
import sqlite3
from functools import lru_cache
from datetime import datetime
from typing import Optional, List
from io import BytesIO
import os

import pandas as pd
import streamlit as st
import yfinance as yf

# ---- Unified DB path ----
os.makedirs("data", exist_ok=True)
DB_PATH = os.path.join("data", "draupnir.db")

# -----------------------------
# Quiet Yahoo resolver
# -----------------------------

CAD_SUFFIX_SECONDARY: List[str] = [".V", ".NE", ".CN"]

def _has_price_data(sym: str) -> bool:
    if not sym:
        return False
    try:
        hist = yf.Ticker(sym).history(period="1d", auto_adjust=False, actions=False, raise_errors=False)
        return (not hist.empty) and ("Close" in hist.columns) and pd.notna(hist["Close"]).any()
    except Exception:
        return False

@lru_cache(maxsize=2048)
def resolve_yahoo_symbol(ticker: str, currency: str) -> Optional[str]:
    if not ticker:
        return None
    t = ticker.strip().upper()
    cur = (currency or "").upper()

    if cur == "CAD":
        primary = f"{t}.TO"
        if _has_price_data(primary):
            return primary
        for suff in CAD_SUFFIX_SECONDARY:
            sym = f"{t}{suff}"
            if _has_price_data(sym):
                return sym
        return primary

    if _has_price_data(t):
        return t
    return t

# -----------------------------
# DB helpers
# -----------------------------

CREATE_TRADES_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio_id INTEGER,
    portfolio_name TEXT,
    account_number TEXT,
    ticker TEXT,
    currency TEXT,
    action TEXT,
    quantity REAL,
    price REAL,
    commission REAL,
    yahoo_symbol TEXT,
    trade_date TEXT,
    created_at TEXT
);
"""

def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def init_db():
    conn = _connect(DB_PATH)
    conn.execute(CREATE_TRADES_SQL)
    conn.commit()
    conn.close()

def fetch_portfolios() -> pd.DataFrame:
    conn = _connect(DB_PATH)
    try:
        df = pd.read_sql(
            "SELECT portfolio_id, portfolio_name, account_number, portfolio_owner, institution, tax_treatment "
            "FROM portfolios ORDER BY portfolio_owner, institution, account_number;",
            conn
        )
        return df
    finally:
        conn.close()

def get_portfolio_by_name(name: str) -> Optional[dict]:
    if not name:
        return None
    conn = _connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT portfolio_id, portfolio_name, account_number, portfolio_owner, institution, tax_treatment "
            "FROM portfolios WHERE portfolio_name = ? LIMIT 1;",
            (name,)
        ).fetchone()
        if not row:
            return None
        cols = ["portfolio_id","portfolio_name","account_number","portfolio_owner","institution","tax_treatment"]
        return dict(zip(cols, row))
    finally:
        conn.close()

def insert_trade(row: dict) -> int:
    conn = _connect(DB_PATH)
    try:
        cols = ["portfolio_id","portfolio_name","account_number","ticker","currency","action",
                "quantity","price","commission","yahoo_symbol","trade_date","created_at"]
        vals = [row.get(c) for c in cols]
        qs = ",".join(["?"] * len(cols))
        conn.execute(CREATE_TRADES_SQL)
        cur = conn.execute(f"INSERT INTO trades ({','.join(cols)}) VALUES ({qs})", vals)
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()

def delete_trade(trade_id: int) -> bool:
    conn = _connect(DB_PATH)
    try:
        conn.execute("DELETE FROM trades WHERE trade_id = ?", (trade_id,))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()

def _load_recent(limit: int = 200) -> pd.DataFrame:
    conn = _connect(DB_PATH)
    try:
        return pd.read_sql(
            "SELECT trade_id, portfolio_name, account_number, ticker, currency, action, "
            "quantity, price, commission, yahoo_symbol, trade_date, created_at "
            "FROM trades ORDER BY datetime(created_at) DESC LIMIT ?;",
            conn, params=(limit,)
        )
    finally:
        conn.close()

def _load_all_trades() -> pd.DataFrame:
    conn = _connect(DB_PATH)
    try:
        return pd.read_sql(
            "SELECT trade_id, portfolio_id, portfolio_name, account_number, ticker, currency, action, "
            "quantity, price, commission, yahoo_symbol, trade_date, created_at "
            "FROM trades ORDER BY datetime(created_at) ASC;",
            conn
        )
    finally:
        conn.close()

# -----------------------------
# UI helpers
# -----------------------------

def _validate_inputs(portfolio_name, ticker, currency, action, quantity, price) -> list[str]:
    errs = []
    if not portfolio_name or not str(portfolio_name).strip():
        errs.append("Portfolio is required.")
    if not ticker or not str(ticker).strip():
        errs.append("Ticker is required.")
    if currency not in ("CAD", "USD"):
        errs.append("Currency must be CAD or USD.")
    if action not in ("BUY", "SELL"):
        errs.append("Action must be BUY or SELL.")
    try:
        q = float(quantity)
        if q <= 0:
            errs.append("Quantity must be greater than 0.")
    except Exception:
        errs.append("Quantity is invalid.")
    try:
        p = float(price)
        if p <= 0:
            errs.append("Price must be greater than 0.")
    except Exception:
        errs.append("Price is invalid.")
    return errs

def _normalize_str(s: Optional[str]) -> str:
    return "" if s is None else str(s).strip().upper()

def _dedupe_key(portfolio_name, ticker, currency, action, quantity, price, trade_date) -> str:
    return f"{_normalize_str(portfolio_name)}|{_normalize_str(ticker)}|{_normalize_str(currency)}|" \
           f"{_normalize_str(action)}|{float(quantity):.6f}|{float(price):.6f}|{str(trade_date)}"

# -----------------------------
# Trade Tab
# -----------------------------

def trades_tab():
    st.subheader("📄 Trade Blotter")

    init_db()

    pf = fetch_portfolios()
    if pf.empty:
        st.warning("No portfolios found. Add rows to the 'portfolios' table first.")
        return

    portfolio_names = sorted(pf["portfolio_name"].astype(str).str.strip().unique().tolist())

    # ---- Add Trade ----
    with st.form("add_trade_form", clear_on_submit=False):
        colA, colB = st.columns(2)
        with colA:
            selected_portfolio_name = st.selectbox("Portfolio", options=portfolio_names, index=0)
            action = st.selectbox("Action", options=["BUY", "SELL"], index=0)
            ticker = st.text_input("Ticker", placeholder="e.g., RY").upper()
            currency = st.selectbox("Currency", options=["CAD", "USD"], index=0)
        with colB:
            quantity = st.number_input("Quantity", min_value=0.0, step=1.0, format="%.4f")
            price = st.number_input("Price", min_value=0.0, step=0.01, format="%.6f")
            commission = st.number_input("Commission", min_value=0.0, step=0.01, format="%.2f")
            trade_date = st.date_input("Trade Date")

        submitted = st.form_submit_button("Add Trade", use_container_width=True)

    if submitted:
        errors = _validate_inputs(selected_portfolio_name, ticker, currency, action, quantity, price)
        if errors:
            for e in errors:
                st.error(e)
            return

        dedupe_key = _dedupe_key(selected_portfolio_name, ticker, currency, action, quantity, price, trade_date)
        if st.session_state.get("last_trade_submit_key") == dedupe_key:
            st.info("This trade was just submitted. Ignoring duplicate submit.")
            return

        p = get_portfolio_by_name(selected_portfolio_name)
        if not p:
            st.error("Selected portfolio not found.")
            return

        yahoo_symbol = resolve_yahoo_symbol(ticker, currency)

        row = {
            "portfolio_id": int(p["portfolio_id"]),
            "portfolio_name": p["portfolio_name"],
            "account_number": p["account_number"],
            "ticker": ticker.strip().upper(),
            "currency": currency.strip().upper(),
            "action": action.strip().upper(),
            "quantity": float(quantity),
            "price": float(price),
            "commission": float(commission or 0.0),
            "yahoo_symbol": yahoo_symbol or "",
            "trade_date": str(trade_date),
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }

        try:
            trade_id = insert_trade(row)
        except Exception as ex:
            st.error(f"Failed to add trade: {ex}")
            return

        st.cache_data.clear()
        st.session_state["portfolio_refresh_token"] = datetime.utcnow().isoformat(timespec="seconds")
        st.session_state["last_trade_submit_key"] = dedupe_key

        st.success(f"Trade added (trade_id={trade_id}) to {row['portfolio_name']}")
        st.dataframe(pd.DataFrame([row]), use_container_width=True, hide_index=True)

    # ---- Recent trades ----
    st.markdown("#### Recent Trades")
    try:
        recent = _load_recent(25)
        if not recent.empty:
            for c in ["quantity","price","commission"]:
                if c in recent.columns:
                    recent[c] = pd.to_numeric(recent[c], errors="coerce")
            st.dataframe(recent, use_container_width=True, hide_index=True)
        else:
            st.info("No trades recorded yet.")
    except Exception as ex:
        st.warning(f"Could not load recent trades: {ex}")

    # ---- Delete Trade Section ----
    st.markdown("### 🗑️ Delete a Trade")
    try:
        all_trades = _load_all_trades()
    except Exception as ex:
        st.error(f"Error loading trades for deletion: {ex}")
        return

    if all_trades.empty:
        st.info("No trades to delete.")
        return

    delete_options = all_trades.apply(
        lambda r: f"ID {r.trade_id} | {r.portfolio_name} | {r.action} {r.quantity} {r.ticker} @ {r.price} ({r.trade_date})",
        axis=1
    ).tolist()
    trade_map = dict(zip(delete_options, all_trades["trade_id"].tolist()))

    selected_delete = st.selectbox("Select a trade to delete", options=delete_options)
    confirm_delete = st.checkbox("Confirm delete")

    if st.button("Delete Trade", type="primary", disabled=not confirm_delete):
        tid = trade_map[selected_delete]
        if delete_trade(tid):
            st.cache_data.clear()
            st.session_state["portfolio_refresh_token"] = datetime.utcnow().isoformat(timespec="seconds")
            st.success(f"Trade ID {tid} deleted successfully.")
            st.rerun()
        else:
            st.error(f"Failed to delete trade ID {tid}.")

    # ---- Export to Excel (bottom, with unique key) ----
    st.markdown("### ⬇️ Export Trade Blotter")
    colx, coly = st.columns([1,3])
    with colx:
        if st.button("Prepare Excel File", key="btn_prepare_trades_excel", type="secondary"):
            st.session_state["export_ready"] = True

    if st.session_state.get("export_ready"):
        df_all = _load_all_trades()
        for c in ["quantity","price","commission"]:
            if c in df_all.columns:
                df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            df_all.to_excel(writer, index=False, sheet_name="Trades")
            ws = writer.sheets["Trades"]
            for col_cells in ws.columns:
                length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col_cells)
                ws.column_dimensions[col_cells[0].column_letter].width = min(max(length + 2, 12), 50)
        bio.seek(0)

        fname = f"trade_blotter_{datetime.utcnow().strftime('%Y%m%d_%H%M%SZ')}.xlsx"
        st.download_button(
            label="Download Excel",
            data=bio.getvalue(),
            file_name=fname,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=False
        )
