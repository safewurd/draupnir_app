import sqlite3
import math
from functools import lru_cache
from typing import Dict, Optional, List, Iterable
from io import BytesIO
from datetime import datetime
import os

import pandas as pd
import yfinance as yf
import streamlit as st

# ---- Unified DB path ----
os.makedirs("data", exist_ok=True)
DB_PATH = os.path.join("data", "draupnir.db")

# =========================
# Quiet Yahoo helpers
# =========================

CAD_SUFFIX_CANDIDATES: List[str] = [".TO", ".V", ".NE", ".CN"]  # TSX, TSXV, NEO, CSE

def _has_price_data(sym: str) -> bool:
    if not sym:
        return False
    try:
        hist = yf.Ticker(sym).history(period="1d", auto_adjust=False, actions=False, raise_errors=False)
        return (not hist.empty) and ("Close" in hist.columns) and pd.notna(hist["Close"]).any()
    except Exception:
        return False

@lru_cache(maxsize=4096)
def resolve_yahoo_symbol(ticker: str, currency: str) -> Optional[str]:
    if not ticker:
        return None
    t = ticker.strip().upper()
    cur = (currency or "").upper()

    if cur == "CAD":
        primary = f"{t}.TO"
        if _has_price_data(primary):
            return primary
        for suff in CAD_SUFFIX_CANDIDATES[1:]:
            sym = f"{t}{suff}"
            if _has_price_data(sym):
                return sym
        return primary

    if _has_price_data(t):
        return t
    return t

# =========================
# DB I/O
# =========================

def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def _read_table(conn, name: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(f"SELECT * FROM {name}", conn)
    except Exception:
        return pd.DataFrame()

def load_trades(db_path: str) -> pd.DataFrame:
    conn = _connect(db_path)
    df = _read_table(conn, "trades")
    conn.close()

    if df.empty:
        return pd.DataFrame(columns=[
            "trade_id","portfolio_id","portfolio_name","account_number","ticker","currency",
            "action","quantity","price","commission","yahoo_symbol","trade_date","created_at"
        ])

    def norm_str(series):
        return series.astype(str).str.strip().str.upper()

    for col in ["portfolio_name","ticker","currency","action","yahoo_symbol","account_number"]:
        if col in df.columns:
            df[col] = norm_str(df[col])
        else:
            df[col] = ""

    for col in ["quantity","price","commission"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    if "portfolio_id" in df.columns:
        df["portfolio_id"] = pd.to_numeric(df["portfolio_id"], errors="coerce").astype("Int64")
    else:
        df["portfolio_id"] = pd.NA

    if "trade_id" in df.columns:
        df["trade_id"] = pd.to_numeric(df["trade_id"], errors="coerce").astype("Int64")
    else:
        df["trade_id"] = pd.NA

    return df

def load_portfolios(db_path: str) -> pd.DataFrame:
    conn = _connect(db_path)
    try:
        return _read_table(conn, "portfolios")
    finally:
        conn.close()

def backfill_yahoo_symbols(db_path: str, trades_df: pd.DataFrame) -> None:
    if trades_df.empty or "trade_id" not in trades_df.columns:
        return
    missing = (trades_df["yahoo_symbol"].isna()) | (trades_df["yahoo_symbol"].str.strip() == "")
    candidates = trades_df.loc[missing, ["trade_id","ticker","currency"]].dropna(subset=["trade_id"])
    if candidates.empty:
        return
    updates: List[tuple] = []
    for _, r in candidates.iterrows():
        sym = resolve_yahoo_symbol(str(r["ticker"]), str(r["currency"]))
        if sym and _has_price_data(sym):
            updates.append((sym, int(r["trade_id"])))
    if not updates:
        return
    conn = _connect(db_path)
    try:
        with conn:
            conn.executemany("UPDATE trades SET yahoo_symbol = ? WHERE trade_id = ?", updates)
    finally:
        conn.close()

# =========================
# Aggregation & valuation
# =========================

def aggregate_positions(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    sym = trades["yahoo_symbol"].str.strip()
    missing = sym.isna() | (sym == "")
    if missing.any():
        sym.loc[missing] = trades.loc[missing].apply(
            lambda r: resolve_yahoo_symbol(r["ticker"], r["currency"]), axis=1
        )
    trades = trades.assign(effective_symbol=sym.fillna(""))

    sign = trades["action"].map(lambda a: 1.0 if a == "BUY" else (-1.0 if a == "SELL" else 0.0))
    trades = trades.assign(signed_qty=trades["quantity"] * sign)

    trades["portfolio_key"] = trades["portfolio_id"].where(trades["portfolio_id"].notna(), trades["portfolio_name"])

    qty = (
        trades.groupby(["portfolio_key","effective_symbol"], dropna=False)["signed_qty"]
        .sum()
        .reset_index()
        .rename(columns={"signed_qty": "current_qty"})
    )

    buys = trades[trades["action"] == "BUY"].copy()
    buys["buy_cost"] = buys["quantity"] * buys["price"]
    book = buys.groupby(["portfolio_key","effective_symbol"], dropna=False).agg(
        total_buy_qty=("quantity","sum"),
        total_buy_cost=("buy_cost","sum"),
    ).reset_index()

    pos = qty.merge(book, on=["portfolio_key","effective_symbol"], how="left").fillna({
        "total_buy_qty": 0.0, "total_buy_cost": 0.0
    })
    pos["avg_book_price"] = pos.apply(
        lambda r: (r["total_buy_cost"] / r["total_buy_qty"]) if r["total_buy_qty"] > 0 else None, axis=1
    )
    pos["book_value"] = pos.apply(
        lambda r: (r["current_qty"] * r["avg_book_price"]) if r["avg_book_price"] is not None else None, axis=1
    )

    meta = trades.groupby(["portfolio_key","effective_symbol"], dropna=False).agg(
        display_ticker=("ticker","first"),
        currency=("currency","first"),
        yahoo_symbol=("yahoo_symbol","first"),
        portfolio_id=("portfolio_id","first"),
        trade_portfolio_name=("portfolio_name","first"),
    ).reset_index()

    pos = pos.merge(meta, on=["portfolio_key","effective_symbol"], how="left")

    return pos[[
        "portfolio_key","portfolio_id","trade_portfolio_name",
        "display_ticker","currency","yahoo_symbol","effective_symbol",
        "current_qty","avg_book_price","book_value"
    ]]

@st.cache_data(show_spinner=False, ttl=60)
def _fetch_price_single(sym: str, cache_buster: str = "") -> Optional[float]:
    if not sym:
        return None
    try:
        hist = yf.Ticker(sym).history(period="1d", auto_adjust=False, actions=False, raise_errors=False)
        if hist.empty or "Close" not in hist.columns:
            return None
        px = hist["Close"].dropna()
        return float(px.iloc[-1]) if not px.empty else None
    except Exception:
        return None

def fetch_prices(symbols: Iterable[str]) -> Dict[str, Optional[float]]:
    buster = str(st.session_state.get("portfolio_refresh_token", ""))
    out: Dict[str, Optional[float]] = {}
    for s in symbols:
        s = (s or "").strip()
        out[s] = _fetch_price_single(s, cache_buster=buster) if s else None
    return out

def value_positions(pos: pd.DataFrame) -> pd.DataFrame:
    if pos.empty:
        return pos
    pos = pos.copy()
    price_map = fetch_prices(pos["effective_symbol"].fillna("").tolist())
    pos["live_price"] = pos["effective_symbol"].map(price_map)
    pos["market_value"] = pos.apply(
        lambda r: (r["current_qty"] * r["live_price"]) if (r["live_price"] is not None) else None, axis=1
    )
    pos["gain_loss"] = pos.apply(
        lambda r: (r["market_value"] - r["book_value"]) if (r["market_value"] is not None and r["book_value"] is not None) else None, axis=1
    )
    pos["return_pct"] = pos.apply(
        lambda r: (100.0 * r["gain_loss"] / r["book_value"]) if (r.get("gain_loss") is not None and r.get("book_value") not in (None, 0)) else None,
        axis=1
    )
    return pos

# =========================
# Formatting
# =========================

def _format_holdings(df: pd.DataFrame):
    if df.empty:
        return df.style
    df = df.copy()
    for col in ["current_qty","avg_book_price","live_price","book_value","market_value","gain_loss","return_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "current_qty" in df.columns:
        df["current_qty"] = df["current_qty"].apply(lambda x: math.floor(x) if pd.notna(x) else x)
    fmt = {}
    if "current_qty" in df.columns:
        fmt["current_qty"] = "{:,.0f}".format
    for c in ["avg_book_price","live_price","book_value","market_value","gain_loss"]:
        if c in df.columns:
            fmt[c] = "{:,.2f}".format
    if "return_pct" in df.columns:
        fmt["return_pct"] = (lambda x: "" if pd.isna(x) else f"{float(x):.1f}%")
    return df.style.format(fmt, na_rep="")

def _render_summary_bar(valued: pd.DataFrame):
    if valued.empty:
        return
    book_total = pd.to_numeric(valued["book_value"], errors="coerce").sum()
    mkt_total  = pd.to_numeric(valued["market_value"], errors="coerce").sum()
    gl_total   = pd.to_numeric(valued["gain_loss"], errors="coerce").sum()
    ret_pct = None
    if book_total and book_total != 0:
        ret_pct = (gl_total / book_total) * 100.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Book Value",   f"{book_total:,.2f}")
    c2.metric("Market Value", f"{mkt_total:,.2f}")
    c3.metric("Gain / Loss",  f"{gl_total:,.2f}")
    c4.metric("Return %",     ("" if ret_pct is None else f"{ret_pct:.1f}%"))

# =========================
# Settings helpers (read default portfolio)
# =========================

def _read_default_portfolio_id() -> Optional[int]:
    """
    Read default portfolio id from a settings table.
    Supports:
      - key/value table: key='default_portfolio_id'
      - wide table: SELECT default_portfolio_id FROM settings LIMIT 1
    """
    try:
        conn = _connect_for_settings()
        cur = conn.cursor()
        # wide-style
        try:
            row = cur.execute("SELECT default_portfolio_id FROM settings LIMIT 1;").fetchone()
            if row and row[0] is not None:
                return int(row[0])
        except Exception:
            pass
        # key/value
        try:
            row = cur.execute(
                "SELECT value FROM settings WHERE key IN ('default_portfolio_id','DEFAULT_PORTFOLIO_ID') LIMIT 1;"
            ).fetchone()
            if row and row[0] is not None and str(row[0]).strip() != "":
                return int(str(row[0]))
        except Exception:
            pass
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return None

def _connect_for_settings():
    return sqlite3.connect(DB_PATH)

# =========================
# Streamlit UI
# =========================

def portfolio_tab():
    st.subheader("📁 Portfolio")

    trades = load_trades(DB_PATH)
    if trades.empty:
        st.info("No trades found. Add trades in the Trade Blotter.")
        return

    try:
        backfill_yahoo_symbols(DB_PATH, trades)
    except Exception:
        pass

    pos = aggregate_positions(trades)
    if pos.empty or (pd.to_numeric(pos["current_qty"], errors="coerce").fillna(0).abs().sum() == 0):
        st.info("No open positions (net quantity is zero).")
        return

    pf = load_portfolios(DB_PATH)
    if not pf.empty and "portfolio_id" in pos.columns:
        pf_small = pf[["portfolio_id","portfolio_name"]].drop_duplicates()
        pos = pos.merge(pf_small, on="portfolio_id", how="left", suffixes=("", "_canon"))
        pos["portfolio_label"] = pos["portfolio_name"].where(pos["portfolio_name"].notna(), pos["trade_portfolio_name"])
    else:
        pos["portfolio_label"] = pos["trade_portfolio_name"]

    # Build list of selectable portfolios (NO "All" option)
    labels = (
        pos[["portfolio_id","portfolio_label"]]
        .drop_duplicates()
        .sort_values(by=["portfolio_label"])
        .reset_index(drop=True)
    )

    # Determine default selection from settings (by id if present)
    default_pid = _read_default_portfolio_id()
    default_index = 0
    if default_pid is not None:
        try:
            default_index = labels.index[labels["portfolio_id"] == default_pid].tolist()[0]
        except Exception:
            default_index = 0  # fallback if id not found in current list

    sel_label = st.selectbox(
        "Portfolio",
        options=labels["portfolio_label"].tolist(),
        index=min(default_index, len(labels) - 1) if len(labels) else 0
    )

    # Filter to the selected portfolio only
    pos = pos[pos["portfolio_label"] == sel_label]

    # Value positions
    valued = value_positions(pos)

    # Summary bar
    _render_summary_bar(valued)

    # Holdings table
    st.markdown("#### Holdings")
    holdings_view = valued[[
        "display_ticker","currency","effective_symbol",
        "current_qty","avg_book_price","live_price","book_value","market_value","gain_loss","return_pct"
    ]]
    st.dataframe(_format_holdings(holdings_view), use_container_width=True, hide_index=True)

    # Export Holdings
    st.markdown("---")
    st.markdown("### ⬇️ Export Portfolio Holdings")
    colx, coly = st.columns([1,3])
    with colx:
        if st.button("Prepare Excel File", key="btn_prepare_holdings_excel", type="secondary"):
            st.session_state["export_holdings_ready"] = True

    if st.session_state.get("export_holdings_ready"):
        export_df = valued.copy()
        for c in ["current_qty","avg_book_price","live_price","book_value","market_value","gain_loss","return_pct"]:
            if c in export_df.columns:
                export_df[c] = pd.to_numeric(export_df[c], errors="coerce")

        book_total = pd.to_numeric(export_df["book_value"], errors="coerce").sum()
        mkt_total  = pd.to_numeric(export_df["market_value"], errors="coerce").sum()
        gl_total   = pd.to_numeric(export_df["gain_loss"], errors="coerce").sum()
        ret_pct    = (gl_total / book_total * 100.0) if (book_total not in (None, 0) and pd.notna(book_total)) else None

        summary_df = pd.DataFrame([{
            "Portfolio": sel_label,
            "Book Value": book_total,
            "Market Value": mkt_total,
            "Gain/Loss": gl_total,
            "Return %": (None if ret_pct is None else ret_pct)
        }])

        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            cols_order = [
                "display_ticker","currency","effective_symbol",
                "current_qty","avg_book_price","live_price","book_value","market_value","gain_loss","return_pct"
            ]
            cols_order = [c for c in cols_order if c in export_df.columns]
            export_df.to_excel(writer, index=False, sheet_name="Holdings", columns=cols_order)
            summary_df.to_excel(writer, index=False, sheet_name="Summary")

            for ws_name in ["Holdings", "Summary"]:
                ws = writer.sheets[ws_name]
                for col_cells in ws.columns:
                    length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col_cells)
                    ws.column_dimensions[col_cells[0].column_letter].width = min(max(length + 2, 12), 50)

        bio.seek(0)
        safe_name = "".join(ch if (ch.isalnum() or ch in ("-","_")) else "_" for ch in sel_label)
        fname = f"portfolio_holdings_{safe_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%SZ')}.xlsx"

        st.download_button(
            label="Download Excel",
            data=bio.getvalue(),
            file_name=fname,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=False
        )

# If running directly for testing:
if __name__ == "__main__":
    st.set_page_config(page_title="Portfolio", layout="wide")
    portfolio_tab()
