# draupnir_core/summary.py
import sqlite3
from typing import Optional, Dict, List

import pandas as pd
import streamlit as st
import yfinance as yf

from .portfolio import (
    DB_PATH,
    load_trades,
    load_portfolios,
    backfill_yahoo_symbols,
    aggregate_positions,
    value_positions,  # cache respects portfolio_refresh_token
)

# -----------------------------
# Settings / Base currency
# -----------------------------

def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def _read_settings_base_currency() -> Optional[str]:
    try:
        conn = _connect(DB_PATH)
        cur = conn.cursor()
        # Try wide schema
        try:
            row = cur.execute("SELECT base_currency FROM settings LIMIT 1;").fetchone()
            if row and row[0]:
                return str(row[0]).strip().upper()
        except Exception:
            pass
        # Try key/value schema
        try:
            row = cur.execute(
                "SELECT value FROM settings WHERE key IN ('base_currency','BASE_CURRENCY') LIMIT 1;"
            ).fetchone()
            if row and row[0]:
                return str(row[0]).strip().upper()
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

def _get_base_currency() -> str:
    base = _read_settings_base_currency()
    if base in ("CAD", "USD"):
        return base
    return "CAD"

# -----------------------------
# FX helpers (lightweight)
# -----------------------------

@st.cache_data(show_spinner=False, ttl=300)
def _fetch_fx_pair(symbol: str) -> Optional[float]:
    try:
        hist = yf.Ticker(symbol).history(period="1d", auto_adjust=False, actions=False, raise_errors=False)
        if hist.empty or "Close" not in hist.columns:
            return None
        close = hist["Close"].dropna()
        return float(close.iloc[-1]) if not close.empty else None
    except Exception:
        return None

def _fx_to_base_rates(base: str) -> Dict[str, float]:
    base = (base or "").upper()
    rates = {"CAD": 1.0, "USD": 1.0}
    if base == "CAD":
        r = _fetch_fx_pair("USDCAD=X")
        rates["USD"] = r if r and r > 0 else 1.35
    elif base == "USD":
        r = _fetch_fx_pair("CADUSD=X")
        rates["CAD"] = r if r and r > 0 else 0.74
    return rates

def _apply_fx(valued: pd.DataFrame, base: str) -> pd.DataFrame:
    if valued.empty:
        return valued
    df = valued.copy()
    for c in ["book_value","market_value","gain_loss"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    fx = _fx_to_base_rates(base)
    def _mult_row(row):
        return fx.get(str(row.get("currency", "")).upper(), 1.0)
    m = df.apply(_mult_row, axis=1)
    df["book_value_base"]   = (df["book_value"]   * m).where(df["book_value"].notna())
    df["market_value_base"] = (df["market_value"] * m).where(df["market_value"].notna())
    df["gain_loss_base"]    = (df["gain_loss"]    * m).where(df["gain_loss"].notna())
    df["return_pct_base"]   = (100.0 * df["gain_loss_base"] / df["book_value_base"]).where(
        (df["book_value_base"].notna()) & (df["book_value_base"] != 0)
    )
    return df

# -----------------------------
# Formatting (simple string fmt for summary tables)
# -----------------------------

def _fmt_money(x) -> str:
    return "" if pd.isna(x) else f"{float(x):,.2f}"

def _fmt_pct(x) -> str:
    return "" if pd.isna(x) else f"{float(x):.1f}%"

def _format_summary(df: pd.DataFrame, cols_money: List[str], col_pct: Optional[str]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for c in cols_money:
        if c in out.columns:
            out[c] = out[c].map(_fmt_money)
    if col_pct and col_pct in out.columns:
        out[col_pct] = out[col_pct].map(_fmt_pct)
    return out

# -----------------------------
# Totals bar (like Portfolio tab)
# -----------------------------

def _render_totals_bar(book_series: pd.Series, mkt_series: pd.Series, gl_series: pd.Series):
    book_total = pd.to_numeric(book_series, errors="coerce").sum()
    mkt_total  = pd.to_numeric(mkt_series,  errors="coerce").sum()
    gl_total   = pd.to_numeric(gl_series,   errors="coerce").sum()
    ret_pct = (gl_total / book_total * 100.0) if (book_total not in (None, 0) and pd.notna(book_total)) else None
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Book Value",   f"{book_total:,.2f}")
    c2.metric("Market Value", f"{mkt_total:,.2f}")
    c3.metric("Gain / Loss",  f"{gl_total:,.2f}")
    c4.metric("Return %",     "" if ret_pct is None else f"{ret_pct:.1f}%")

# -----------------------------
# Summary Tab
# -----------------------------

def summary_tab():
    st.subheader("📈 Summary")

    base = _get_base_currency()
    st.caption(f"All amounts shown in base currency: **{base}**")

    # 1) Load trades
    trades = load_trades(DB_PATH)
    if trades.empty:
        st.info("No trades found. Add trades in the Trade Blotter.")
        return

    # 2) Backfill any missing symbols (quiet)
    try:
        backfill_yahoo_symbols(DB_PATH, trades)
    except Exception:
        pass

    # 3) Aggregate and value using the Portfolio helpers (auto-refresh via token)
    pos = aggregate_positions(trades)
    if pos.empty:
        st.info("No open positions.")
        return

    # Attach canonical portfolio labels (same as portfolio tab)
    pf = load_portfolios(DB_PATH)
    if not pf.empty and "portfolio_id" in pos.columns:
        pf_small = pf[["portfolio_id","portfolio_name"]].drop_duplicates()
        pos = pos.merge(pf_small, on="portfolio_id", how="left", suffixes=("", "_canon"))
        pos["portfolio_label"] = pos["portfolio_name"].where(pos["portfolio_name"].notna(), pos["trade_portfolio_name"])
    else:
        pos["portfolio_label"] = pos["trade_portfolio_name"]

    valued = value_positions(pos)
    if valued.empty:
        st.info("No valued positions.")
        return

    # 4) FX to base
    valued = _apply_fx(valued, base)

    # View switcher
    view = st.radio("View", options=["By Portfolio", "By Asset"], horizontal=True)

    if view == "By Portfolio":
        grp = valued.groupby("portfolio_label", dropna=False).agg(
            book_value_base=("book_value_base", "sum"),
            market_value_base=("market_value_base", "sum"),
            gain_loss_base=("gain_loss_base", "sum"),
        ).reset_index()
        grp["return_pct_base"] = (100.0 * grp["gain_loss_base"] / grp["book_value_base"]).where(
            (grp["book_value_base"].notna()) & (grp["book_value_base"] != 0)
        )

        # Totals bar (over all portfolios in view)
        _render_totals_bar(grp["book_value_base"], grp["market_value_base"], grp["gain_loss_base"])

        display = grp.rename(columns={
            "portfolio_label": "Portfolio",
            "book_value_base": "Book Value",
            "market_value_base": "Market Value",
            "gain_loss_base": "Gain/Loss",
            "return_pct_base": "Return %",
        })
        display = _format_summary(display, ["Book Value","Market Value","Gain/Loss"], "Return %")
        st.dataframe(display, use_container_width=True, hide_index=True)

    else:  # By Asset
        grp = valued.groupby(["display_ticker"], dropna=False).agg(
            book_value_base=("book_value_base", "sum"),
            market_value_base=("market_value_base", "sum"),
            gain_loss_base=("gain_loss_base", "sum"),
        ).reset_index()
        grp["return_pct_base"] = (100.0 * grp["gain_loss_base"] / grp["book_value_base"]).where(
            (grp["book_value_base"].notna()) & (grp["book_value_base"] != 0)
        )

        # Totals bar (over all tickers in view)
        _render_totals_bar(grp["book_value_base"], grp["market_value_base"], grp["gain_loss_base"])

        display = grp.rename(columns={
            "display_ticker": "Ticker",
            "book_value_base": "Book Value",
            "market_value_base": "Market Value",
            "gain_loss_base": "Gain/Loss",
            "return_pct_base": "Return %",
        })
        display = _format_summary(display, ["Book Value","Market Value","Gain/Loss"], "Return %")
        st.dataframe(display, use_container_width=True, hide_index=True)
