# draupnir_core/forecast_engine.py
"""
Forecast Engine

- Seeds starting assets from CURRENT HOLDINGS (via trades) or from Assets.
- Choice of valuation basis: "market" (live-price) or "book".
- Uses portfolio_flows as source of truth for contributions/withdrawals (respects start/end/frequency).
- Applies distributions from `portfolios`:
    dividend_yield_annual, interest_yield_annual, reinvest_dividends (0/1), reinvest_interest (0/1).
- Dividends are quarterly (Mar/Jun/Sep/Dec), Interest is semiannual (Jun/Dec).
- Dividends are split:
    *eligible* (CAD-weighted portion OR explicit column) vs *non-eligible* (everything else).
- Persists only split dividend fields + interest in results tables (no generic dividend totals).
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# ---------- Core imports ----------
try:
    from .draupnir import (
        get_macro_rates,
        build_inflation_factors,
        simulate_portfolio_growth,
        apply_taxes,
        DEFAULT_GROWTH,
        DEFAULT_INFL,
        DEFAULT_FX,
    )
except Exception:
    from draupnir import (
        get_macro_rates,
        build_inflation_factors,
        simulate_portfolio_growth,
        apply_taxes,
        DEFAULT_GROWTH,
        DEFAULT_INFL,
        DEFAULT_FX,
    )

# ---- DB path ----
os.makedirs("data", exist_ok=True)
DB_PATH_DEFAULT = os.path.join("data", "draupnir.db")

# -----------------------------
# Params
# -----------------------------
@dataclass
class ForecastParams:
    years: int = 30
    cadence: str = "monthly"
    start_date: Optional[str] = None
    use_macro: bool = True
    manual_growth: Optional[float] = None
    manual_inflation: Optional[float] = None
    manual_fx: Optional[float] = None
    notes: Optional[str] = None
    seed_from_holdings: bool = False
    holdings_valuation: str = "market"  # "market" | "book"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

# -----------------------------
# DB helpers
# -----------------------------
def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?;", (table,)).fetchone() is not None

def _get_table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    try:
        return [r[1] for r in conn.execute(f"PRAGMA table_info({table});").fetchall()]
    except Exception:
        return []

def _load_df(conn: sqlite3.Connection, sql: str, params: Tuple = ()) -> pd.DataFrame:
    try: return pd.read_sql(sql, conn, params=params)
    except Exception: return pd.DataFrame()

def _load_global_settings(conn: sqlite3.Connection) -> Dict[str, Any]:
    if not _table_exists(conn, "global_settings"): return {}
    df = _load_df(conn, "SELECT key, value FROM global_settings")
    return {str(k): str(v) for k, v in zip(df["key"], df["value"])}

def load_assets(conn: sqlite3.Connection) -> pd.DataFrame:
    return _load_df(conn, "SELECT * FROM Assets") if _table_exists(conn, "Assets") else pd.DataFrame()

def load_macro_forecast(conn: sqlite3.Connection) -> pd.DataFrame:
    return _load_df(conn, "SELECT * FROM MacroForecast") if _table_exists(conn, "MacroForecast") else pd.DataFrame()

def load_employment_income(conn: sqlite3.Connection) -> pd.DataFrame:
    return _load_df(conn, "SELECT * FROM EmploymentIncome") if _table_exists(conn, "EmploymentIncome") else pd.DataFrame()

def ensure_portfolio_flows_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_flows (
            flow_id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER NOT NULL,
            kind TEXT NOT NULL,
            amount REAL NOT NULL,
            frequency TEXT NOT NULL,   -- monthly | annual | quarterly | semiannual | once
            start_date TEXT NOT NULL,  -- 'YYYY-MM-01'
            end_date TEXT,
            index_with_inflation INTEGER NOT NULL DEFAULT 1,
            notes TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()

def load_portfolio_flows(conn: sqlite3.Connection) -> pd.DataFrame:
    ensure_portfolio_flows_table(conn)
    return _load_df(conn, """
        SELECT flow_id, portfolio_id, kind, amount, frequency, start_date, end_date,
               index_with_inflation, notes, created_at
        FROM portfolio_flows
        ORDER BY portfolio_id, datetime(start_date)
    """)

# -----------------------------
# Holdings & Yahoo helpers
# -----------------------------
def _load_trades(conn: sqlite3.Connection) -> pd.DataFrame:
    if not _table_exists(conn, "trades"): return pd.DataFrame()
    df = _load_df(conn, """
        SELECT trade_id, portfolio_id, portfolio_name, ticker, currency, action, quantity, price,
               commission, yahoo_symbol, trade_date, created_at
        FROM trades
    """)
    for c in ["quantity","price","commission"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

def _load_portfolios_basic(conn: sqlite3.Connection) -> pd.DataFrame:
    if not _table_exists(conn, "portfolios"): return pd.DataFrame()
    return _load_df(conn, "SELECT portfolio_id, portfolio_name, tax_treatment FROM portfolios")

def _aggregate_holdings(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["portfolio_id","portfolio_name","yahoo_symbol","ticker","currency","current_qty","avg_book_price","book_value"])
    t = trades.copy()
    t["action"] = t["action"].astype(str).str.upper()
    t["signed_qty"] = t["quantity"] * t["action"].map(lambda a: 1.0 if a=="BUY" else (-1.0 if a=="SELL" else 0.0))
    qty = (t.groupby(["portfolio_id","portfolio_name","yahoo_symbol","ticker","currency"])["signed_qty"]
             .sum().reset_index().rename(columns={"signed_qty":"current_qty"}))
    buys = t[t["action"]=="BUY"].copy()
    buys["buy_cost"] = buys["quantity"] * buys["price"]
    book = (buys.groupby(["portfolio_id","portfolio_name","yahoo_symbol","ticker","currency"])
                .agg(total_buy_qty=("quantity","sum"), total_buy_cost=("buy_cost","sum")).reset_index())
    pos = qty.merge(book, on=["portfolio_id","portfolio_name","yahoo_symbol","ticker","currency"], how="left").fillna({"total_buy_qty":0.0,"total_buy_cost":0.0})
    pos["avg_book_price"] = pos.apply(lambda r: (r["total_buy_cost"]/r["total_buy_qty"]) if r["total_buy_qty"]>0 else None, axis=1)
    pos["book_value"] = pos.apply(lambda r: (r["current_qty"]*r["avg_book_price"]) if r["avg_book_price"] is not None else None, axis=1)
    pos = pos[pd.to_numeric(pos["current_qty"], errors="coerce").fillna(0).ne(0)]
    return pos.reset_index(drop=True)

def _fetch_prices(symbols: Iterable[str]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for s in symbols:
        key = (s or "").strip()
        if not key: out[key] = None; continue
        try:
            hist = yf.Ticker(key).history(period="1d", auto_adjust=False, actions=False, raise_errors=False)
            if hist.empty or "Close" not in hist.columns: out[key] = None
            else:
                ser = hist["Close"].dropna()
                out[key] = float(ser.iloc[-1]) if not ser.empty else None
        except Exception:
            out[key] = None
    return out

def _fetch_currencies(symbols: Iterable[str]) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for s in symbols:
        key = (s or "").strip()
        if not key: out[key] = None; continue
        curr = None
        try:
            t = yf.Ticker(key)
            fi = getattr(t, "fast_info", None)
            if fi and hasattr(fi, "currency"):
                curr = fi.currency
            if not curr:
                info = {}
                try: info = t.info or {}
                except Exception: info = {}
                curr = info.get("currency")
        except Exception:
            curr = None
        out[key] = curr or None
    return out

def _holdings_to_assets(conn: sqlite3.Connection, valuation: str = "market") -> pd.DataFrame:
    trades = _load_trades(conn)
    if trades.empty: return pd.DataFrame()
    pos = _aggregate_holdings(trades)
    if pos.empty: return pd.DataFrame()
    if valuation.lower() == "market":
        price_map = _fetch_prices(pos["yahoo_symbol"].fillna("").tolist())
        pos["live_price"] = pos["yahoo_symbol"].map(price_map)
        pos["mkt_value"] = pos.apply(lambda r: (r["current_qty"] * r["live_price"]) if pd.notna(r.get("live_price")) else None, axis=1)
        pos["mkt_or_book"] = pos["mkt_value"].where(pos["mkt_value"].notna(), pos["book_value"])
        per_pf = pos.groupby(["portfolio_id","portfolio_name"], dropna=False)["mkt_or_book"].sum(min_count=1).reset_index().rename(columns={"mkt_or_book":"starting_value"})
    else:
        per_pf = pos.groupby(["portfolio_id","portfolio_name"], dropna=False)["book_value"].sum(min_count=1).reset_index().rename(columns={"book_value":"starting_value"})
    pf = _load_portfolios_basic(conn)
    per_pf = per_pf.merge(pf[["portfolio_id","tax_treatment"]], on="portfolio_id", how="left")
    per_pf = per_pf[["portfolio_id","portfolio_name","tax_treatment","starting_value"]]
    per_pf["starting_value"] = pd.to_numeric(per_pf["starting_value"], errors="coerce").fillna(0.0)
    return per_pf[per_pf["starting_value"] > 0].reset_index(drop=True)

def _portfolio_currency_eligible_weight(conn: sqlite3.Connection, valuation: str = "market") -> pd.DataFrame:
    """Per-portfolio CAD weight (0..1). CAD→eligible dividends."""
    trades = _load_trades(conn)
    if trades.empty: return pd.DataFrame(columns=["portfolio_id","eligible_weight"])
    pos = _aggregate_holdings(trades)
    if pos.empty: return pd.DataFrame(columns=["portfolio_id","eligible_weight"])

    if valuation.lower() == "market":
        price_map = _fetch_prices(pos["yahoo_symbol"].fillna("").tolist())
        pos["live_price"] = pos["yahoo_symbol"].map(price_map)
        pos["value"] = pos.apply(lambda r: (r["current_qty"] * r["live_price"]) if pd.notna(r.get("live_price")) else None, axis=1)
        pos["value"] = pos["value"].where(pos["value"].notna(), pos["book_value"])
    else:
        pos["value"] = pos["book_value"]

    missing = pos["currency"].isna() | (pos["currency"].astype(str).str.strip() == "")
    if missing.any():
        cur_map = _fetch_currencies(pos.loc[missing, "yahoo_symbol"].fillna("").tolist())
        pos.loc[missing, "currency"] = pos.loc[missing, "yahoo_symbol"].map(cur_map)

    pos["currency"] = pos["currency"].astype(str).str.upper().str.strip()
    pos["elig_flag"] = np.where(pos["currency"] == "CAD", 1.0, 0.0)
    pos["value"] = pd.to_numeric(pos["value"], errors="coerce").fillna(0.0)
    pos["elig_value"] = pos["value"] * pos["elig_flag"]

    grp = pos.groupby("portfolio_id", as_index=False).agg(total_value=("value","sum"), elig_value=("elig_value","sum"))
    grp["eligible_weight"] = grp.apply(lambda r: (r["elig_value"]/r["total_value"]) if r["total_value"]>0 else 0.0, axis=1)
    return grp[["portfolio_id","eligible_weight"]]

# -----------------------------
# Flows → schedule
# -----------------------------
def _parse_ym(s: Optional[str]) -> Optional[pd.Timestamp]:
    if not s: return None
    ts = pd.to_datetime(str(s), errors="coerce")
    if pd.isna(ts): return None
    return ts.normalize().replace(day=1)

def _months_between(d0: pd.Timestamp, d1: pd.Timestamp) -> int:
    return (d1.year - d0.year) * 12 + (d1.month - d0.month)

def _build_flows_schedule(flows: pd.DataFrame, sim_start: pd.Timestamp, years: int) -> pd.DataFrame:
    if flows is None or flows.empty:
        return pd.DataFrame(columns=["portfolio_id","m_idx","contrib","withdraw","index_flag"])
    horizon = max(1, int(years)) * 12
    rows: List[Dict[str, Any]] = []
    for r in flows.itertuples(index=False):
        pid = int(getattr(r, "portfolio_id"))
        kind = str(getattr(r, "kind", "")).strip().upper()
        freq = str(getattr(r, "frequency", "")).strip().lower()
        amt = float(getattr(r, "amount", 0.0) or 0.0)
        idx = int(getattr(r, "index_with_inflation", 1) or 0)
        sd = _parse_ym(getattr(r, "start_date", None))
        ed = _parse_ym(getattr(r, "end_date", None))
        if amt <= 0 or sd is None: continue

        start_m = max(0, _months_between(sim_start, sd))
        end_m = (horizon - 1) if ed is None else min(horizon - 1, _months_between(sim_start, ed))
        if end_m < 0 or start_m > (horizon - 1): continue

        base_contrib = amt if kind == "CONTRIBUTION" else 0.0
        base_withdr = amt if kind == "WITHDRAWAL" else 0.0
        step_map = {"":1,"monthly":1,"annual":12,"quarterly":3,"semiannual":6}
        if freq == "once":
            if start_m <= end_m:
                rows.append({"portfolio_id": pid, "m_idx": start_m, "contrib": base_contrib, "withdraw": base_withdr, "index_flag": idx})
        else:
            step = step_map.get(freq, 1)
            m = start_m
            while m <= end_m:
                rows.append({"portfolio_id": pid, "m_idx": m, "contrib": base_contrib, "withdraw": base_withdr, "index_flag": idx})
                m += step
    if not rows:
        return pd.DataFrame(columns=["portfolio_id","m_idx","contrib","withdraw","index_flag"])
    sched = pd.DataFrame(rows)
    sched["m_idx"] = sched["m_idx"].astype(int)
    sched["portfolio_id"] = sched["portfolio_id"].astype(int)
    return sched

def _flows_to_monthly_inputs(flows: pd.DataFrame) -> pd.DataFrame:
    if flows is None or flows.empty:
        return pd.DataFrame(columns=["portfolio_id","monthly_contribution","monthly_withdrawal","index_with_inflation"])
    f = flows.copy()
    f["frequency"] = f["frequency"].str.lower().str.strip()
    f = f[f["frequency"].isin(["monthly","annual","quarterly","semiannual","once"])]
    def _to_m(row) -> float:
        amt = float(row["amount"]); frq = str(row["frequency"])
        return {"monthly":amt,"annual":amt/12.0,"quarterly":amt/3.0,"semiannual":amt/6.0}.get(frq, 0.0 if frq=="once" else amt)
    f["m_eq"] = f.apply(_to_m, axis=1)
    contrib = f[f["kind"].str.upper()=="CONTRIBUTION"].groupby("portfolio_id")["m_eq"].sum().rename("monthly_contribution")
    withd = f[f["kind"].str.upper()=="WITHDRAWAL"].groupby("portfolio_id")["m_eq"].sum().rename("monthly_withdrawal")
    infl = f.groupby("portfolio_id")["index_with_inflation"].max().rename("index_with_inflation")
    out = pd.DataFrame({"portfolio_id": sorted(set(f["portfolio_id"].tolist()))})
    out = out.merge(contrib, on="portfolio_id", how="left").merge(withd, on="portfolio_id", how="left").merge(infl, on="portfolio_id", how="left")
    return out.fillna({"monthly_contribution":0.0,"monthly_withdrawal":0.0,"index_with_inflation":1})

# -----------------------------
# Distribution settings
# -----------------------------
DIV_MONTHS = {3, 6, 9, 12}
INT_MONTHS = {6, 12}

def _normalize_yield(series: pd.Series) -> pd.Series:
    """Accept 0.04 or 4 -> always return decimals."""
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    over1 = s > 1.0
    if over1.any():
        s.loc[over1] = s.loc[over1] / 100.0
    return s.clip(lower=0.0)

def _load_portfolio_distribution_settings(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load per-portfolio yield + reinvest flags.

    Supports BOTH the new explicit split fields AND your existing production column names:
      - eligible split aliases:
          ['eligible_dividend_yield_annual', 'dividend_yield_eligible_annual', 'div_eligible_yield']
      - non-eligible split aliases:
          ['noneligible_dividend_yield_annual', 'dividend_yield_noneligible_annual', 'div_noneligible_yield']
      - total dividend yield aliases:
          ['dividend_yield_annual', 'dividend_yield', 'dividend_yield_total']
      - interest yield aliases:
          ['interest_yield_annual', 'interest_yield']
      - reinvest flags:
          ['reinvest_dividends', 'reinvest_dividend'] and ['reinvest_interest']

    Returns a frame with standardized columns the engine uses:
      eligible_dividend_yield_annual, noneligible_dividend_yield_annual,
      dividend_yield_annual, interest_yield_annual,
      reinvest_dividends, reinvest_interest, has_explicit_div_split
    """
    def first_present(cols_available: List[str], *aliases: str) -> Optional[str]:
        for a in aliases:
            if a in cols_available:
                return a
        return None

    if not _table_exists(conn, "portfolios"):
        return pd.DataFrame(columns=[
            "portfolio_id",
            "eligible_dividend_yield_annual","noneligible_dividend_yield_annual",
            "dividend_yield_annual","interest_yield_annual",
            "reinvest_dividends","reinvest_interest","has_explicit_div_split"
        ])

    cols = _get_table_columns(conn, "portfolios")

    # Resolve actual column names present in your DB
    elig_col = first_present(cols,
        "eligible_dividend_yield_annual", "dividend_yield_eligible_annual", "div_eligible_yield"
    )
    non_col = first_present(cols,
        "noneligible_dividend_yield_annual", "dividend_yield_noneligible_annual", "div_noneligible_yield"
    )
    div_total_col = first_present(cols,
        "dividend_yield_annual", "dividend_yield", "dividend_yield_total"
    )
    int_col = first_present(cols,
        "interest_yield_annual", "interest_yield"
    )
    reinv_div_col = first_present(cols, "reinvest_dividends", "reinvest_dividend")
    reinv_int_col = first_present(cols, "reinvest_interest")

    select_cols = ["portfolio_id"]
    for c in [elig_col, non_col, div_total_col, int_col, reinv_div_col, reinv_int_col]:
        if c and c not in select_cols:
            select_cols.append(c)

    # Build SELECT safely
    df = _load_df(conn, f"SELECT {', '.join(select_cols)} FROM portfolios")

    # Create standardized columns (fill missing with defaults)
    def _col(src, default=0.0):
        if src and src in df.columns:
            return df[src]
        return pd.Series([default] * len(df))

    out = pd.DataFrame({
        "portfolio_id": df["portfolio_id"],
        "eligible_dividend_yield_annual": _col(elig_col, 0.0),
        "noneligible_dividend_yield_annual": _col(non_col, 0.0),
        "dividend_yield_annual": _col(div_total_col, 0.0),
        "interest_yield_annual": _col(int_col, 0.0),
        "reinvest_dividends": _col(reinv_div_col, 0).astype("Int64").fillna(0),
        "reinvest_interest": _col(reinv_int_col, 0).astype("Int64").fillna(0),
    })

    # Normalize yields: accept 3.27 or 0.0327 → store as decimals
    for ycol in [
        "eligible_dividend_yield_annual",
        "noneligible_dividend_yield_annual",
        "dividend_yield_annual",
        "interest_yield_annual",
    ]:
        out[ycol] = _normalize_yield(out[ycol])

    # Coerce reinvest flags to 0/1 ints
    out["reinvest_dividends"] = pd.to_numeric(out["reinvest_dividends"], errors="coerce").fillna(0).astype(int).clip(0,1)
    out["reinvest_interest"]  = pd.to_numeric(out["reinvest_interest"],  errors="coerce").fillna(0).astype(int).clip(0,1)

    # Signal whether an explicit split is available (> 0 on either side)
    out["has_explicit_div_split"] = (
        (out["eligible_dividend_yield_annual"] > 0) |
        (out["noneligible_dividend_yield_annual"] > 0)
    ).astype(int)

    return out

# -----------------------------
# Apply distributions
# -----------------------------
def _apply_div_interest_distributions(monthly_df: pd.DataFrame,
                                      portfolio_settings_df: pd.DataFrame,
                                      elig_weights_df: pd.DataFrame) -> pd.DataFrame:
    if monthly_df is None or monthly_df.empty: return monthly_df
    df = monthly_df.copy()
    if "date" not in df.columns: return df

    df["month"] = pd.to_datetime(df["date"], errors="coerce").dt.month

    pset = portfolio_settings_df.copy()
    ew = elig_weights_df.copy() if elig_weights_df is not None else pd.DataFrame(columns=["portfolio_id","eligible_weight"])
    if "eligible_weight" not in ew.columns: ew["eligible_weight"] = 0.0
    df = df.merge(pset, on="portfolio_id", how="left").merge(ew, on="portfolio_id", how="left")
    df["eligible_weight"] = pd.to_numeric(df["eligible_weight"], errors="coerce").fillna(0.0).clip(0.0,1.0)

    # Telemetry we persist
    for c in ["interest_income_base","interest_reinvested_base","cash_flow_from_distributions_base",
              "eligible_dividend_income_base","noneligible_dividend_income_base",
              "eligible_dividends_reinvested_base","noneligible_dividends_reinvested_base"]:
        if c not in df.columns: df[c] = 0.0

    defl = (pd.to_numeric(df.get("value_nominal", 0.0), errors="coerce") /
            pd.to_numeric(df.get("value_real", 0.0), errors="coerce")).replace([np.inf,-np.inf], np.nan).fillna(1.0)

    # Process per-portfolio
    for _, grp in df.groupby("portfolio_id"):
        idx = grp.index
        is_div = grp["month"].isin(DIV_MONTHS)
        is_int = grp["month"].isin(INT_MONTHS)

        val = pd.to_numeric(grp["value_nominal"], errors="coerce").fillna(0.0)
        rdiv = (pd.to_numeric(grp["reinvest_dividends"], errors="coerce").fillna(0).astype(int) != 0)
        rint = (pd.to_numeric(grp["reinvest_interest"],  errors="coerce").fillna(0).astype(int) != 0)

        # --- Dividend split source ---
        has_split = (pd.to_numeric(grp["has_explicit_div_split"], errors="coerce").fillna(0).astype(int) != 0)
        # yields as decimals
        elig_y = pd.to_numeric(grp.get("eligible_dividend_yield_annual", 0.0), errors="coerce").fillna(0.0)
        non_y  = pd.to_numeric(grp.get("noneligible_dividend_yield_annual", 0.0), errors="coerce").fillna(0.0)
        total_y = pd.to_numeric(grp.get("dividend_yield_annual", 0.0), errors="coerce").fillna(0.0)
        w = pd.to_numeric(grp.get("eligible_weight", 0.0), errors="coerce").fillna(0.0).clip(0.0,1.0)

        # If explicit split is provided, ignore weight and use the two yields directly.
        # Else split the single total yield by CAD weight.
        D_elig = np.where(is_div,
                          val * ((np.where(has_split, elig_y, total_y * w)) / 4.0),
                          0.0)
        D_non  = np.where(is_div,
                          val * ((np.where(has_split, non_y, total_y * (1.0 - w))) / 4.0),
                          0.0)
        D_total = D_elig + D_non

        # Dividends reinvestment
        val_after_div = val + np.where(rdiv, D_total, 0.0)

        # Interest (semi-annual) on post-div value
        iy = pd.to_numeric(grp.get("interest_yield_annual", 0.0), errors="coerce").fillna(0.0)
        I = np.where(is_int, val_after_div * (iy / 2.0), 0.0)

        # Telemetry
        df.loc[idx, "eligible_dividend_income_base"] = D_elig
        df.loc[idx, "noneligible_dividend_income_base"] = D_non
        df.loc[idx, "interest_income_base"] = I

        df.loc[idx, "eligible_dividends_reinvested_base"]   = np.where(rdiv, D_elig, 0.0)
        df.loc[idx, "noneligible_dividends_reinvested_base"] = np.where(rdiv, D_non, 0.0)
        df.loc[idx, "interest_reinvested_base"]             = np.where(rint, I, 0.0)

        df.loc[idx, "cash_flow_from_distributions_base"] = np.where(rdiv, 0.0, D_total) + np.where(rint, 0.0, I)

        # Update portfolio values (reinvested amounts raise NAV)
        val_new = val_after_div + np.where(rint, I, 0.0)
        df.loc[idx, "value_nominal"] = val_new
        df.loc[idx, "value_real"] = pd.to_numeric(grp["value_real"], errors="coerce").fillna(0.0) + \
            (np.where(rdiv, D_total, 0.0) + np.where(rint, I, 0.0)) / defl.loc[idx].replace(0,1.0)

    df.drop(columns=["month"], inplace=True, errors="ignore")
    return df

# -----------------------------
# Core runner
# -----------------------------
def _constant_macro_df(years: int, growth: float, inflation: float, fx: float) -> pd.DataFrame:
    yrs = list(range(0, max(120, int(years) + 1)))
    return pd.DataFrame({"year": yrs, "growth": growth, "inflation": inflation, "fx": fx})

def _resolve_sim_start(start_date: Optional[str]) -> pd.Timestamp:
    if start_date:
        try: return pd.to_datetime(start_date).normalize().replace(day=1)
        except Exception: pass
    return pd.Timestamp.today().normalize().replace(day=1)

def run_forecast(
    db_path: str = DB_PATH_DEFAULT,
    params: ForecastParams | Dict[str, Any] = ForecastParams(),
    write_to_db: bool = True,
    return_frames: bool = True,
) -> Dict[str, Any]:
    fp = ForecastParams(**params) if isinstance(params, dict) else params
    conn = _connect(db_path)
    try:
        settings = _load_global_settings(conn)

        assets_df = _holdings_to_assets(conn, valuation=fp.holdings_valuation) if fp.seed_from_holdings else load_assets(conn)

        if fp.use_macro:
            macro_df = load_macro_forecast(conn)
        else:
            g = fp.manual_growth if fp.manual_growth is not None else DEFAULT_GROWTH
            i = fp.manual_inflation if fp.manual_inflation is not None else DEFAULT_INFL
            x = fp.manual_fx if fp.manual_fx is not None else DEFAULT_FX
            macro_df = _constant_macro_df(fp.years, g, i, x)

        income_df = load_employment_income(conn)

        flows_df = load_portfolio_flows(conn)
        sim_start = _resolve_sim_start(fp.start_date)
        schedule_df = _build_flows_schedule(flows_df, sim_start, fp.years)
        inputs_df = _flows_to_monthly_inputs(flows_df)
        if not schedule_df.empty and not inputs_df.empty:
            mask = inputs_df["portfolio_id"].isin(set(schedule_df["portfolio_id"].unique()))
            if "monthly_contribution" in inputs_df.columns: inputs_df.loc[mask, "monthly_contribution"] = 0.0
            if "monthly_withdrawal"   in inputs_df.columns: inputs_df.loc[mask, "monthly_withdrawal"] = 0.0

        macro_rates  = get_macro_rates(macro_df, settings, inputs_df)
        infl_factors = build_inflation_factors(macro_rates, settings, inputs_df)

        monthly_df = simulate_portfolio_growth(
            assets=assets_df, inputs=inputs_df, macro=macro_rates, inflation=infl_factors,
            settings=settings, years=fp.years, cadence=fp.cadence,
            start_date=str(sim_start.date()), flows_schedule=schedule_df
        )

        # Prefer explicit eligible/non-eligible yields; fall back to CAD-weight split
        elig_w = _portfolio_currency_eligible_weight(
            conn,
            valuation=("market" if fp.seed_from_holdings and fp.holdings_valuation == "market" else "book")
        )
        pset_df = _load_portfolio_distribution_settings(conn)
        monthly_df = _apply_div_interest_distributions(monthly_df, pset_df, elig_w)

        annual_df = apply_taxes(monthly_df=monthly_df, employment_income=income_df, tax_rules_conn=conn, settings=settings, macro=macro_rates)

        run_id = None
        if write_to_db:
            _ensure_forecast_run_tables(conn)
            _ensure_annual_schema(conn)
            run_id = _insert_forecast_run(conn, fp, settings)
            _write_results(conn, run_id, monthly_df, annual_df)

        return {
            "run_id": run_id,
            "monthly_df": monthly_df if return_frames else None,
            "annual_df": annual_df if return_frames else None,
            "settings": settings,
            "params": asdict(fp),
            "seeded_from": ("holdings" if fp.seed_from_holdings else "assets"),
        }
    finally:
        conn.close()

# -----------------------------
# Results persistence (lean schemas)
# -----------------------------
def _ensure_forecast_run_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS forecast_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            params_json TEXT,
            settings_json TEXT
        );
    """)
    want_cols = [
        "run_id","period","portfolio_id","portfolio_name","tax_treatment",
        "nominal_value","real_value","contributions","withdrawals",
        "cash_flow_from_distributions_base",
        "interest_income_base","interest_reinvested_base",
        "eligible_dividend_income_base","noneligible_dividend_income_base",
        "eligible_dividends_reinvested_base","noneligible_dividends_reinvested_base"
    ]
    if not _table_exists(conn, "forecast_results_monthly"):
        conn.executescript("""
            CREATE TABLE forecast_results_monthly (
                run_id INTEGER,
                period TEXT,
                portfolio_id INTEGER,
                portfolio_name TEXT,
                tax_treatment TEXT,
                nominal_value REAL,
                real_value REAL,
                contributions REAL,
                withdrawals REAL,
                cash_flow_from_distributions_base REAL,
                interest_income_base REAL,
                interest_reinvested_base REAL,
                eligible_dividend_income_base REAL,
                noneligible_dividend_income_base REAL,
                eligible_dividends_reinvested_base REAL,
                noneligible_dividends_reinvested_base REAL,
                PRIMARY KEY (run_id, period, portfolio_id)
            );
        """)
        conn.commit()
        return
    have = _get_table_columns(conn, "forecast_results_monthly")
    if set(have) == set(want_cols): return
    try: old = pd.read_sql("SELECT * FROM forecast_results_monthly", conn)
    except Exception: old = pd.DataFrame()
    new = pd.DataFrame({c: old[c] if c in old.columns else None for c in want_cols})
    conn.execute("DROP TABLE IF EXISTS forecast_results_monthly_new;")
    conn.executescript("""
        CREATE TABLE forecast_results_monthly_new (
            run_id INTEGER,
            period TEXT,
            portfolio_id INTEGER,
            portfolio_name TEXT,
            tax_treatment TEXT,
            nominal_value REAL,
            real_value REAL,
            contributions REAL,
            withdrawals REAL,
            cash_flow_from_distributions_base REAL,
            interest_income_base REAL,
            interest_reinvested_base REAL,
            eligible_dividend_income_base REAL,
            noneligible_dividend_income_base REAL,
            eligible_dividends_reinvested_base REAL,
            noneligible_dividends_reinvested_base REAL,
            PRIMARY KEY (run_id, period, portfolio_id)
        );
    """)
    if not new.empty:
        new.to_sql("forecast_results_monthly_new", conn, if_exists="append", index=False)
    conn.execute("DROP TABLE IF EXISTS forecast_results_monthly;")
    conn.execute("ALTER TABLE forecast_results_monthly_new RENAME TO forecast_results_monthly;")
    conn.commit()

_ANNUAL_COLUMNS = [
    ("run_id","INTEGER"),("year","INTEGER"),("portfolio_id","INTEGER"),
    ("portfolio_name","TEXT"),("tax_treatment","TEXT"),
    ("nominal_pretax_income","REAL"),("nominal_taxes_paid","REAL"),
    ("nominal_after_tax_income","REAL"),("nominal_effective_tax_rate","REAL"),
    ("real_pretax_income","REAL"),("real_taxes_paid","REAL"),
    ("real_after_tax_income","REAL"),("real_effective_tax_rate","REAL"),
    ("contributions","REAL"),("withdrawals","REAL"),
    ("interest_income_base","REAL"),("interest_reinvested_base","REAL"),
    ("eligible_dividend_income_base","REAL"),("noneligible_dividend_income_base","REAL"),
    ("eligible_dividends_reinvested_base","REAL"),("noneligible_dividends_reinvested_base","REAL"),
]

def _create_annual_table(conn: sqlite3.Connection, table_name: str = "forecast_results_annual") -> None:
    cols_sql = ", ".join([f"{name} {typ}" for name, typ in _ANNUAL_COLUMNS])
    conn.executescript(f"CREATE TABLE IF NOT EXISTS {table_name} ({cols_sql}, PRIMARY KEY (run_id, year, portfolio_id));")

def _ensure_annual_schema(conn: sqlite3.Connection) -> None:
    want_cols = [c[0] for c in _ANNUAL_COLUMNS]
    if not _table_exists(conn, "forecast_results_annual"):
        _create_annual_table(conn, "forecast_results_annual"); conn.commit(); return
    have = _get_table_columns(conn, "forecast_results_annual")
    if set(have) == set(want_cols): return
    try: old = pd.read_sql("SELECT * FROM forecast_results_annual", conn)
    except Exception:
        conn.execute("DROP TABLE IF EXISTS forecast_results_annual;")
        _create_annual_table(conn, "forecast_results_annual"); conn.commit(); return

    def g(col, default=None): return old[col] if col in old.columns else default
    new = pd.DataFrame({
        "run_id": g("run_id", 0), "year": g("year", 0), "portfolio_id": g("portfolio_id", 0),
        "portfolio_name": g("portfolio_name", ""), "tax_treatment": g("tax_treatment", ""),
        "nominal_pretax_income": ( g("nominal_pretax_income", None)
                                   if "nominal_pretax_income" in old.columns else
                                   ((g("after_tax_income", 0.0).fillna(0.0) + g("taxes_paid", 0.0).fillna(0.0))
                                     if "after_tax_income" in old.columns or "taxes_paid" in old.columns else 0.0) ),
        "nominal_taxes_paid": g("nominal_taxes_paid", g("taxes_paid", 0.0)),
        "nominal_after_tax_income": g("nominal_after_tax_income", g("after_tax_income", 0.0)),
        "nominal_effective_tax_rate": None,
        "real_pretax_income": g("real_pretax_income", 0.0),
        "real_taxes_paid": g("real_taxes_paid", 0.0),
        "real_after_tax_income": g("real_after_tax_income", 0.0),
        "real_effective_tax_rate": None,
        "contributions": g("contributions", 0.0),
        "withdrawals": g("withdrawals", 0.0),
        "interest_income_base": g("interest_income_base", 0.0),
        "interest_reinvested_base": g("interest_reinvested_base", 0.0),
        "eligible_dividend_income_base": g("eligible_dividend_income_base", 0.0),
        "noneligible_dividend_income_base": g("noneligible_dividend_income_base", 0.0),
        "eligible_dividends_reinvested_base": g("eligible_dividends_reinvested_base", 0.0),
        "noneligible_dividends_reinvested_base": g("noneligible_dividends_reinvested_base", 0.0),
    })
    for c in ["nominal_pretax_income","nominal_taxes_paid","nominal_after_tax_income",
              "real_pretax_income","real_taxes_paid","real_after_tax_income",
              "contributions","withdrawals",
              "interest_income_base","interest_reinvested_base",
              "eligible_dividend_income_base","noneligible_dividend_income_base",
              "eligible_dividends_reinvested_base","noneligible_dividends_reinvested_base"]:
        new[c] = pd.to_numeric(new[c], errors="coerce").fillna(0.0)
    new["nominal_effective_tax_rate"] = new.apply(lambda r: (r["nominal_taxes_paid"]/r["nominal_pretax_income"]) if r["nominal_pretax_income"]>0 else 0.0, axis=1)
    new["real_effective_tax_rate"] = new.apply(lambda r: (r["real_taxes_paid"]/r["real_pretax_income"]) if r["real_pretax_income"]>0 else 0.0, axis=1)

    conn.execute("DROP TABLE IF EXISTS forecast_results_annual_new;")
    _create_annual_table(conn, "forecast_results_annual_new")
    new = new[want_cols]
    if not new.empty:
        new.to_sql("forecast_results_annual_new", conn, if_exists="append", index=False)
    conn.execute("DROP TABLE IF EXISTS forecast_results_annual;")
    conn.execute("ALTER TABLE forecast_results_annual_new RENAME TO forecast_results_annual;")
    conn.commit()

def _insert_forecast_run(conn: sqlite3.Connection, fp: ForecastParams, settings: Dict[str, Any]) -> int:
    cur = conn.execute(
        "INSERT INTO forecast_runs (created_at, params_json, settings_json) VALUES (?, ?, ?)",
        (datetime.utcnow().isoformat(timespec="seconds") + "Z", fp.to_json(), json.dumps(settings)),
    )
    conn.commit()
    return int(cur.lastrowid)

def _write_results(conn: sqlite3.Connection, run_id: int, monthly_df: pd.DataFrame, annual_df: pd.DataFrame) -> None:
    # Monthly
    m = monthly_df.copy()
    m = m.rename(columns={"date":"period","portfolio":"portfolio_name","value_nominal":"nominal_value","value_real":"real_value"})
    base_cols = ["period","portfolio_id","portfolio_name","tax_treatment","nominal_value","real_value","contributions","withdrawals"]
    for col in base_cols:
        if col not in m.columns: m[col] = None
    telem = ["cash_flow_from_distributions_base","interest_income_base","interest_reinvested_base",
             "eligible_dividend_income_base","noneligible_dividend_income_base",
             "eligible_dividends_reinvested_base","noneligible_dividends_reinvested_base"]
    for c in telem:
        if c not in m.columns: m[c] = 0.0
    m["run_id"] = run_id
    m_cols = ["run_id"] + base_cols + telem
    m[m_cols].to_sql("forecast_results_monthly", conn, if_exists="append", index=False)

    # Annual
    a = annual_df.copy() if annual_df is not None else pd.DataFrame()
    if a.empty:
        a = pd.DataFrame(columns=["year","portfolio_id","portfolio","tax_treatment","after_tax","after_tax_real","taxes"])
    a = a.rename(columns={"portfolio":"portfolio_name","after_tax":"after_tax_income","after_tax_real":"real_after_tax_income","taxes":"taxes_paid"})
    for col in ["year","portfolio_id","portfolio_name","tax_treatment","after_tax_income","real_after_tax_income","taxes_paid"]:
        if col not in a.columns: a[col] = None

    mm = monthly_df.copy()
    mm["year"] = pd.to_datetime(mm["date"]).dt.year
    defl = (pd.to_numeric(mm["value_nominal"], errors="coerce") / pd.to_numeric(mm["value_real"], errors="coerce")).replace([np.inf,-np.inf], np.nan).fillna(1.0)
    mm["_defl"] = defl
    for fld in ["contributions","withdrawals"]:
        if fld not in mm.columns: mm[fld] = 0.0
        mm[f"{fld}_real"] = pd.to_numeric(mm[fld], errors="coerce").fillna(0.0) / mm["_defl"].replace(0,1.0)

    dist_cols = ["interest_income_base","interest_reinvested_base",
                 "eligible_dividend_income_base","noneligible_dividend_income_base",
                 "eligible_dividends_reinvested_base","noneligible_dividends_reinvested_base"]
    for dc in dist_cols:
        if dc not in mm.columns: mm[dc] = 0.0
    dist_annual = mm.groupby(["year","portfolio_id"], as_index=False)[dist_cols].sum()

    flows_real = (mm.groupby(["year","portfolio_id"], as_index=False)[["contributions_real","withdrawals_real"]].sum()
                    .rename(columns={"contributions_real":"contributions","withdrawals_real":"withdrawals"}))

    a_num = a.copy()
    a_num["after_tax_income"] = pd.to_numeric(a_num["after_tax_income"], errors="coerce").fillna(0.0)
    a_num["real_after_tax_income"] = pd.to_numeric(a_num["real_after_tax_income"], errors="coerce").fillna(0.0)
    a_num["taxes_paid"] = pd.to_numeric(a_num["taxes_paid"], errors="coerce").fillna(0.0)
    a_num["nominal_pretax_income"] = a_num["after_tax_income"] + a_num["taxes_paid"]
    a_num["nominal_taxes_paid"] = a_num["taxes_paid"]
    a_num["nominal_after_tax_income"] = a_num["after_tax_income"]
    a_num["real_taxes_paid"] = a_num["nominal_taxes_paid"] / a_num.apply(lambda r: (r["after_tax_income"]/r["real_after_tax_income"]) if r["real_after_tax_income"]>0 else 1.0, axis=1).replace([np.inf,-np.inf], np.nan).fillna(1.0)
    a_num["real_pretax_income"] = a_num["real_after_tax_income"] + a_num["real_taxes_paid"]
    a_num["nominal_effective_tax_rate"] = a_num.apply(lambda r: (r["nominal_taxes_paid"]/r["nominal_pretax_income"]) if r["nominal_pretax_income"]>0 else 0.0, axis=1)
    a_num["real_effective_tax_rate"] = a_num.apply(lambda r: (r["real_taxes_paid"]/r["real_pretax_income"]) if r["real_pretax_income"]>0 else 0.0, axis=1)

    a_enriched = a_num.merge(flows_real, on=["year","portfolio_id"], how="left").merge(dist_annual, on=["year","portfolio_id"], how="left")
    a_enriched[["contributions","withdrawals"]] = a_enriched[["contributions","withdrawals"]].fillna(0.0)
    a_enriched["run_id"] = run_id

    _ensure_annual_schema(conn)
    out_cols = [c[0] for c in _ANNUAL_COLUMNS]
    conn.execute("DELETE FROM forecast_results_annual WHERE run_id = ?;", (run_id,))
    conn.commit()
    a_enriched[out_cols].to_sql("forecast_results_annual", conn, if_exists="append", index=False)
    conn.commit()
