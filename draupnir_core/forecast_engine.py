# draupnir_core/forecast_engine.py
"""
Forecast Engine (callable)

Enhancement C:
- Option to seed starting assets from CURRENT HOLDINGS (derived from trades)
- Choice of valuation basis: "market" (live-price) or "book"
- Maps portfolios → tax_treatment automatically via portfolios table

Also retains Feature A (portfolio_flows) and the rest of the engine.
Imports for draupnir/tax_engine are robust whether they live in the package or root.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, Iterable
import os

import pandas as pd
import yfinance as yf

# ---------- Robust imports for core math ----------
try:
    from .draupnir import (
        get_macro_rates,
        build_inflation_factors,
        simulate_portfolio_growth,
        apply_taxes,
    )
    from . import tax_engine as tax_engine
except Exception:
    from draupnir import (
        get_macro_rates,
        build_inflation_factors,
        simulate_portfolio_growth,
        apply_taxes,
    )
    import tax_engine as tax_engine

# ---- Unified DB default ----
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
    inflation_mode: Optional[str] = None
    growth_mode: Optional[str] = None
    fx_mode: Optional[str] = None
    notes: Optional[str] = None

    # Enhancement C:
    seed_from_holdings: bool = False                  # if True, ignore Assets table and derive from trades
    holdings_valuation: str = "market"                # "market" or "book"

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
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?;", (table,)
    ).fetchone()
    return row is not None

def _load_df(conn: sqlite3.Connection, sql: str, params: Tuple = ()) -> pd.DataFrame:
    try:
        return pd.read_sql(sql, conn, params=params)
    except Exception:
        return pd.DataFrame()

def _load_global_settings(conn: sqlite3.Connection) -> Dict[str, Any]:
    """
    Read key/value pairs from global_settings table. Never crashes; returns {} on error.
    """
    try:
        if not _table_exists(conn, "global_settings"):
            return {}
        df = pd.read_sql("SELECT key, value FROM global_settings", conn)
        return {str(k): str(v) for k, v in zip(df["key"], df["value"])}
    except Exception:
        return {}

def load_assets(conn: sqlite3.Connection) -> pd.DataFrame:
    if not _table_exists(conn, "Assets"):
        return pd.DataFrame()
    return _load_df(conn, "SELECT * FROM Assets")

def load_projection_inputs(conn: sqlite3.Connection) -> pd.DataFrame:
    if not _table_exists(conn, "ProjectionInputs"):
        return pd.DataFrame()
    return _load_df(conn, "SELECT * FROM ProjectionInputs")

def load_macro_forecast(conn: sqlite3.Connection) -> pd.DataFrame:
    if not _table_exists(conn, "MacroForecast"):
        return pd.DataFrame()
    return _load_df(conn, "SELECT * FROM MacroForecast")

def load_employment_income(conn: sqlite3.Connection) -> pd.DataFrame:
    if not _table_exists(conn, "EmploymentIncome"):
        return pd.DataFrame()
    return _load_df(conn, "SELECT * FROM EmploymentIncome")

def ensure_portfolio_flows_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_flows (
            flow_id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER NOT NULL,
            kind TEXT NOT NULL,                  -- 'CONTRIBUTION' or 'WITHDRAWAL'
            amount REAL NOT NULL,                -- amount in portfolio currency
            frequency TEXT NOT NULL,             -- 'monthly' or 'annual'
            start_date TEXT NOT NULL,            -- 'YYYY-MM-01'
            end_date TEXT,                       -- nullable; if NULL, open-ended
            index_with_inflation INTEGER NOT NULL DEFAULT 1, -- 1/0
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
# Enhancement C: build assets from holdings
# -----------------------------

def _load_trades(conn: sqlite3.Connection) -> pd.DataFrame:
    if not _table_exists(conn, "trades"):
        return pd.DataFrame()
    df = _load_df(conn, """
        SELECT trade_id, portfolio_id, portfolio_name, ticker, currency, action, quantity, price,
               commission, yahoo_symbol, trade_date, created_at
        FROM trades
    """)
    for c in ["quantity", "price", "commission"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

def _load_portfolios(conn: sqlite3.Connection) -> pd.DataFrame:
    if not _table_exists(conn, "portfolios"):
        return pd.DataFrame()
    return _load_df(conn, "SELECT portfolio_id, portfolio_name, tax_treatment FROM portfolios")

def _aggregate_holdings(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=[
            "portfolio_id","portfolio_name","yahoo_symbol","ticker","currency",
            "current_qty","avg_book_price","book_value"
        ])

    t = trades.copy()
    t["action"] = t["action"].astype(str).str.upper()
    t["signed_qty"] = t["quantity"] * t["action"].map(lambda a: 1.0 if a=="BUY" else (-1.0 if a=="SELL" else 0.0))
    qty = (t.groupby(["portfolio_id","portfolio_name","yahoo_symbol","ticker","currency"])["signed_qty"]
             .sum().reset_index().rename(columns={"signed_qty":"current_qty"}))

    buys = t[t["action"]=="BUY"].copy()
    buys["buy_cost"] = buys["quantity"] * buys["price"]
    book = (buys.groupby(["portfolio_id","portfolio_name","yahoo_symbol","ticker","currency"])
                .agg(total_buy_qty=("quantity","sum"), total_buy_cost=("buy_cost","sum"))
                .reset_index())
    pos = qty.merge(book, on=["portfolio_id","portfolio_name","yahoo_symbol","ticker","currency"], how="left").fillna({
        "total_buy_qty":0.0,"total_buy_cost":0.0
    })
    pos["avg_book_price"] = pos.apply(
        lambda r: (r["total_buy_cost"]/r["total_buy_qty"]) if r["total_buy_qty"]>0 else None, axis=1
    )
    pos["book_value"] = pos.apply(
        lambda r: (r["current_qty"]*r["avg_book_price"]) if r["avg_book_price"] is not None else None, axis=1
    )
    pos = pos[pd.to_numeric(pos["current_qty"], errors="coerce").fillna(0).ne(0)]
    return pos.reset_index(drop=True)

def _fetch_prices(symbols: Iterable[str]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for s in symbols:
        s = (s or "").strip()
        if not s:
            out[s] = None
            continue
        try:
            hist = yf.Ticker(s).history(period="1d", auto_adjust=False, actions=False, raise_errors=False)
            if hist.empty or "Close" not in hist.columns:
                out[s] = None
            else:
                ser = hist["Close"].dropna()
                out[s] = float(ser.iloc[-1]) if not ser.empty else None
        except Exception:
            out[s] = None
    return out

def _holdings_to_assets(conn: sqlite3.Connection, valuation: str = "market") -> pd.DataFrame:
    """
    Build starting assets per portfolio from current holdings. If valuation='market',
    we use live prices where available and **fall back to book_value** where prices
    are missing to avoid zero starts.
    """
    trades = _load_trades(conn)
    if trades.empty:
        return pd.DataFrame()

    pos = _aggregate_holdings(trades)
    if pos.empty:
        return pd.DataFrame()

    if valuation.lower() == "market":
        price_map = _fetch_prices(pos["yahoo_symbol"].fillna("").tolist())
        pos["live_price"] = pos["yahoo_symbol"].map(price_map)
        pos["mkt_value"] = pos.apply(
            lambda r: (r["current_qty"] * r["live_price"]) if pd.notna(r.get("live_price")) else None,
            axis=1
        )
        # Fallback to book_value where market price is missing
        pos["mkt_or_book"] = pos["mkt_value"].where(pos["mkt_value"].notna(), pos["book_value"])
        per_pf = pos.groupby(["portfolio_id","portfolio_name"], dropna=False)["mkt_or_book"].sum(min_count=1).reset_index()
        per_pf = per_pf.rename(columns={"mkt_or_book":"starting_value"})
    else:
        per_pf = pos.groupby(["portfolio_id","portfolio_name"], dropna=False)["book_value"].sum(min_count=1).reset_index()
        per_pf = per_pf.rename(columns={"book_value":"starting_value"})

    pf = _load_portfolios(conn)
    per_pf = per_pf.merge(pf[["portfolio_id","tax_treatment"]], on="portfolio_id", how="left")

    per_pf = per_pf[["portfolio_id","portfolio_name","tax_treatment","starting_value"]]
    per_pf["starting_value"] = pd.to_numeric(per_pf["starting_value"], errors="coerce").fillna(0.0)
    per_pf = per_pf[per_pf["starting_value"] > 0].reset_index(drop=True)
    return per_pf

# -----------------------------
# Flows → monthly inputs (Feature A)
# -----------------------------

def _flows_to_monthly_inputs(flows: pd.DataFrame) -> pd.DataFrame:
    if flows is None or flows.empty:
        return pd.DataFrame(columns=[
            "portfolio_id", "monthly_contribution", "monthly_withdrawal", "index_with_inflation"
        ])

    f = flows.copy()
    f["frequency"] = f["frequency"].str.lower().str.strip()
    f = f[f["frequency"].isin(["monthly", "annual"])]

    f["m_eq"] = f.apply(
        lambda r: float(r["amount"]) if r["frequency"] == "monthly" else float(r["amount"]) / 12.0,
        axis=1
    )

    contrib = (
        f[f["kind"].str.upper() == "CONTRIBUTION"]
        .groupby("portfolio_id")["m_eq"].sum()
        .rename("monthly_contribution")
    )
    withd = (
        f[f["kind"].str.upper() == "WITHDRAWAL"]
        .groupby("portfolio_id")["m_eq"].sum()
        .rename("monthly_withdrawal")
    )
    infl = (
        f.groupby("portfolio_id")["index_with_inflation"].max().rename("index_with_inflation")
    )

    out = pd.DataFrame({"portfolio_id": sorted(set(f["portfolio_id"].tolist()))})
    out = out.merge(contrib, on="portfolio_id", how="left").merge(withd, on="portfolio_id", how="left").merge(infl, on="portfolio_id", how="left")
    out = out.fillna({"monthly_contribution": 0.0, "monthly_withdrawal": 0.0, "index_with_inflation": 1})
    return out

# -----------------------------
# Core runner
# -----------------------------

def run_forecast(
    db_path: str = DB_PATH_DEFAULT,
    params: ForecastParams | Dict[str, Any] = ForecastParams(),
    write_to_db: bool = True,
    return_frames: bool = True,
) -> Dict[str, Any]:
    if isinstance(params, dict):
        fp = ForecastParams(**params)
    else:
        fp = params

    conn = _connect(db_path)
    try:
        # Use DB-driven global settings (avoid utils signature mismatches)
        settings = _load_global_settings(conn)

        # Apply per-run overrides if supplied
        for k in ("inflation_mode", "growth_mode", "fx_mode"):
            v = getattr(fp, k)
            if v:
                settings[k] = str(v).lower()

        # Seed assets (holdings or legacy Assets table)
        if fp.seed_from_holdings:
            assets_df = _holdings_to_assets(conn, valuation=fp.holdings_valuation)
        else:
            assets_df = load_assets(conn)

        inputs_df   = load_projection_inputs(conn)
        macro_df    = load_macro_forecast(conn) if fp.use_macro else pd.DataFrame()
        income_df   = load_employment_income(conn)

        # Portfolio flows → monthly inputs
        flows_df = load_portfolio_flows(conn)
        flows_inputs = _flows_to_monthly_inputs(flows_df)
        if not flows_inputs.empty:
            if inputs_df is None or inputs_df.empty:
                inputs_df = flows_inputs.copy()
            else:
                if "portfolio_id" in inputs_df.columns:
                    inputs_df = inputs_df.merge(flows_inputs, on="portfolio_id", how="left")
                else:
                    inputs_df = inputs_df.copy()
                    inputs_df["portfolio_id"] = None
                    inputs_df = inputs_df.merge(flows_inputs, on="portfolio_id", how="left")
            for c, v in [("monthly_contribution", 0.0), ("monthly_withdrawal", 0.0), ("index_with_inflation", 1)]:
                if c in inputs_df.columns:
                    inputs_df[c] = pd.to_numeric(inputs_df[c], errors="coerce").fillna(v)

        macro_rates = get_macro_rates(macro_df, settings, inputs_df)
        infl_factors = build_inflation_factors(macro_rates, settings, inputs_df)

        monthly_df = simulate_portfolio_growth(
            assets=assets_df,
            inputs=inputs_df,
            macro=macro_rates,
            inflation=infl_factors,
            settings=settings,
            years=fp.years,
            cadence=fp.cadence,
            start_date=fp.start_date,
        )

        annual_df = apply_taxes(
            monthly_df=monthly_df,
            employment_income=income_df,
            tax_rules_conn=conn,
            settings=settings,
            macro=macro_rates,
        )

        run_id = None
        if write_to_db:
            _ensure_forecast_run_tables(conn)
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
# Results persistence
# -----------------------------

def _ensure_forecast_run_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS forecast_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            params_json TEXT,
            settings_json TEXT
        );
        CREATE TABLE IF NOT EXISTS forecast_results_monthly (
            run_id INTEGER,
            period TEXT,
            portfolio_id INTEGER,
            portfolio_name TEXT,
            tax_treatment TEXT,
            nominal_value REAL,
            real_value REAL,
            contributions REAL,
            withdrawals REAL,
            PRIMARY KEY (run_id, period, portfolio_id)
        );
        CREATE TABLE IF NOT EXISTS forecast_results_annual (
            run_id INTEGER,
            year INTEGER,
            portfolio_id INTEGER,
            portfolio_name TEXT,
            tax_treatment TEXT,
            after_tax_income REAL,
            real_after_tax_income REAL,
            taxes_paid REAL,
            PRIMARY KEY (run_id, year, portfolio_id)
        );
        """
    )
    conn.commit()

def _insert_forecast_run(conn: sqlite3.Connection, fp: ForecastParams, settings: Dict[str, Any]) -> int:
    cur = conn.execute(
        "INSERT INTO forecast_runs (created_at, params_json, settings_json) VALUES (?, ?, ?)",
        (datetime.utcnow().isoformat(timespec="seconds") + "Z", fp.to_json(), json.dumps(settings)),
    )
    conn.commit()
    return int(cur.lastrowid)

def _write_results(
    conn: sqlite3.Connection,
    run_id: int,
    monthly_df: pd.DataFrame,
    annual_df: pd.DataFrame,
) -> None:
    m = monthly_df.copy()
    a = annual_df.copy()

    m = m.rename(columns={
        "date": "period",
        "portfolio": "portfolio_name",
        "value_nominal": "nominal_value",
        "value_real": "real_value",
    })
    for col in ["period","portfolio_id","portfolio_name","tax_treatment","nominal_value","real_value","contributions","withdrawals"]:
        if col not in m.columns:
            m[col] = None

    a = a.rename(columns={
        "year": "year",
        "portfolio": "portfolio_name",
        "after_tax": "after_tax_income",
        "after_tax_real": "real_after_tax_income",
        "taxes": "taxes_paid",
    })
    for col in ["year","portfolio_id","portfolio_name","tax_treatment","after_tax_income","real_after_tax_income","taxes_paid"]:
        if col not in a.columns:
            a[col] = None

    m["run_id"] = run_id
    a["run_id"] = run_id

    m_cols = [
        "run_id","period","portfolio_id","portfolio_name","tax_treatment",
        "nominal_value","real_value","contributions","withdrawals"
    ]
    a_cols = [
        "run_id","year","portfolio_id","portfolio_name","tax_treatment",
        "after_tax_income","real_after_tax_income","taxes_paid"
    ]

    m[m_cols].to_sql("forecast_results_monthly", conn, if_exists="append", index=False)
    a[a_cols].to_sql("forecast_results_annual",  conn, if_exists="append", index=False)
    conn.commit()
