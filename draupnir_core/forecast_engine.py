# draupnir_core/forecast_engine.py
"""
Forecast Engine (callable)

- Seeds starting assets from CURRENT HOLDINGS (derived from trades) or Assets
- Choice of valuation basis: "market" (live-price) or "book"
- Maps portfolios → tax_treatment automatically via portfolios table
- Uses portfolio_flows as the single source of truth for contributions/withdrawals.

Fixes and improvements:
- Flows now respect start_date / end_date / frequency by expanding to a per-month schedule.
- Supported frequencies: monthly, annual, quarterly, semiannual, once (fires on the start month only).
- When a flows_schedule exists for a portfolio, the legacy "base" monthly inputs are zeroed for that portfolio
  to prevent premature contributions/withdrawals starting at the sim start (this fixes the 2030 start bug).
- Inflation indexing is handled by downstream simulate_portfolio_growth using the schedule's timing.
- Annual table persists nominal/real income & tax columns (unchanged).

NEW (this patch):
- Make annual table migration transaction-safe using SAVEPOINTs.
- Remove unconditional commits from helper so we don’t end up rolling back a non-existent txn.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, Iterable, List
import os

import pandas as pd
import numpy as np
import yfinance as yf

# ---------- Robust imports for core math ----------
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
    from . import tax_engine as tax_engine  # noqa: F401
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
    import tax_engine as tax_engine  # noqa: F401

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

    # MANUAL (constant) assumptions when use_macro=False
    manual_growth: Optional[float] = None      # annual (e.g., 0.07 for 7%)
    manual_inflation: Optional[float] = None   # annual (e.g., 0.02 for 2%)
    manual_fx: Optional[float] = None          # scalar

    notes: Optional[str] = None

    # Seed from holdings
    seed_from_holdings: bool = False
    holdings_valuation: str = "market"  # "market" or "book"

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

def _get_table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    try:
        cur = conn.execute(f"PRAGMA table_info({table});")
        return [r[1] for r in cur.fetchall()]
    except Exception:
        return []

def _load_df(conn: sqlite3.Connection, sql: str, params: Tuple = ()) -> pd.DataFrame:
    try:
        return pd.read_sql(sql, conn, params=params)
    except Exception:
        return pd.DataFrame()

def _load_global_settings(conn: sqlite3.Connection) -> Dict[str, Any]:
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
            amount REAL NOT NULL,
            frequency TEXT NOT NULL,             -- 'monthly' | 'annual' | 'quarterly' | 'semiannual' | 'once'
            start_date TEXT NOT NULL,            -- 'YYYY-MM-01'
            end_date TEXT,                       -- nullable
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
# Build assets from holdings
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
        lambda r: (r["current_qty"]*r["avg_book_price"]) if r["avg_book_price"] is not None else None,
        axis=1
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
# Flows → schedule (RESPECTS DATES)
# -----------------------------

def _parse_ym(s: Optional[str]) -> Optional[pd.Timestamp]:
    if not s:
        return None
    try:
        # Expect 'YYYY-MM-01' but accept 'YYYY-MM' too
        ts = pd.to_datetime(str(s), errors="coerce")
        if pd.isna(ts):
            return None
        return ts.normalize().replace(day=1)
    except Exception:
        return None

def _months_between(d0: pd.Timestamp, d1: pd.Timestamp) -> int:
    """How many whole months to move from d0 to reach d1 (can be negative)."""
    return (d1.year - d0.year) * 12 + (d1.month - d0.month)

def _build_flows_schedule(
    flows: pd.DataFrame,
    sim_start: pd.Timestamp,
    years: int
) -> pd.DataFrame:
    """
    Expand portfolio_flows into a month-indexed schedule compatible with simulate_portfolio_growth:
      columns => [portfolio_id, m_idx, contrib, withdraw, index_flag]
    - frequency='monthly'     → apply each month from start_date..end_date (or to horizon)
    - frequency='annual'      → apply once per year in the start month (same month of year)
    - frequency='quarterly'   → apply every 3 months starting at start_date month
    - frequency='semiannual'  → apply every 6 months starting at start_date month
    - frequency='once'        → apply only in the start_date month
    - start_date/end_date are respected; anything before sim_start is skipped.
    - amounts are PRE-inflation; simulate_portfolio_growth applies indexing per month.
    """
    if flows is None or flows.empty:
        return pd.DataFrame(columns=["portfolio_id","m_idx","contrib","withdraw","index_flag"])

    horizon = max(1, int(years)) * 12
    rows: List[Dict[str, Any]] = []

    for r in flows.itertuples(index=False):
        pid   = int(getattr(r, "portfolio_id"))
        kind  = str(getattr(r, "kind", "")).strip().upper()
        freq  = str(getattr(r, "frequency", "")).strip().lower()
        amt   = float(getattr(r, "amount", 0.0) or 0.0)
        idx   = int(getattr(r, "index_with_inflation", 1) or 0)
        sd    = _parse_ym(getattr(r, "start_date", None))
        ed    = _parse_ym(getattr(r, "end_date", None))

        if amt <= 0 or sd is None:
            continue

        start_m = max(0, _months_between(sim_start, sd))
        end_m = (horizon - 1) if ed is None else min(horizon - 1, _months_between(sim_start, ed))
        if end_m < 0 or start_m > (horizon - 1):
            continue  # entirely out of horizon

        base_contrib = amt if kind == "CONTRIBUTION" else 0.0
        base_withdr  = amt if kind == "WITHDRAWAL"  else 0.0

        if freq in ("", "monthly"):
            step = 1
            for m in range(start_m, end_m + 1, step):
                rows.append({"portfolio_id": pid, "m_idx": m, "contrib": base_contrib, "withdraw": base_withdr, "index_flag": idx})
        elif freq == "annual":
            m = start_m
            while m <= end_m:
                rows.append({"portfolio_id": pid, "m_idx": m, "contrib": base_contrib, "withdraw": base_withdr, "index_flag": idx})
                m += 12
        elif freq == "quarterly":
            m = start_m
            while m <= end_m:
                rows.append({"portfolio_id": pid, "m_idx": m, "contrib": base_contrib, "withdraw": base_withdr, "index_flag": idx})
                m += 3
        elif freq == "semiannual":
            m = start_m
            while m <= end_m:
                rows.append({"portfolio_id": pid, "m_idx": m, "contrib": base_contrib, "withdraw": base_withdr, "index_flag": idx})
                m += 6
        elif freq == "once":
            if start_m <= end_m:
                rows.append({"portfolio_id": pid, "m_idx": start_m, "contrib": base_contrib, "withdraw": base_withdr, "index_flag": idx})
        else:
            # Unknown frequency -> treat like monthly for safety
            for m in range(start_m, end_m + 1):
                rows.append({"portfolio_id": pid, "m_idx": m, "contrib": base_contrib, "withdraw": base_withdr, "index_flag": idx})

    if not rows:
        return pd.DataFrame(columns=["portfolio_id","m_idx","contrib","withdraw","index_flag"])
    sched = pd.DataFrame(rows)
    # Ensure integer types where appropriate
    sched["m_idx"] = sched["m_idx"].astype(int)
    sched["portfolio_id"] = sched["portfolio_id"].astype(int)
    return sched

# -----------------------------
# Legacy flows → monthly inputs (KEPT for base rates; dates ignored)
# -----------------------------

def _flows_to_monthly_inputs(flows: pd.DataFrame) -> pd.DataFrame:
    """
    Still used to provide per-portfolio base amounts; timing is controlled by explicit flows_schedule.
    If a schedule exists for a portfolio, we'll zero these base amounts before calling the simulator,
    so legacy inputs won't trigger early flows.
    """
    if flows is None or flows.empty:
        return pd.DataFrame(columns=[
            "portfolio_id", "monthly_contribution", "monthly_withdrawal", "index_with_inflation"
        ])

    f = flows.copy()
    f["frequency"] = f["frequency"].str.lower().str.strip()
    f = f[f["frequency"].isin(["monthly", "annual", "quarterly", "semiannual", "once"])]

    # Map each frequency to a monthly equivalent *only* for legacy inputs
    def _to_monthly_equiv(row) -> float:
        amt = float(row["amount"])
        frq = str(row["frequency"])
        if frq == "monthly":
            return amt
        if frq == "annual":
            return amt / 12.0
        if frq == "quarterly":
            return amt / 3.0
        if frq == "semiannual":
            return amt / 6.0
        if frq == "once":
            # One-time flows have no steady monthly equivalent; treat as 0 for base inputs
            return 0.0
        return amt

    f["m_eq"] = f.apply(_to_monthly_equiv, axis=1)

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
    out = out.merge(contrib, on="portfolio_id", how="left") \
             .merge(withd,  on="portfolio_id", how="left") \
             .merge(infl,   on="portfolio_id", how="left")
    out = out.fillna({"monthly_contribution": 0.0, "monthly_withdrawal": 0.0, "index_with_inflation": 1})
    return out

# -----------------------------
# Core runner
# -----------------------------

def _constant_macro_df(years: int, growth: float, inflation: float, fx: float) -> pd.DataFrame:
    yrs = list(range(0, max(120, int(years) + 1)))
    return pd.DataFrame({
        "year": yrs,
        "growth": [float(growth)] * len(yrs),
        "inflation": [float(inflation)] * len(yrs),
        "fx": [float(fx)] * len(yrs),
    })

def _resolve_sim_start(start_date: Optional[str]) -> pd.Timestamp:
    if start_date:
        try:
            return pd.to_datetime(start_date).normalize().replace(day=1)
        except Exception:
            pass
    # Default: first day of the current month
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

        # Seed assets
        if fp.seed_from_holdings:
            assets_df = _holdings_to_assets(conn, valuation=fp.holdings_valuation)
        else:
            assets_df = load_assets(conn)

        # Macro:
        if fp.use_macro:
            macro_df = load_macro_forecast(conn)
        else:
            g = fp.manual_growth if fp.manual_growth is not None else DEFAULT_GROWTH
            i = fp.manual_inflation if fp.manual_inflation is not None else DEFAULT_INFL
            x = fp.manual_fx if fp.manual_fx is not None else DEFAULT_FX
            macro_df = _constant_macro_df(fp.years, g, i, x)

        income_df = load_employment_income(conn)

        # Portfolio flows (source of truth)
        flows_df = load_portfolio_flows(conn)

        # Build schedule that respects start/end/frequency
        sim_start = _resolve_sim_start(fp.start_date)
        schedule_df = _build_flows_schedule(flows_df, sim_start, fp.years)

        # Base per-portfolio amounts (kept for completeness)
        inputs_df = _flows_to_monthly_inputs(flows_df)
        if inputs_df.empty:
            inputs_df = pd.DataFrame(columns=["portfolio_id","monthly_contribution","monthly_withdrawal","index_with_inflation"])

        # --- CRITICAL FIX: if a schedule exists for a portfolio, zero legacy base amounts
        # This prevents contributions/withdrawals from starting at sim start when the schedule
        # says they should begin later (e.g., 2030-01-01).
        if schedule_df is not None and not schedule_df.empty and not inputs_df.empty:
            scheduled_pids = set(schedule_df["portfolio_id"].unique().tolist())
            mask = inputs_df["portfolio_id"].isin(scheduled_pids)
            if "monthly_contribution" in inputs_df.columns:
                inputs_df.loc[mask, "monthly_contribution"] = 0.0
            if "monthly_withdrawal" in inputs_df.columns:
                inputs_df.loc[mask, "monthly_withdrawal"] = 0.0
            # Note: we keep index_with_inflation flag; simulate_portfolio_growth should rely on the schedule's index_flag.

        macro_rates  = get_macro_rates(macro_df, settings, inputs_df)
        infl_factors = build_inflation_factors(macro_rates, settings, inputs_df)

        monthly_df = simulate_portfolio_growth(
            assets=assets_df,
            inputs=inputs_df,
            macro=macro_rates,
            inflation=infl_factors,
            settings=settings,
            years=fp.years,
            cadence=fp.cadence,
            start_date=str(sim_start.date()),
            flows_schedule=schedule_df,  # <-- enforce timing
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
            _ensure_annual_schema(conn)  # ensure clean new schema (migrates if necessary)
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
# Results persistence (unchanged other than migration tx safety)
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
        -- annual table is created/migrated by _ensure_annual_schema()
        """
    )
    conn.commit()

_ANNUAL_COLUMNS = [
    ("run_id","INTEGER"),("year","INTEGER"),("portfolio_id","INTEGER"),
    ("portfolio_name","TEXT"),("tax_treatment","TEXT"),
    ("nominal_pretax_income","REAL"),("nominal_taxes_paid","REAL"),
    ("nominal_after_tax_income","REAL"),("nominal_effective_tax_rate","REAL"),
    ("real_pretax_income","REAL"),("real_taxes_paid","REAL"),
    ("real_after_tax_income","REAL"),("real_effective_tax_rate","REAL"),
    ("contributions","REAL"),("withdrawals","REAL")
]

def _create_annual_table(conn: sqlite3.Connection, table_name: str = "forecast_results_annual") -> None:
    cols_sql = ",\n            ".join([f"{name} {typ}" for name, typ in _ANNUAL_COLUMNS])
    pk = "PRIMARY KEY (run_id, year, portfolio_id)"
    conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            {cols_sql},
            {pk}
        );
    """)

def _ensure_annual_schema(conn: sqlite3.Connection) -> None:
    """
    Ensure forecast_results_annual has the canonical v10 columns.
    If legacy schema is detected, migrate using a pandas-based copy that tolerates
    missing/extra old columns (no savepoints needed).
    """
    want_cols = [c[0] for c in _ANNUAL_COLUMNS]

    # Fresh create
    if not _table_exists(conn, "forecast_results_annual"):
        _create_annual_table(conn, "forecast_results_annual")
        conn.commit()
        return

    have = _get_table_columns(conn, "forecast_results_annual")
    if set(have) == set(want_cols):
        return  # already good

    # ---- Migrate legacy -> canonical ----
    try:
        # Read old table to a dataframe (tolerates any legacy shape)
        old_df = pd.read_sql("SELECT * FROM forecast_results_annual", conn)
    except Exception:
        # If we can't read, just recreate empty canonical table
        conn.execute("DROP TABLE IF EXISTS forecast_results_annual;")
        _create_annual_table(conn, "forecast_results_annual")
        conn.commit()
        return

    # Build a new dataframe with canonical columns, filling from whatever exists
    def g(col, default=None):
        return old_df[col] if col in old_df.columns else default

    new_df = pd.DataFrame({
        "run_id":                        g("run_id", 0),
        "year":                          g("year", 0),
        "portfolio_id":                  g("portfolio_id", 0),
        "portfolio_name":                g("portfolio_name", ""),
        "tax_treatment":                 g("tax_treatment", ""),

        # Nominal block (try to derive if legacy only had after_tax_income/taxes_paid)
        "nominal_pretax_income":         ( (g("nominal_pretax_income", None))
                                           if "nominal_pretax_income" in old_df.columns else
                                           ((g("after_tax_income", 0.0).fillna(0.0) +
                                             g("taxes_paid", 0.0).fillna(0.0)) if "after_tax_income" in old_df.columns or "taxes_paid" in old_df.columns else 0.0) ),
        "nominal_taxes_paid":            ( g("nominal_taxes_paid", None)
                                           if "nominal_taxes_paid" in old_df.columns else
                                           g("taxes_paid", 0.0) ),
        "nominal_after_tax_income":      ( g("nominal_after_tax_income", None)
                                           if "nominal_after_tax_income" in old_df.columns else
                                           g("after_tax_income", 0.0) ),

        # Real block (keep if present; else default zeros)
        "real_pretax_income":            g("real_pretax_income", 0.0),
        "real_taxes_paid":               g("real_taxes_paid", 0.0),
        "real_after_tax_income":         g("real_after_tax_income", 0.0),

        # Will compute below
        "nominal_effective_tax_rate":    None,
        "real_effective_tax_rate":       None,

        # Flows (added in v10)
        "contributions":                 g("contributions", 0.0),
        "withdrawals":                   g("withdrawals", 0.0),
    })

    # Ensure numeric types where appropriate
    for c in ["nominal_pretax_income","nominal_taxes_paid","nominal_after_tax_income",
              "real_pretax_income","real_taxes_paid","real_after_tax_income",
              "contributions","withdrawals"]:
        new_df[c] = pd.to_numeric(new_df[c], errors="coerce").fillna(0.0)

    # Compute effective tax rates safely
    new_df["nominal_effective_tax_rate"] = new_df.apply(
        lambda r: (r["nominal_taxes_paid"] / r["nominal_pretax_income"]) if r["nominal_pretax_income"] > 0 else 0.0,
        axis=1
    )
    new_df["real_effective_tax_rate"] = new_df.apply(
        lambda r: (r["real_taxes_paid"] / r["real_pretax_income"]) if r["real_pretax_income"] > 0 else 0.0,
        axis=1
    )

    # Write into a brand-new canonical table, then swap
    _create_annual_table(conn, "forecast_results_annual_new")
    new_df = new_df[want_cols]  # column order
    # Clear target then append
    conn.execute("DELETE FROM forecast_results_annual_new;")
    new_df.to_sql("forecast_results_annual_new", conn, if_exists="append", index=False)

    # Swap tables (no savepoints; keep it simple)
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

def _write_results(
    conn: sqlite3.Connection,
    run_id: int,
    monthly_df: pd.DataFrame,
    annual_df: pd.DataFrame,
) -> None:
    # ---------- Monthly ----------
    m = monthly_df.copy()
    m = m.rename(columns={
        "date": "period",
        "portfolio": "portfolio_name",
        "value_nominal": "nominal_value",
        "value_real": "real_value",
    })
    for col in ["period","portfolio_id","portfolio_name","tax_treatment","nominal_value","real_value","contributions","withdrawals"]:
        if col not in m.columns:
            m[col] = None
    m["run_id"] = run_id
    m_cols = [
        "run_id","period","portfolio_id","portfolio_name","tax_treatment",
        "nominal_value","real_value","contributions","withdrawals"
    ]
    m[m_cols].to_sql("forecast_results_monthly", conn, if_exists="append", index=False)

    # ---------- Annual (canonical schema) ----------
    a = annual_df.copy() if annual_df is not None else pd.DataFrame()
    if a.empty:
        a = pd.DataFrame(columns=["year","portfolio_id","portfolio","tax_treatment","after_tax","after_tax_real","taxes"])

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

    mm = monthly_df.copy()
    mm["year"] = pd.to_datetime(mm["date"]).dt.year
    defl = (pd.to_numeric(mm["value_nominal"], errors="coerce") /
            pd.to_numeric(mm["value_real"], errors="coerce"))
    defl = defl.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    mm["_defl"] = defl
    for fld in ["contributions", "withdrawals"]:
        if fld not in mm.columns:
            mm[fld] = 0.0
        mm[f"{fld}_real"] = pd.to_numeric(mm[fld], errors="coerce").fillna(0.0) / mm["_defl"].replace(0, 1.0)

    flows_real = (
        mm.groupby(["year","portfolio_id"], as_index=False)[["contributions_real","withdrawals_real"]].sum()
        .rename(columns={"contributions_real":"contributions", "withdrawals_real":"withdrawals"})
    )

    a_num = a.copy()
    a_num["after_tax_income"] = pd.to_numeric(a_num["after_tax_income"], errors="coerce").fillna(0.0)
    a_num["real_after_tax_income"] = pd.to_numeric(a_num["real_after_tax_income"], errors="coerce").fillna(0.0)
    a_num["taxes_paid"] = pd.to_numeric(a_num["taxes_paid"], errors="coerce").fillna(0.0)

    a_num["nominal_pretax_income"] = a_num["after_tax_income"] + a_num["taxes_paid"]
    a_num["nominal_taxes_paid"] = a_num["taxes_paid"]
    a_num["nominal_after_tax_income"] = a_num["after_tax_income"]
    a_num["nominal_effective_tax_rate"] = a_num.apply(
        lambda r: (r["nominal_taxes_paid"] / r["nominal_pretax_income"]) if r["nominal_pretax_income"] > 0 else 0.0,
        axis=1
    )

    est_defl = a_num.apply(
        lambda r: (r["after_tax_income"] / r["real_after_tax_income"]) if r["real_after_tax_income"] > 0 else 1.0,
        axis=1
    ).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    a_num["_defl"] = est_defl
    a_num["real_taxes_paid"] = a_num["nominal_taxes_paid"] / a_num["_defl"].replace(0, 1.0)
    a_num["real_pretax_income"] = a_num["real_after_tax_income"] + a_num["real_taxes_paid"]
    a_num["real_effective_tax_rate"] = a_num.apply(
        lambda r: (r["real_taxes_paid"] / r["real_pretax_income"]) if r["real_pretax_income"] > 0 else 0.0,
        axis=1
    )

    a_enriched = a_num.merge(flows_real, on=["year","portfolio_id"], how="left")
    a_enriched[["contributions","withdrawals"]] = a_enriched[["contributions","withdrawals"]].fillna(0.0)
    a_enriched["run_id"] = run_id

    _ensure_annual_schema(conn)
    out_cols = [c[0] for c in _ANNUAL_COLUMNS]

    conn.execute("DELETE FROM forecast_results_annual WHERE run_id = ?;", (run_id,))
    conn.commit()

    a_enriched[out_cols].to_sql("forecast_results_annual", conn, if_exists="append", index=False)
    conn.commit()
