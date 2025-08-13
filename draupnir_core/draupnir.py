# draupnir_core/draupnir.py
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd

# Robust imports: allow usage inside package or project root
try:
    from .tax_engine import calculate_annual_tax
except Exception:
    from tax_engine import calculate_annual_tax


# --------------------------------
# Public API expected by forecast_engine.py
#   get_macro_rates(macro_df, settings, inputs_df) -> pd.DataFrame[year,growth,inflation,fx]
#   build_inflation_factors(macro_rates, settings, inputs_df) -> pd.DataFrame[year,cpi_factor]
#   simulate_portfolio_growth(assets, inputs, macro, inflation, settings, years, cadence, start_date) -> monthly pd.DataFrame
#   apply_taxes(monthly_df, employment_income, tax_rules_conn, settings, macro) -> annual pd.DataFrame
# --------------------------------

DEFAULT_GROWTH = 0.05
DEFAULT_INFL = 0.02
DEFAULT_FX = 1.0


def _normalize_macro_df(macro_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Accepts flexible schemas and normalizes into columns: year, growth, inflation, fx
    Fills missing series with defaults, and ensures years start at 0,1,2,...
    """
    if macro_df is None or macro_df.empty:
        yrs = list(range(0, 120))
        return pd.DataFrame({
            "year": yrs,
            "growth": [DEFAULT_GROWTH] * len(yrs),
            "inflation": [DEFAULT_INFL] * len(yrs),
            "fx": [DEFAULT_FX] * len(yrs),
        })

    df = macro_df.copy()
    # Try common column names
    colmap = {}
    for c in df.columns:
        lc = c.strip().lower()
        if lc in ("yr", "year", "t"):
            colmap[c] = "year"
        elif lc in ("growth", "equities_growth", "return", "ret"):
            colmap[c] = "growth"
        elif lc in ("inflation", "inflation_rate", "cpi"):
            colmap[c] = "inflation"
        elif lc in ("fx", "fx_rate", "usd_cad", "fxratio"):
            colmap[c] = "fx"
    df = df.rename(columns=colmap)

    # Fill missing columns
    if "year" not in df.columns:
        df["year"] = list(range(len(df)))
    if "growth" not in df.columns:
        df["growth"] = DEFAULT_GROWTH
    if "inflation" not in df.columns:
        df["inflation"] = DEFAULT_INFL
    if "fx" not in df.columns:
        df["fx"] = DEFAULT_FX

    # Sanitize
    df = df[["year", "growth", "inflation", "fx"]].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    df["growth"] = pd.to_numeric(df["growth"], errors="coerce").fillna(DEFAULT_GROWTH)
    df["inflation"] = pd.to_numeric(df["inflation"], errors="coerce").fillna(DEFAULT_INFL)
    df["fx"] = pd.to_numeric(df["fx"], errors="coerce").fillna(DEFAULT_FX)

    # Ensure we have a long-enough horizon (pad to 120 yrs)
    last = int(df["year"].max())
    if last < 119:
        pad = pd.DataFrame({
            "year": list(range(last + 1, 120)),
            "growth": DEFAULT_GROWTH,
            "inflation": DEFAULT_INFL,
            "fx": DEFAULT_FX,
        })
        df = pd.concat([df, pad], ignore_index=True)

    # Rebase years so they start at 0
    miny = int(df["year"].min())
    if miny != 0:
        df["year"] = df["year"] - miny

    return df.sort_values("year").reset_index(drop=True)


def get_macro_rates(macro_df: pd.DataFrame | None, settings: Dict[str, Any], inputs_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Return a normalized macro path DataFrame with columns: year, growth, inflation, fx.
    'settings' may contain modes; we keep it simple and just return the path.
    """
    return _normalize_macro_df(macro_df)


def build_inflation_factors(macro_rates: pd.DataFrame, settings: Dict[str, Any], inputs_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Convert annual inflation to cumulative CPI factor by year (for real conversions).
    Returns DataFrame: [year, cpi_factor], where cpi_factor is cumulative CPI from year 0.
    """
    out = []
    cpi = 1.0
    for _, r in macro_rates.iterrows():
        i = float(r["inflation"])
        # Annual CPI multiplier
        cpi *= (1.0 + i)
        out.append({"year": int(r["year"]), "cpi_factor": cpi})
    return pd.DataFrame(out)


@dataclass
class _SimRow:
    date: pd.Timestamp
    portfolio_id: int
    portfolio_name: str
    tax_treatment: str
    value_nominal: float
    value_real: float
    contributions: float
    withdrawals: float


def _month_iter(start_date: Optional[str], months: int) -> list[pd.Timestamp]:
    """
    Build a list of month-start timestamps, length 'months', starting from start_date or today@month-start.
    """
    if start_date:
        try:
            dt0 = pd.to_datetime(start_date)  # expects 'YYYY-MM-01'
        except Exception:
            dt0 = pd.Timestamp.today().normalize().replace(day=1)
    else:
        dt0 = pd.Timestamp.today().normalize().replace(day=1)
    return [dt0 + pd.DateOffset(months=k) for k in range(months)]


def simulate_portfolio_growth(
    assets: pd.DataFrame,
    inputs: pd.DataFrame | None,
    macro: pd.DataFrame,
    inflation: pd.DataFrame,
    settings: Dict[str, Any],
    years: int = 30,
    cadence: str = "monthly",
    start_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Minimal, robust simulator:
      - One row per month per portfolio.
      - Apply monthly growth to value.
      - Add contributions (optionally indexed with inflation).
      - Subtract withdrawals (optionally indexed with inflation).
      - Compute real value using cumulative monthly inflation from macro path.

    Expects 'assets' with: [portfolio_id, portfolio_name, tax_treatment, starting_value]
    Expects 'inputs' with: [portfolio_id, monthly_contribution, monthly_withdrawal, index_with_inflation]
    Returns columns: [date, portfolio_id, portfolio_name, tax_treatment, value_nominal, value_real, contributions, withdrawals]
    """
    if assets is None or assets.empty:
        return pd.DataFrame(columns=[
            "date","portfolio_id","portfolio_name","tax_treatment",
            "value_nominal","value_real","contributions","withdrawals"
        ])

    months = int(years) * (12 if cadence == "monthly" else 1)
    dates = _month_iter(start_date, months)

    # Normalize inputs
    if inputs is None:
        inputs = pd.DataFrame(columns=["portfolio_id","monthly_contribution","monthly_withdrawal","index_with_inflation"])
    inputs = inputs.copy()
    for c, dv in (("monthly_contribution", 0.0), ("monthly_withdrawal", 0.0), ("index_with_inflation", 1)):
        if c not in inputs.columns:
            inputs[c] = dv
    inputs["monthly_contribution"] = pd.to_numeric(inputs["monthly_contribution"], errors="coerce").fillna(0.0)
    inputs["monthly_withdrawal"] = pd.to_numeric(inputs["monthly_withdrawal"], errors="coerce").fillna(0.0)
    inputs["index_with_inflation"] = inputs["index_with_inflation"].fillna(1).astype(int)

    # Join assets+inputs
    A = assets.copy()
    A["portfolio_id"] = pd.to_numeric(A["portfolio_id"], errors="coerce").astype("Int64")
    A = A.merge(inputs[["portfolio_id","monthly_contribution","monthly_withdrawal","index_with_inflation"]],
                on="portfolio_id", how="left").fillna({"monthly_contribution":0.0,"monthly_withdrawal":0.0,"index_with_inflation":1})

    # Macro helpers by year
    macro = macro.copy()
    macro.index = macro["year"].astype(int)
    infl_by_year = macro["inflation"].to_dict()
    grow_by_year = macro["growth"].to_dict()

    rows: list[_SimRow] = []

    for _, p in A.iterrows():
        pid = int(p["portfolio_id"])
        pname = str(p.get("portfolio_name", f"PID {pid}"))
        ttreat = str(p.get("tax_treatment", "TAXABLE") or "TAXABLE").upper()
        val = float(p.get("starting_value", 0.0))

        contrib0 = float(p.get("monthly_contribution", 0.0))
        withdr0 = float(p.get("monthly_withdrawal", 0.0))
        index_flag = int(p.get("index_with_inflation", 1))

        cpi = 1.0  # cumulative inflation since t0
        for m, dt in enumerate(dates, start=1):
            y = (m - 1) // 12  # year index
            g_y = float(grow_by_year.get(y, DEFAULT_GROWTH))
            i_y = float(infl_by_year.get(y, DEFAULT_INFL))

            r_m = (1.0 + g_y) ** (1.0 / 12.0) - 1.0
            i_m = (1.0 + i_y) ** (1.0 / 12.0) - 1.0

            # Index flows with inflation if requested
            if index_flag:
                contrib = contrib0 * ((1.0 + i_m) ** m)
                withdr = withdr0 * ((1.0 + i_m) ** m)
            else:
                contrib = contrib0
                withdr = withdr0

            # Apply growth, then add/sub flows
            val = val * (1.0 + r_m)
            val = val + contrib
            val = val - withdr

            # Real value via cumulative monthly inflation
            cpi *= (1.0 + i_m)
            real_val = val / cpi

            rows.append(_SimRow(
                date=dt, portfolio_id=pid, portfolio_name=pname, tax_treatment=ttreat,
                value_nominal=round(val, 2), value_real=round(real_val, 2),
                contributions=round(contrib, 2), withdrawals=round(withdr, 2),
            ))

    out = pd.DataFrame([r.__dict__ for r in rows])
    # Ensure column order
    return out[[
        "date","portfolio_id","portfolio_name","tax_treatment",
        "value_nominal","value_real","contributions","withdrawals"
    ]]


def apply_taxes(
    monthly_df: pd.DataFrame,
    employment_income: pd.DataFrame | Dict[str, float] | None,
    tax_rules_conn: sqlite3.Connection,
    settings: Dict[str, Any],
    macro: pd.DataFrame,
) -> pd.DataFrame:
    """
    Collapse monthly results to annual after-tax income by portfolio.
    For simplicity, 'portfolio income' is proxied by annual withdrawals.
    Employment income is read from EmploymentIncome (year->amount) if provided.
    """
    if monthly_df is None or monthly_df.empty:
        return pd.DataFrame(columns=[
            "year","portfolio_id","portfolio_name","tax_treatment",
            "after_tax_income","real_after_tax_income","taxes_paid"
        ])

    # Employment income map
    emp_map: Dict[int, float] = {}
    if isinstance(employment_income, dict):
        emp_map = {int(k): float(v) for k, v in employment_income.items()}
    elif employment_income is not None and not isinstance(employment_income, dict) and not employment_income.empty:
        # Expect columns: year, amount
        df = employment_income.copy()
        ycol = "year" if "year" in df.columns else df.columns[0]
        acol = "amount" if "amount" in df.columns else df.columns[1]
        df[ycol] = pd.to_numeric(df[ycol], errors="coerce").astype("Int64")
        df[acol] = pd.to_numeric(df[acol], errors="coerce").fillna(0.0)
        emp_map = {int(r[ycol]): float(r[acol]) for _, r in df.iterrows() if pd.notna(r[ycol])}

    # Build annual CPI factors for "real" conversion
    macro_norm = _normalize_macro_df(macro)
    cpi_factors = {int(r["year"]): float((1.0 + r["inflation"])) for _, r in macro_norm.iterrows()}
    # Precompute cumulative CPI up to each year-end
    cum_cpi: Dict[int, float] = {}
    acc = 1.0
    for y in range(0, 200):
        acc *= cpi_factors.get(y, 1.0 + DEFAULT_INFL)
        cum_cpi[y] = acc

    monthly_df = monthly_df.copy()
    monthly_df["year"] = monthly_df["date"].dt.year  # absolute year
    # Rebase to year offsets (0,1,2,...) relative to start year
    y0 = int(monthly_df["year"].min())
    monthly_df["y_idx"] = monthly_df["year"] - y0

    # Aggregate withdrawals as 'portfolio pre-tax income'
    agg = (monthly_df
           .groupby(["y_idx","portfolio_id","portfolio_name","tax_treatment"], dropna=False)["withdrawals"]
           .sum()
           .reset_index()
           .rename(columns={"withdrawals":"portfolio_income_pre"}))

    records = []
    for _, r in agg.iterrows():
        y = int(r["y_idx"])
        pid = int(r["portfolio_id"])
        pname = str(r["portfolio_name"])
        ttreat = str(r["tax_treatment"] or "TAXABLE").upper()

        port_income_nom = float(r["portfolio_income_pre"])
        emp_nom = float(emp_map.get(y0 + y, 0.0))

        tax_port = calculate_annual_tax(tax_rules_conn, port_income_nom, ttreat, year=(y0 + y))
        tax_emp = calculate_annual_tax(tax_rules_conn, emp_nom, "EMPLOYMENT", year=(y0 + y))
        taxes = round(tax_port + tax_emp, 2)

        after_tax_nom = round(port_income_nom + emp_nom - taxes, 2)
        disc = float(cum_cpi.get(y, 1.0))
        after_tax_real = round(after_tax_nom / disc, 2)

        records.append({
            "year": (y0 + y),
            "portfolio_id": pid,
            "portfolio_name": pname,
            "tax_treatment": ttreat,
            "after_tax_income": after_tax_nom,
            "real_after_tax_income": after_tax_real,
            "taxes_paid": taxes,
        })

    out = pd.DataFrame(records)
    return out[[
        "year","portfolio_id","portfolio_name","tax_treatment",
        "after_tax_income","real_after_tax_income","taxes_paid"
    ]]




