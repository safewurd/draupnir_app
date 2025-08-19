# draupnir_core/draupnir.py
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Optional, Dict, Any

import pandas as pd

# Robust import whether running as a package or from project root
try:
    from .tax_engine import calculate_annual_tax
except Exception:
    from tax_engine import calculate_annual_tax


# -----------------------------
# Defaults
# -----------------------------

DEFAULT_GROWTH = 0.05      # 5% nominal asset growth
DEFAULT_INFL = 0.02        # 2% inflation
DEFAULT_FX = 1.0


# -----------------------------
# Macro helpers
# -----------------------------

def _normalize_macro_df(macro_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Normalize into columns: [year, growth, inflation, fx] with year index 0..N.
    Pads to at least 120 years.
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

    if "year" not in df.columns:      df["year"] = list(range(len(df)))
    if "growth" not in df.columns:    df["growth"] = DEFAULT_GROWTH
    if "inflation" not in df.columns: df["inflation"] = DEFAULT_INFL
    if "fx" not in df.columns:        df["fx"] = DEFAULT_FX

    df = df[["year", "growth", "inflation", "fx"]].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    df["growth"] = pd.to_numeric(df["growth"], errors="coerce").fillna(DEFAULT_GROWTH)
    df["inflation"] = pd.to_numeric(df["inflation"], errors="coerce").fillna(DEFAULT_INFL)
    df["fx"] = pd.to_numeric(df["fx"], errors="coerce").fillna(DEFAULT_FX)

    last = int(df["year"].max())
    if last < 119:
        pad = pd.DataFrame({
            "year": list(range(last + 1, 120)),
            "growth": DEFAULT_GROWTH,
            "inflation": DEFAULT_INFL,
            "fx": DEFAULT_FX,
        })
        df = pd.concat([df, pad], ignore_index=True)

    miny = int(df["year"].min())
    if miny != 0:
        df["year"] = df["year"] - miny

    return df.sort_values("year").reset_index(drop=True)


def get_macro_rates(macro_df: pd.DataFrame | None, settings: Dict[str, Any], inputs_df: pd.DataFrame | None) -> pd.DataFrame:
    return _normalize_macro_df(macro_df)


def build_inflation_factors(macro_rates: pd.DataFrame, settings: Dict[str, Any], inputs_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Cumulative CPI by year (1.0 grows forward each year by (1+inflation)).
    """
    out = []
    cpi = 1.0
    for _, r in macro_rates.iterrows():
        i = float(r["inflation"])
        cpi *= (1.0 + i)
        out.append({"year": int(r["year"]), "cpi_factor": cpi})
    return pd.DataFrame(out)


# -----------------------------
# Simulation (monthly) with explicit t0 row
# -----------------------------

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
    if start_date:
        try:
            dt0 = pd.to_datetime(start_date)  # 'YYYY-MM-01'
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
    flows_schedule: pd.DataFrame | None = None,  # optional override schedule per month
) -> pd.DataFrame:
    """
    Emits one **t0** row per portfolio (beginning-of-period) then monthly end-of-period rows t1..tN.
    Columns: date, portfolio_id, portfolio_name, tax_treatment, value_nominal, value_real, contributions, withdrawals
    """
    if assets is None or assets.empty:
        return pd.DataFrame(columns=[
            "date","portfolio_id","portfolio_name","tax_treatment",
            "value_nominal","value_real","contributions","withdrawals"
        ])

    months = int(years) * (12 if cadence == "monthly" else 1)
    # We want a t0 + N months → generate N+1 dates, where dates[0] is the t0 timestamp
    dates = _month_iter(start_date, months + 1)

    # Normalize inputs
    if inputs is None or inputs.empty:
        inputs = pd.DataFrame(columns=["portfolio_id","monthly_contribution","monthly_withdrawal","index_with_inflation"])
    inputs = inputs.copy()
    for c, dv in (("monthly_contribution", 0.0), ("monthly_withdrawal", 0.0), ("index_with_inflation", 1)):
        if c not in inputs.columns:
            inputs[c] = dv
    inputs["monthly_contribution"]  = pd.to_numeric(inputs["monthly_contribution"], errors="coerce").fillna(0.0)
    inputs["monthly_withdrawal"]    = pd.to_numeric(inputs["monthly_withdrawal"], errors="coerce").fillna(0.0)
    inputs["index_with_inflation"]  = inputs["index_with_inflation"].fillna(1).astype(int)

    # Join assets + inputs
    A = assets.copy()
    A["portfolio_id"] = pd.to_numeric(A["portfolio_id"], errors="coerce").astype("Int64")
    A = A.merge(
        inputs[["portfolio_id","monthly_contribution","monthly_withdrawal","index_with_inflation"]],
        on="portfolio_id", how="left"
    ).fillna({"monthly_contribution":0.0,"monthly_withdrawal":0.0,"index_with_inflation":1})

    # Macro by year
    macro = macro.copy()
    macro.index = macro["year"].astype(int)
    infl_by_year = macro["inflation"].to_dict()
    grow_by_year = macro["growth"].to_dict()

    # Optional flows schedule map: (pid, m_idx) -> (contrib, withdraw, index_flag)
    sched_map: Dict[tuple, tuple] = {}
    if flows_schedule is not None and not flows_schedule.empty:
        fs = flows_schedule.copy()
        fs["m_idx"] = pd.to_numeric(fs["m_idx"], errors="coerce").fillna(-1).astype(int)
        fs = fs[(fs["m_idx"] >= 0) & (fs["m_idx"] < months)]
        for _, r in fs.iterrows():
            key = (int(r["portfolio_id"]), int(r["m_idx"]))
            prev = sched_map.get(key, (0.0, 0.0, int(r.get("index_flag", 1))))
            sched_map[key] = (
                float(prev[0]) + float(r.get("contrib", 0.0)),
                float(prev[1]) + float(r.get("withdraw", 0.0)),
                int(r.get("index_flag", 1))
            )

    rows: list[_SimRow] = []

    for _, p in A.iterrows():
        pid   = int(p["portfolio_id"])
        pname = str(p.get("portfolio_name", f"PID {pid}"))
        tt    = str(p.get("tax_treatment", "TAXABLE") or "TAXABLE").upper()
        val   = float(p.get("starting_value", 0.0))

        base_contrib = float(p.get("monthly_contribution", 0.0))
        base_withdr  = float(p.get("monthly_withdrawal", 0.0))
        base_index   = int(p.get("index_with_inflation", 1))

        # --- t0 (beginning-of-period) -> real == nominal, zero flows ---
        rows.append(_SimRow(
            date=dates[0],
            portfolio_id=pid,
            portfolio_name=pname,
            tax_treatment=tt,
            value_nominal=round(val, 2),
            value_real=round(val, 2),
            contributions=0.0,
            withdrawals=0.0,
        ))

        # CPI accumulator for real conversions (monthly)
        cpi_cum = 1.0

        # --- Months: t1..tN (end-of-period values) ---
        for m_idx in range(months):
            dt = dates[m_idx + 1]
            y  = m_idx // 12
            g_y = float(grow_by_year.get(y, DEFAULT_GROWTH))
            i_y = float(infl_by_year.get(y, DEFAULT_INFL))

            # Monthly factors
            r_m = (1.0 + g_y) ** (1.0 / 12.0) - 1.0
            i_m = (1.0 + i_y) ** (1.0 / 12.0) - 1.0

            # Flows this month (schedule overrides inputs)
            sch = sched_map.get((pid, m_idx))
            if sch is not None:
                s_contrib, s_withdr, s_idx_flag = sch
                idx_factor = (1.0 + i_y) ** y if int(s_idx_flag) else 1.0
                contrib = s_contrib * idx_factor
                withdr  = s_withdr  * idx_factor
            else:
                idx_factor = (1.0 + i_y) ** y if int(base_index) else 1.0
                contrib = base_contrib * idx_factor
                withdr  = base_withdr  * idx_factor

            # Apply growth then flows to move from t(m) → t(m+1)
            val = val * (1.0 + r_m)
            val = val + contrib - withdr

            # Real value via cumulative monthly inflation
            cpi_cum *= (1.0 + i_m)
            real_val = val / cpi_cum

            rows.append(_SimRow(
                date=dt,
                portfolio_id=pid,
                portfolio_name=pname,
                tax_treatment=tt,
                value_nominal=round(val, 2),
                value_real=round(real_val, 2),
                contributions=round(contrib, 2),
                withdrawals=round(withdr, 2),
            ))

    out = pd.DataFrame([r.__dict__ for r in rows])
    return out[[
        "date","portfolio_id","portfolio_name","tax_treatment",
        "value_nominal","value_real","contributions","withdrawals"
    ]]


# -----------------------------
# Annual after-tax aggregation
# -----------------------------

def apply_taxes(
    monthly_df: pd.DataFrame,
    employment_income: pd.DataFrame | Dict[str, float] | None,
    tax_rules_conn: sqlite3.Connection,
    settings: Dict[str, Any],
    macro: pd.DataFrame,
) -> pd.DataFrame:
    """
    Annual, per-portfolio:
      pretax income = withdrawals + (non‑reinvested distributions) + employment
      taxes = sum of taxes on each component (withdrawals by tax_treatment,
              interest, eligible div, non-eligible div, employment)
    Returns:
      [year, portfolio_id, portfolio_name, tax_treatment,
       after_tax_income, real_after_tax_income, taxes_paid]
    """
    if monthly_df is None or monthly_df.empty:
        return pd.DataFrame(columns=[
            "year","portfolio_id","portfolio_name","tax_treatment",
            "after_tax_income","real_after_tax_income","taxes_paid"
        ])

    # Employment income map (calendar year -> amount)
    emp_map: Dict[int, float] = {}
    if isinstance(employment_income, dict):
        emp_map = {int(k): float(v) for k, v in employment_income.items()}
    elif isinstance(employment_income, pd.DataFrame) and not employment_income.empty:
        df_emp = employment_income.copy()
        ycol = "year" if "year" in df_emp.columns else df_emp.columns[0]
        acol = "amount" if "amount" in df_emp.columns else df_emp.columns[1]
        df_emp[ycol] = pd.to_numeric(df_emp[ycol], errors="coerce").astype("Int64")
        df_emp[acol] = pd.to_numeric(df_emp[acol], errors="coerce").fillna(0.0)
        emp_map = {int(r[ycol]): float(r[acol]) for _, r in df_emp.iterrows() if pd.notna(r[ycol])}

    # CPI for deflation (annual)
    macro_norm = _normalize_macro_df(macro)
    cpi_step = {int(r["year"]): float(1.0 + r["inflation"]) for _, r in macro_norm.iterrows()}
    cum_cpi: Dict[int, float] = {}
    acc = 1.0
    for y in range(0, 200):
        acc *= cpi_step.get(y, 1.0 + DEFAULT_INFL)
        cum_cpi[y] = acc

    m = monthly_df.copy()
    m["year"] = pd.to_datetime(m["date"]).dt.year
    y0 = int(m["year"].min())
    m["y_idx"] = m["year"] - y0

    # Ensure telemetry columns exist
    for col in [
        "contributions","withdrawals",
        "interest_income_base","interest_reinvested_base",
        "eligible_dividend_income_base","noneligible_dividend_income_base",
        "eligible_dividends_reinvested_base","noneligible_dividends_reinvested_base",
    ]:
        if col not in m.columns:
            m[col] = 0.0

    # Annual per-portfolio aggregation
    grp = (m.groupby(["y_idx","portfolio_id","portfolio_name","tax_treatment"], dropna=False)
             .agg(
                 withdrawals=("withdrawals","sum"),
                 interest_income=("interest_income_base","sum"),
                 interest_reinvested=("interest_reinvested_base","sum"),
                 elig_div_income=("eligible_dividend_income_base","sum"),
                 elig_div_reinv=("eligible_dividends_reinvested_base","sum"),
                 nonelig_div_income=("noneligible_dividend_income_base","sum"),
                 nonelig_div_reinv=("noneligible_dividends_reinvested_base","sum"),
             )
             .reset_index())

    records = []
    for _, r in grp.iterrows():
        y     = int(r["y_idx"])
        pid   = int(r["portfolio_id"])
        pname = str(r["portfolio_name"])
        tt    = str(r["tax_treatment"] or "TAXABLE").upper()

        # Cash distributions that are taxable this year (exclude reinvested)
        taxable_interest   = max(0.0, float(r["interest_income"]   - r["interest_reinvested"]))
        taxable_elig_div   = max(0.0, float(r["elig_div_income"]   - r["elig_div_reinv"]))
        taxable_nonelig_div= max(0.0, float(r["nonelig_div_income"]- r["nonelig_div_reinv"]))

        # Withdrawals and employment
        withdrawals_nom = float(r["withdrawals"])
        emp_nom = float(emp_map.get(y0 + y, 0.0))

        # ---- Taxes by component ----
        tax_withdrawals = calculate_annual_tax(tax_rules_conn, withdrawals_nom, tt, year=(y0 + y))
        tax_interest    = calculate_annual_tax(tax_rules_conn, taxable_interest, "INTEREST", year=(y0 + y))
        tax_elig        = calculate_annual_tax(tax_rules_conn, taxable_elig_div, "ELIGIBLE_DIVIDEND", year=(y0 + y))
        tax_nonelig     = calculate_annual_tax(tax_rules_conn, taxable_nonelig_div, "NONELIGIBLE_DIVIDEND", year=(y0 + y))
        tax_emp         = calculate_annual_tax(tax_rules_conn, emp_nom, "EMPLOYMENT", year=(y0 + y))

        taxes = round(tax_withdrawals + tax_interest + tax_elig + tax_nonelig + tax_emp, 2)

        pretax_nom = withdrawals_nom + taxable_interest + taxable_elig_div + taxable_nonelig_div + emp_nom
        after_tax_nom = round(pretax_nom - taxes, 2)

        disc = float(cum_cpi.get(y, 1.0))  # deflate to real
        after_tax_real = round(after_tax_nom / disc, 2)

        records.append({
            "year": (y0 + y),
            "portfolio_id": pid,
            "portfolio_name": pname,
            "tax_treatment": tt,
            "after_tax_income": after_tax_nom,
            "real_after_tax_income": after_tax_real,
            "taxes_paid": taxes,
        })

    out = pd.DataFrame(records)
    return out[[
        "year","portfolio_id","portfolio_name","tax_treatment",
        "after_tax_income","real_after_tax_income","taxes_paid"
    ]]
