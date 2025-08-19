# draupnir_core/tax_engine.py
from __future__ import annotations

import sqlite3
from typing import List, Tuple, Optional, Dict


# =============================
# Brackets I/O
# =============================

def _get_tax_brackets(
    conn: sqlite3.Connection,
    year: int,
    jurisdiction: str,
    income_type: str,
) -> List[Tuple[float, float | None, float, float]]:
    """
    Return [(bracket_min, bracket_max_or_None, rate, credit), ...]
    Fallback to latest available year if requested year missing.
    """
    q = """
        SELECT bracket_min, bracket_max, rate, credit
        FROM TaxRules
        WHERE year = ? AND jurisdiction = ? AND income_type = ?
        ORDER BY bracket_min ASC
    """
    rows = conn.execute(q, (year, jurisdiction, income_type)).fetchall()
    if rows:
        return rows

    # Fallback year (latest available for that jurisdiction/type)
    fallback = conn.execute(
        "SELECT MAX(year) FROM TaxRules WHERE jurisdiction = ? AND income_type = ?",
        (jurisdiction, income_type),
    ).fetchone()
    if not fallback or fallback[0] is None:
        return []
    fy = int(fallback[0])
    return conn.execute(q, (fy, jurisdiction, income_type)).fetchall()


# =============================
# Dividend policy (Canadian)
# =============================

# Sensible fallbacks if tax_policy table is missing or incomplete
# (These match what the Tax tab seeds by default.)
_FALLBACK_POLICY = {
    "cap_gains_inclusion": 0.50,
    "eligible_div_gross_up": 0.38,
    "noneligible_div_gross_up": 0.15,
    "fed_eligible_div_credit_rate": 0.150198,
    "fed_noneligible_div_credit_rate": 0.090301,
    # Ontario defaults (if province lookup missing)
    "prov_eligible_div_credit_rate": 0.1000,
    "prov_noneligible_div_credit_rate": 0.0299,
}

_PROVINCE_MAP = {
    # Normalize short codes seen elsewhere into full TaxRules jurisdiction names
    "ON": "Ontario", "BC": "British Columbia", "AB": "Alberta", "SK": "Saskatchewan",
    "MB": "Manitoba", "QC": "Quebec", "NB": "New Brunswick", "NS": "Nova Scotia",
    "PE": "Prince Edward Island", "NL": "Newfoundland and Labrador", "YT": "Yukon",
    "NT": "Northwest Territories", "NU": "Nunavut",
}

def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?;", (name,)).fetchone()
    return row is not None


def get_province_for_year(conn: sqlite3.Connection, year: int, fallback: str = "Ontario") -> str:
    """
    Reads tax_profile(year->province) if available; otherwise returns fallback.
    Returns full province name when possible for matching TaxRules.
    """
    prov = None
    if _table_exists(conn, "tax_profile"):
        row = conn.execute("SELECT province FROM tax_profile WHERE year = ? LIMIT 1;", (int(year),)).fetchone()
        if row and row[0]:
            prov = str(row[0]).strip().upper()

    if not prov:
        # Try global settings/settings wide tables (some deployments store a single province there)
        for q in (
            "SELECT province FROM settings LIMIT 1;",
            "SELECT value FROM settings WHERE key IN ('province','PROVINCE') LIMIT 1;",
            "SELECT value FROM global_settings WHERE key IN ('province','PROVINCE') LIMIT 1;",
        ):
            try:
                r = conn.execute(q).fetchone()
                if r and r[0]:
                    prov = str(r[0]).strip().upper()
                    break
            except Exception:
                pass

    if not prov:
        prov = fallback

    # Return full name if we can map it
    return _PROVINCE_MAP.get(prov, prov.title())


def _get_dividend_policy(conn: sqlite3.Connection, year: int, province_fullname: str) -> Dict[str, float]:
    """
    Loads gross‑ups and dividend tax credit rates from tax_policy(year, province),
    with robust fallbacks.
    """
    pol = dict(_FALLBACK_POLICY)

    if _table_exists(conn, "tax_policy"):
        row = conn.execute("""
            SELECT cap_gains_inclusion,
                   eligible_div_gross_up, noneligible_div_gross_up,
                   fed_eligible_div_credit_rate, fed_noneligible_div_credit_rate,
                   prov_eligible_div_credit_rate, prov_noneligible_div_credit_rate
            FROM tax_policy
            WHERE year = ? AND UPPER(province) IN (?, ?)
            LIMIT 1;
        """, (int(year), province_fullname.upper(), province_fullname)).fetchone()
        if row:
            keys = list(pol.keys())
            for i, k in enumerate(keys):
                try:
                    v = float(row[i])
                    if v is not None:
                        pol[k] = v
                except Exception:
                    pass

    return pol


# =============================
# Progressive calculation
# =============================

def _calc_progressive(amount: float, brackets: List[Tuple[float, float | None, float, float]]) -> float:
    """
    Apply progressive brackets. 'credit' fields in the bracket table are summed and
    netted at the end (very simple non‑refundable style).
    """
    if amount <= 0 or not brackets:
        return 0.0

    tax = 0.0
    credits = 0.0
    for bmin, bmax, rate, credit in brackets:
        credits += float(credit or 0.0)
        bmin = float(bmin)
        bmax = (None if bmax is None else float(bmax))
        rate = float(rate)
        if bmax is None:  # top bracket
            taxable = max(0.0, amount - bmin)
            tax += taxable * rate
            break
        if amount > bmin:
            span = max(0.0, min(amount, bmax) - bmin)
            tax += span * rate

    tax = max(0.0, tax - credits)
    return round(tax, 2)


# =============================
# Public tax APIs
# =============================

def calculate_annual_tax(
    conn: sqlite3.Connection,
    annual_income: float,
    tax_treatment: str,
    year: int,
    province: str = "Ontario",
) -> float:
    """
    Legacy helper retained for compatibility.

    - TFSA -> 0
    - RRSP/RRIF/PENSION -> employment schedule on withdrawals
    - TAXABLE -> (legacy approximation) 50% inclusion like capital gains — not used by the new pipeline
    - EMPLOYMENT or anything else -> employment schedule
    """
    if annual_income <= 0:
        return 0.0

    tt = (tax_treatment or "").strip().upper()
    income_for_tax = float(annual_income)

    if tt == "TFSA":
        return 0.0
    if tt == "TAXABLE":
        income_for_tax *= 0.5  # old approximation — superseded by calculate_personal_tax

    fed = _get_tax_brackets(conn, year, "Federal", "Employment")
    prov = _get_tax_brackets(conn, year, province, "Employment")
    return round(_calc_progressive(income_for_tax, fed) + _calc_progressive(income_for_tax, prov), 2)


def calculate_personal_tax(
    conn: sqlite3.Connection,
    year: int,
    *,
    province: Optional[str] = None,
    employment_like_income: float = 0.0,   # wages + RRSP/RRIF withdrawals etc.
    interest_income: float = 0.0,          # fully included
    eligible_dividends_cash: float = 0.0,  # cash amount received
    noneligible_dividends_cash: float = 0.0,
    capital_gains: float = 0.0             # full amount of the gain; inclusion rate applied from policy
) -> float:
    """
    Canadian personal tax (federal + provincial) on a combined income stack.

    Steps:
      1) Convert dividends to grossed-up (taxable) amounts.
      2) Apply capital gains inclusion rate.
      3) Compute base tax using Employment brackets (typical PIT schedule).
      4) Subtract dividend tax credits (federal + provincial) computed on the grossed-up dividends.

    Returns total tax payable (floored at 0).
    """
    prov_full = province or get_province_for_year(conn, year, "Ontario")
    policy = _get_dividend_policy(conn, year, prov_full)

    # 1) Gross-up dividends to taxable income
    gross_elig = float(eligible_dividends_cash) * (1.0 + float(policy["eligible_div_gross_up"]))
    gross_non  = float(noneligible_dividends_cash) * (1.0 + float(policy["noneligible_div_gross_up"]))

    # 2) Apply capital gains inclusion to gains
    included_gains = float(capital_gains) * float(policy["cap_gains_inclusion"])

    # 3) Base tax using progressive employment brackets
    taxable_income = (
        float(employment_like_income) +
        float(interest_income) +
        gross_elig + gross_non +
        included_gains
    )
    if taxable_income <= 0:
        return 0.0

    fed_br = _get_tax_brackets(conn, year, "Federal", "Employment")
    prov_br = _get_tax_brackets(conn, year, prov_full, "Employment")
    base_tax = _calc_progressive(taxable_income, fed_br) + _calc_progressive(taxable_income, prov_br)

    # 4) Non-refundable dividend credits (very simplified model)
    dtc_fed = (
        float(policy["fed_eligible_div_credit_rate"]) * gross_elig +
        float(policy["fed_noneligible_div_credit_rate"]) * gross_non
    )
    dtc_prov = (
        float(policy["prov_eligible_div_credit_rate"]) * gross_elig +
        float(policy["prov_noneligible_div_credit_rate"]) * gross_non
    )
    tax_after_credits = max(0.0, base_tax - dtc_fed - dtc_prov)
    return round(tax_after_credits, 2)


def allocate_monthly_taxes(annual_tax: float, monthly_incomes: list[float]) -> list[float]:
    """
    Allocate annual tax proportionally across months by income share.
    """
    total = sum(monthly_incomes) if monthly_incomes else 0.0
    if total <= 0:
        return [0.0] * (len(monthly_incomes) if monthly_incomes else 12)
    return [round((x / total) * annual_tax, 2) for x in monthly_incomes]
