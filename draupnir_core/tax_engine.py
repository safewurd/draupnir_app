# draupnir_core/tax_engine.py
from __future__ import annotations

import sqlite3
from typing import List, Tuple


# -----------------------------
# Brackets I/O
# -----------------------------

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

    # Fallback year
    fallback = conn.execute(
        "SELECT MAX(year) FROM TaxRules WHERE jurisdiction = ? AND income_type = ?",
        (jurisdiction, income_type),
    ).fetchone()
    if not fallback or fallback[0] is None:
        return []
    fy = int(fallback[0])
    return conn.execute(q, (fy, jurisdiction, income_type)).fetchall()


# -----------------------------
# Calculations
# -----------------------------

def _calc_progressive(amount: float, brackets: List[Tuple[float, float | None, float, float]]) -> float:
    """
    Apply progressive brackets. 'credit' fields are summed and netted at the end.
    """
    if amount <= 0 or not brackets:
        return 0.0

    tax = 0.0
    credits = 0.0
    for bmin, bmax, rate, credit in brackets:
        credits += float(credit or 0.0)
        if bmax is None:
            taxable = max(0.0, amount - float(bmin))
            tax += taxable * float(rate)
            break
        if amount > float(bmin):
            span = max(0.0, min(amount, float(bmax)) - float(bmin))
            tax += span * float(rate)

    tax = max(0.0, tax - credits)
    return round(tax, 2)


def calculate_annual_tax(
    conn: sqlite3.Connection,
    annual_income: float,
    tax_treatment: str,
    year: int,
    province: str = "Ontario",
) -> float:
    """
    Compute annual tax on 'annual_income' under a given account tax_treatment.

    Treatments:
      - TFSA -> 0
      - RRSP/RRIF/PENSION -> employment schedule on withdrawals
      - TAXABLE -> 50% inclusion (treated as employment for bracket schedule simplicity)
      - EMPLOYMENT or anything else -> employment schedule
    """
    if annual_income <= 0:
        return 0.0

    tt = (tax_treatment or "").strip().upper()
    income_for_tax = float(annual_income)

    if tt == "TFSA":
        return 0.0
    if tt == "TAXABLE":
        income_for_tax *= 0.5  # capital gains inclusion approximation

    # Use Employment bracket schedules for simplicity/availability
    fed = _get_tax_brackets(conn, year, "Federal", "Employment")
    prov = _get_tax_brackets(conn, year, province, "Employment")

    return round(_calc_progressive(income_for_tax, fed) + _calc_progressive(income_for_tax, prov), 2)


def allocate_monthly_taxes(annual_tax: float, monthly_incomes: list[float]) -> list[float]:
    """
    Allocate annual tax proportionally across months by income share.
    """
    total = sum(monthly_incomes) if monthly_incomes else 0.0
    if total <= 0:
        return [0.0] * (len(monthly_incomes) if monthly_incomes else 12)
    return [round((x / total) * annual_tax, 2) for x in monthly_incomes]
