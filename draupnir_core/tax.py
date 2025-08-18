# draupnir_core/tax.py
import sqlite3
from typing import List
import pandas as pd
import streamlit as st
from datetime import datetime

from draupnir_core.db_config import get_db_path

PROVINCES = ["ON","BC","AB","SK","MB","QC","NB","NS","PE","NL","NT","YT","NU"]

DEFAULT_DISTRIBUTION = {
    "scope": "global",            # future: 'asset:<symbol>' overrides
    "interest_yield": 0.01,       # 1.0% annualized (fraction, not %)
    "eligible_div_yield": 0.02,   # 2.0% annualized
    "noneligible_div_yield": 0.005, # 0.5% annualized
}

DEFAULT_POLICY = {
    "cap_gains_inclusion": 0.50,
    "eligible_div_gross_up": 0.38,       # eligible dividend gross-up
    "noneligible_div_gross_up": 0.15,    # non-eligible dividend gross-up
    "fed_eligible_div_credit_rate": 0.150198,
    "fed_noneligible_div_credit_rate": 0.090301,
    # Ontario defaults for prov credits; we seed per-province below
    "prov_eligible_div_credit_rate": 0.1000,
    "prov_noneligible_div_credit_rate": 0.0299,
}

# --- Current policy constants (seed data) ---
FED_ELIG_DTC = 0.150198
FED_NONELIG_DTC = 0.090301
ELIG_GROSS_UP = 0.38
NONELIG_GROSS_UP = 0.15

PROV_ELIG_DTC = {
    "BC": 0.1200, "AB": 0.0812, "SK": 0.1100, "MB": 0.0800, "ON": 0.1000,
    "QC": 0.1170, "NB": 0.1400, "NS": 0.0885, "PE": 0.1050, "NL": 0.0630,
    "YT": 0.1202, "NT": 0.1150, "NU": 0.0551,
}
PROV_NONELIG_DTC_2024 = {
    "BC": 0.0196, "AB": 0.0218, "SK": 0.0252, "MB": 0.0078, "ON": 0.0299,
    "QC": 0.0342, "NB": 0.0275, "NS": 0.0299, "PE": 0.0130, "NL": 0.0320,
    "YT": 0.0067, "NT": 0.0600, "NU": 0.0261,
}
PROV_NONELIG_DTC_2025PLUS = dict(PROV_NONELIG_DTC_2024, **{"SK": 0.0294})
def _prov_nonelig_rate_for_year(prov: str, year: int) -> float:
    if year >= 2025:
        return PROV_NONELIG_DTC_2025PLUS.get(prov, PROV_NONELIG_DTC_2025PLUS["ON"])
    return PROV_NONELIG_DTC_2024.get(prov, PROV_NONELIG_DTC_2024["ON"])

def _ensure_tables(conn: sqlite3.Connection) -> None:
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS tax_profile (
            year INTEGER PRIMARY KEY,
            province TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS tax_policy (
            year INTEGER NOT NULL,
            province TEXT NOT NULL,
            cap_gains_inclusion REAL NOT NULL,
            eligible_div_gross_up REAL NOT NULL,
            noneligible_div_gross_up REAL NOT NULL,
            fed_eligible_div_credit_rate REAL NOT NULL,
            fed_noneligible_div_credit_rate REAL NOT NULL,
            prov_eligible_div_credit_rate REAL NOT NULL,
            prov_noneligible_div_credit_rate REAL NOT NULL,
            PRIMARY KEY (year, province)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS tax_distribution_assumptions (
            scope TEXT PRIMARY KEY,             -- 'global' for now; later: 'asset:XYZ'
            interest_yield REAL NOT NULL,
            eligible_div_yield REAL NOT NULL,
            noneligible_div_yield REAL NOT NULL,
            last_updated TEXT
        )
    """)
    conn.commit()

def _get_forecast_year_range(conn: sqlite3.Connection) -> List[int]:
    c = conn.cursor()
    try:
        c.execute("""
            SELECT MIN(CAST(STRFTIME('%Y', date) AS INTEGER)),
                   MAX(CAST(STRFTIME('%Y', date) AS INTEGER))
            FROM forecast_results_monthly
        """)
        r = c.fetchone()
        if r and r[0] and r[1] and r[0] <= r[1]:
            return list(range(int(r[0]), int(r[1]) + 1))
    except Exception:
        pass
    this_year = datetime.now().year
    return list(range(this_year, this_year + 31))

def seed_with_current_canadian_policy(conn: sqlite3.Connection, years: List[int]) -> None:
    c = conn.cursor()
    # provinces per year (default to ON)
    for y in years:
        c.execute("INSERT OR IGNORE INTO tax_profile(year, province) VALUES (?,?)", (y, "ON"))

    # upsert tax policy for each (year, province)
    rows = c.execute("SELECT year, province FROM tax_profile ORDER BY year").fetchall()
    if not rows:
        rows = [(y, "ON") for y in years]
    for y, p in rows:
        p = (p or "ON").strip().upper()
        elig_prov = PROV_ELIG_DTC.get(p, PROV_ELIG_DTC["ON"])
        nonelig_prov = _prov_nonelig_rate_for_year(p, int(y))
        c.execute("""
            INSERT OR REPLACE INTO tax_policy(
                year, province,
                cap_gains_inclusion,
                eligible_div_gross_up, noneligible_div_gross_up,
                fed_eligible_div_credit_rate, fed_noneligible_div_credit_rate,
                prov_eligible_div_credit_rate, prov_noneligible_div_credit_rate
            ) VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            int(y), p,
            0.50,
            ELIG_GROSS_UP, NONELIG_GROSS_UP,
            FED_ELIG_DTC, FED_NONELIG_DTC,
            float(elig_prov), float(nonelig_prov),
        ))
    conn.commit()

def _seed_distribution_if_missing(conn: sqlite3.Connection) -> None:
    c = conn.cursor()
    c.execute("SELECT 1 FROM tax_distribution_assumptions WHERE scope='global'")
    if not c.fetchone():
        c.execute("""
            INSERT INTO tax_distribution_assumptions(scope, interest_yield, eligible_div_yield,
                noneligible_div_yield, last_updated)
            VALUES (?,?,?,?, datetime('now'))
        """, (
            "global",
            DEFAULT_DISTRIBUTION["interest_yield"],
            DEFAULT_DISTRIBUTION["eligible_div_yield"],
            DEFAULT_DISTRIBUTION["noneligible_div_yield"],
        ))
    conn.commit()

def _load_profile_df(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT year, province FROM tax_profile ORDER BY year", conn)

def _load_policy_df(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("""
        SELECT year, province,
               cap_gains_inclusion,
               eligible_div_gross_up, noneligible_div_gross_up,
               fed_eligible_div_credit_rate, fed_noneligible_div_credit_rate,
               prov_eligible_div_credit_rate, prov_noneligible_div_credit_rate
        FROM tax_policy
        ORDER BY year, province
    """, conn)

def _load_distribution_df(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("""
        SELECT scope, interest_yield, eligible_div_yield, noneligible_div_yield, last_updated
        FROM tax_distribution_assumptions
        WHERE scope='global'
    """, conn)

def _save_profile_df(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    c = conn.cursor()
    c.execute("DELETE FROM tax_profile")
    for _, r in df.iterrows():
        if pd.isna(r["year"]) or pd.isna(r["province"]):
            continue
        c.execute("INSERT OR REPLACE INTO tax_profile(year, province) VALUES (?,?)",
                  (int(r["year"]), str(r["province"]).strip().upper()))
    conn.commit()

def _save_policy_df(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    c = conn.cursor()
    c.execute("DELETE FROM tax_policy")
    for _, r in df.iterrows():
        c.execute("""
            INSERT OR REPLACE INTO tax_policy(
                year, province, cap_gains_inclusion,
                eligible_div_gross_up, noneligible_div_gross_up,
                fed_eligible_div_credit_rate, fed_noneligible_div_credit_rate,
                prov_eligible_div_credit_rate, prov_noneligible_div_credit_rate
            ) VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            int(r["year"]), str(r["province"]).strip().upper(),
            float(r["cap_gains_inclusion"]),
            float(r["eligible_div_gross_up"]), float(r["noneligible_div_gross_up"]),
            float(r["fed_eligible_div_credit_rate"]), float(r["fed_noneligible_div_credit_rate"]),
            float(r["prov_eligible_div_credit_rate"]), float(r["prov_noneligible_div_credit_rate"]),
        ))
    conn.commit()

def _save_distribution_df(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    c = conn.cursor()
    for _, r in df.iterrows():
        c.execute("""
            INSERT OR REPLACE INTO tax_distribution_assumptions(
                scope, interest_yield, eligible_div_yield, noneligible_div_yield, last_updated
            ) VALUES (?,?,?,?, datetime('now'))
        """, (
            "global",
            float(r["interest_yield"]), float(r["eligible_div_yield"]), float(r["noneligible_div_yield"])
        ))
    conn.commit()

def tax_tab():
    st.header("Tax — Assumptions & Policy")
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)

    _ensure_tables(conn)
    years = _get_forecast_year_range(conn)
    seed_with_current_canadian_policy(conn, years)
    _seed_distribution_if_missing(conn)

    with st.expander("1) Province by Year (Multi-Province)", expanded=True):
        prof_df = _load_profile_df(conn)
        c1, c2 = st.columns([3, 2])
        with c1:
            st.caption("Assign the province used for personal income tax by calendar year.")
        with c2:
            add_year = st.number_input("Add year", min_value=1900, max_value=3000,
                                       value=years[0] if years else datetime.now().year, step=1)
            add_btn = st.button("Add/Show Year", use_container_width=True)
            if add_btn and add_year not in prof_df["year"].tolist():
                prof_df.loc[len(prof_df)] = {"year": int(add_year), "province": "ON"}

        prof_df = st.data_editor(
            prof_df,
            key="tax_profile_editor",
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "year": st.column_config.NumberColumn("Year", help="Calendar year"),
                "province": st.column_config.SelectboxColumn("Province", options=PROVINCES),
            },
        )
        if st.button("Save Province Schedule", type="primary"):
            _save_profile_df(conn, prof_df)
            seed_with_current_canadian_policy(conn, years)
            st.success("Saved province schedule and refreshed policy for those years.")

    with st.expander("2) Distribution Detail (Taxable Accounts)", expanded=True):
        st.caption("Set annual distribution yields (as **fractions**) applied to the average market value of **Taxable** portfolios. These are treated as distributed (and reinvested) for tax purposes.")
        dist_df = _load_distribution_df(conn)
        if dist_df.empty:
            dist_df = pd.DataFrame([DEFAULT_DISTRIBUTION]).drop(columns=["scope"], errors="ignore")
        edit_df = dist_df[["interest_yield","eligible_div_yield","noneligible_div_yield"]].copy()
        edit_df = st.data_editor(
            edit_df,
            key="tax_distribution_editor",
            num_rows=1,
            use_container_width=True,
            column_config={
                "interest_yield": st.column_config.NumberColumn("Interest Yield (annual)", min_value=0.0, max_value=1.0, step=0.001),
                "eligible_div_yield": st.column_config.NumberColumn("Eligible Dividend Yield (annual)", min_value=0.0, max_value=1.0, step=0.001),
                "noneligible_div_yield": st.column_config.NumberColumn("Non-Eligible Dividend Yield (annual)", min_value=0.0, max_value=1.0, step=0.001),
            },
        )
        if st.button("Save Distribution Assumptions", type="primary"):
            _save_distribution_df(conn, edit_df)
            st.success("Saved distribution assumptions.")

    with st.expander("3) Policy Knobs (Gross-Ups, Credits, Inclusion)", expanded=False):
        pol_df = _load_policy_df(conn)
        pol_df = st.data_editor(
            pol_df,
            key="tax_policy_editor",
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "year": st.column_config.NumberColumn("Year"),
                "province": st.column_config.SelectboxColumn("Province", options=PROVINCES),
                "cap_gains_inclusion": st.column_config.NumberColumn("Capital Gains Inclusion", min_value=0.0, max_value=1.0, step=0.01),
                "eligible_div_gross_up": st.column_config.NumberColumn("Eligible Div Gross-Up", min_value=0.0, max_value=1.0, step=0.01),
                "noneligible_div_gross_up": st.column_config.NumberColumn("Non-Eligible Div Gross-Up", min_value=0.0, max_value=1.0, step=0.01),
                "fed_eligible_div_credit_rate": st.column_config.NumberColumn("Fed Eligible Credit Rate", min_value=0.0, max_value=1.0, step=0.001),
                "fed_noneligible_div_credit_rate": st.column_config.NumberColumn("Fed Non-Eligible Credit Rate", min_value=0.0, max_value=1.0, step=0.001),
                "prov_eligible_div_credit_rate": st.column_config.NumberColumn("Prov Eligible Credit Rate", min_value=0.0, max_value=1.0, step=0.001),
                "prov_noneligible_div_credit_rate": st.column_config.NumberColumn("Prov Non-Eligible Credit Rate", min_value=0.0, max_value=1.0, step=0.001),
            },
        )
        c3, c4 = st.columns([1,1])
        with c3:
            if st.button("Save Policy", type="primary"):
                _save_policy_df(conn, pol_df)
                st.success("Saved policy.")
        with c4:
            if st.button("Load Defaults for All Rows"):
                now = pol_df.copy()
                now["cap_gains_inclusion"] = DEFAULT_POLICY["cap_gains_inclusion"]
                now["eligible_div_gross_up"] = DEFAULT_POLICY["eligible_div_gross_up"]
                now["noneligible_div_gross_up"] = DEFAULT_POLICY["noneligible_div_gross_up"]
                now["fed_eligible_div_credit_rate"] = DEFAULT_POLICY["fed_eligible_div_credit_rate"]
                now["fed_noneligible_div_credit_rate"] = DEFAULT_POLICY["fed_noneligible_div_credit_rate"]
                now["prov_eligible_div_credit_rate"] = DEFAULT_POLICY["prov_eligible_div_credit_rate"]
                now["prov_noneligible_div_credit_rate"] = DEFAULT_POLICY["prov_noneligible_div_credit_rate"]
                st.session_state["tax_policy_editor"] = now
                st.info("Defaults staged. Click 'Save Policy' to persist.")

    st.caption("Note: These are modelling assumptions. Nothing in this tab removes existing functionality.")
    conn.close()
