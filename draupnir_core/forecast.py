# draupnir_core/forecast.py
import sqlite3
import os
from io import BytesIO
from datetime import datetime

import pandas as pd
import streamlit as st

from .forecast_engine import run_forecast, ForecastParams, ensure_portfolio_flows_table

# ---- Unified DB path ----
os.makedirs("data", exist_ok=True)
DB_PATH = os.path.join("data", "draupnir.db")

# -----------------------------
# DB helpers
# -----------------------------

def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?;", (name,)
    ).fetchone()
    return row is not None

def _load_portfolios() -> pd.DataFrame:
    conn = _connect(DB_PATH)
    try:
        return pd.read_sql("SELECT portfolio_id, portfolio_name FROM portfolios ORDER BY portfolio_name;", conn)
    finally:
        conn.close()

def _get_setting(key: str) -> str | None:
    conn = _connect(DB_PATH)
    try:
        if not _table_exists(conn, "global_settings"):
            return None
        row = conn.execute("SELECT value FROM global_settings WHERE key = ? LIMIT 1;", (key,)).fetchone()
        return None if not row else (row[0] or None)
    finally:
        conn.close()

# -----------------------------
# Introspection helpers (monthly)
# -----------------------------

def _get_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
        return {r[1] for r in rows}
    except Exception:
        return set()

def _monthly_real_value_column(conn: sqlite3.Connection) -> str | None:
    """
    Return the real value column name in forecast_results_monthly.
    New writer uses 'real_value'; legacy rows might have 'value_real'.
    """
    cols = _get_columns(conn, "forecast_results_monthly")
    if "real_value" in cols:
        return "real_value"
    if "value_real" in cols:
        return "value_real"
    return None

# -----------------------------
# Annual results loader (from DB) — NEW-ONLY columns
# -----------------------------

def _load_annual_results_for_run(run_id: int) -> pd.DataFrame:
    conn = _connect(DB_PATH)
    try:
        if not _table_exists(conn, "forecast_results_annual"):
            return pd.DataFrame()

        # Pull NEW columns only
        a = pd.read_sql("""
            SELECT
                year,
                portfolio_id,
                contributions,
                withdrawals,
                real_pretax_income,
                real_taxes_paid,
                real_after_tax_income,
                real_effective_tax_rate
            FROM forecast_results_annual
            WHERE run_id = ?
        """, conn, params=(run_id,))

        # Bring in year-end real value (per portfolio) from monthly table
        if not _table_exists(conn, "forecast_results_monthly"):
            a["value_real"] = 0.0
            return a

        real_col = _monthly_real_value_column(conn)
        if not real_col:
            a["value_real"] = 0.0
            return a

        m_sql = f"""
            SELECT year, portfolio_id, value_real
            FROM (
                SELECT
                    CAST(strftime('%Y', period) AS INTEGER) AS year,
                    portfolio_id,
                    {real_col} AS value_real,
                    ROW_NUMBER() OVER (
                        PARTITION BY CAST(strftime('%Y', period) AS INTEGER), portfolio_id
                        ORDER BY period DESC
                    ) AS rn
                FROM forecast_results_monthly
                WHERE run_id = ?
            ) t
            WHERE rn = 1
        """
        m = pd.read_sql(m_sql, conn, params=(run_id,))

        out = a.merge(m, on=["year","portfolio_id"], how="left")
        out["value_real"] = pd.to_numeric(out["value_real"], errors="coerce").fillna(0.0)
        return out
    finally:
        conn.close()

# -----------------------------
# Export (single sheet: Annual Results)
# -----------------------------

def _export_excel_ui(summary_df: pd.DataFrame, run_id: int | None):
    st.markdown("### ⬇️ Export Forecast")

    if st.button("Prepare Excel", key="btn_prepare_forecast_excel", type="secondary"):
        st.session_state["export_forecast_ready"] = True

    output_dir = _get_setting("forecast_output_dir") or ""
    save_disabled = (not output_dir.strip())
    if save_disabled:
        st.caption("Tip: set a **Forecast Output Directory** in Settings to enable server-side save.")

    if st.button("Save to Output Directory", key="btn_save_forecast_excel", disabled=save_disabled):
        if summary_df is None or summary_df.empty:
            st.error("No results to export.")
            return

        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            summary_df.to_excel(writer, index=False, sheet_name="Annual_Results")
            ws = writer.sheets["Annual_Results"]
            for col_cells in ws.columns:
                length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col_cells)
                ws.column_dimensions[col_cells[0].column_letter].width = min(max(length + 2, 12), 50)

        bio.seek(0)
        ts = datetime.utcnow().strftime('%Y%m%d_%H%M%SZ')
        fname = f"forecast_run_{(run_id or 0)}_{ts}.xlsx"

        try:
            os.makedirs(output_dir, exist_ok=True)
            fpath = os.path.abspath(os.path.join(output_dir, fname))
            with open(fpath, "wb") as f:
                f.write(bio.getvalue())
            st.success(f"✅ Saved to: `{fpath}`")
        except Exception as ex:
            st.error(f"Failed to save file: {ex}")

    if not st.session_state.get("export_forecast_ready"):
        return

    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="Annual_Results")
        ws = writer.sheets["Annual_Results"]
        for col_cells in ws.columns:
            length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col_cells)
            ws.column_dimensions[col_cells[0].column_letter].width = min(max(length + 2, 12), 50)
    bio.seek(0)

    fname = f"forecast_run_{(run_id or 0)}_{datetime.utcnow().strftime('%Y%m%d_%H%M%SZ')}.xlsx"
    st.download_button(
        label="Download Excel",
        data=bio.getvalue(),
        file_name=fname,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=False,
        key="btn_download_forecast_excel"
    )

# -----------------------------
# Cash Flows CRUD
# -----------------------------

def _ensure_flows_table():
    conn = _connect(DB_PATH)
    try:
        ensure_portfolio_flows_table(conn)
    finally:
        conn.close()

def _insert_flow(row: dict):
    conn = _connect(DB_PATH)
    try:
        with conn:
            conn.execute("""
                INSERT INTO portfolio_flows (portfolio_id, kind, amount, frequency, start_date, end_date, index_with_inflation, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                int(row["portfolio_id"]),
                row["kind"].strip().upper(),
                float(row["amount"]),
                row["frequency"].strip().lower(),
                row["start_date"],
                row["end_date"],
                1 if row.get("index_with_inflation") else 0,
                row.get("notes", "").strip(),
            ))
    finally:
        conn.close()

def _delete_flow(flow_id: int):
    conn = _connect(DB_PATH)
    try:
        with conn:
            conn.execute("DELETE FROM portfolio_flows WHERE flow_id = ?;", (int(flow_id),))
    finally:
        conn.close()

def _load_flows_for_portfolio(pid: int) -> pd.DataFrame:
    conn = _connect(DB_PATH)
    try:
        return pd.read_sql("""
            SELECT flow_id, kind, amount, frequency, start_date, end_date, index_with_inflation, notes, created_at
            FROM portfolio_flows
            WHERE portfolio_id = ?
            ORDER BY datetime(start_date) ASC;
        """, conn, params=(int(pid),))
    finally:
        conn.close()

# -----------------------------
# Assumptions editors
# -----------------------------

def _ensure_macro_table():
    conn = _connect(DB_PATH)
    try:
        if not _table_exists(conn, "MacroForecast"):
            conn.execute("""
                CREATE TABLE IF NOT EXISTS MacroForecast (
                    year INTEGER,
                    inflation REAL,
                    growth REAL,
                    fx REAL,
                    note TEXT
                );
            """)
            conn.commit()
    finally:
        conn.close()

def _ensure_employment_table():
    conn = _connect(DB_PATH)
    try:
        if not _table_exists(conn, "EmploymentIncome"):
            conn.execute("""
                CREATE TABLE IF NOT EXISTS EmploymentIncome (
                    year INTEGER,
                    amount REAL,
                    employer TEXT,
                    note TEXT
                );
            """)
            conn.commit()
    finally:
        conn.close()

def _load_generic_table(name: str) -> pd.DataFrame:
    conn = _connect(DB_PATH)
    try:
        if not _table_exists(conn, name):
            return pd.DataFrame()
        return pd.read_sql(f"SELECT * FROM {name}", conn)
    finally:
        conn.close()

def _save_generic_table(name: str, df: pd.DataFrame):
    conn = _connect(DB_PATH)
    try:
        with conn:
            df.to_sql(name, conn, if_exists="replace", index=False)
    finally:
        conn.close()

# -----------------------------
# UI
# -----------------------------

def forecast_tab():
    st.subheader("🔮 Forecast")

    # ========== A) Cash Flows by Portfolio (CRUD) ==========
    st.markdown("### 💵 Cash Flows by Portfolio")
    _ensure_flows_table()

    pf = _load_portfolios()
    if pf.empty:
        st.info("No portfolios found. Create one in Settings first.")
        return

    colsel, _ = st.columns([2, 1])
    with colsel:
        sel_name = st.selectbox("Portfolio", options=pf["portfolio_name"].tolist(), index=0, key="flows_pf_sel")
    sel_pid = int(pf.loc[pf["portfolio_name"] == sel_name, "portfolio_id"].iloc[0])

    with st.form("add_flow_form", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kind = st.selectbox("Type", options=["CONTRIBUTION", "WITHDRAWAL"], index=0, key="flow_kind")
            amount = st.number_input("Amount", min_value=0.0, step=100.0, format="%.2f", key="flow_amount")
        with c2:
            frequency = st.selectbox("Frequency", options=["monthly", "annual"], index=0, key="flow_freq")
            index_infl = st.checkbox("Index with inflation", value=True, key="flow_index_infl")
        with c3:
            start_date = st.text_input("Start (YYYY-MM-01)", placeholder="YYYY-MM-01", key="flow_start")
            end_date = st.text_input("End (optional YYYY-MM-01)", placeholder="", key="flow_end")
        with c4:
            notes = st.text_input("Notes", placeholder="optional", key="flow_notes")
        add_submitted = st.form_submit_button("Add Flow", use_container_width=True)

    if add_submitted:
        errs = []
        if not start_date or len(start_date) != 10:
            errs.append("Start date must be YYYY-MM-01.")
        if amount <= 0:
            errs.append("Amount must be greater than 0.")
        if errs:
            for e in errs:
                st.error(e)
        else:
            _insert_flow({
                "portfolio_id": sel_pid,
                "kind": kind,
                "amount": amount,
                "frequency": frequency,
                "start_date": start_date,
                "end_date": (end_date.strip() or None),
                "index_with_inflation": index_infl,
                "notes": notes,
            })
            st.success("Flow added.")
            st.rerun()

    flows_df = _load_flows_for_portfolio(sel_pid)
    if not flows_df.empty:
        st.dataframe(flows_df, use_container_width=True, hide_index=True)
        # Delete control
        del_col1, del_col2 = st.columns([2, 1])
        with del_col1:
            del_opt = st.selectbox("Select a flow to delete", options=[
                f"ID {r.flow_id} • {r.kind} {r.amount} {r.frequency} from {r.start_date}"
                + (f" to {r.end_date}" if r.end_date else "")
                for r in flows_df.itertuples(index=False)
            ], key="flow_del_opt")
        del_id = int(del_opt.split()[1]) if del_opt else None
        with del_col2:
            if st.button("Delete Flow", key="btn_delete_flow", disabled=(del_id is None)):
                _delete_flow(del_id)
                st.success(f"Deleted flow ID {del_id}.")
                st.rerun()
    else:
        st.info("No flows defined for this portfolio.")

    st.markdown("---")

    # ========== B) Assumptions ==========
    st.markdown("## 🧭 Assumptions")

    # -- Macro Forecast table editor
    st.markdown("### 📈 Macro Forecast")
    _ensure_macro_table()
    macro_df = _load_generic_table("MacroForecast")

    st.caption("Tip: include columns your engine expects (e.g., year, inflation, growth, fx). "
               "This editor is schema-agnostic and will preserve whatever columns you use.")

    macro_edit = st.data_editor(
        macro_df if not macro_df.empty else pd.DataFrame(columns=["year","inflation","growth","fx","note"]),
        num_rows="dynamic",
        use_container_width=True,
        key="macro_editor"
    )

    colm1, colm2 = st.columns([1, 3])
    with colm1:
        if st.button("Save Macro Forecast", key="btn_save_macro", type="primary"):
            try:
                if "year" in macro_edit.columns:
                    macro_edit["year"] = pd.to_numeric(macro_edit["year"], errors="coerce").astype("Int64")
                for c in ["inflation","growth","fx"]:
                    if c in macro_edit.columns:
                        macro_edit[c] = pd.to_numeric(macro_edit[c], errors="coerce")
                _save_generic_table("MacroForecast", macro_edit)
                st.success("✅ Macro Forecast saved.")
            except Exception as ex:
                st.error(f"Failed to save Macro Forecast: {ex}")

    st.markdown("---")

    # -- Employment Income table editor
    st.markdown("### 💼 Employment Income")
    _ensure_employment_table()
    emp_df = _load_generic_table("EmploymentIncome")

    emp_edit = st.data_editor(
        emp_df if not emp_df.empty else pd.DataFrame(columns=["year","amount","employer","note"]),
        num_rows="dynamic",
        use_container_width=True,
        key="emp_editor"
    )

    cole1, cole2 = st.columns([1, 3])
    with cole1:
        if st.button("Save Employment Income", key="btn_save_emp", type="primary"):
            try:
                if "year" in emp_edit.columns:
                    emp_edit["year"] = pd.to_numeric(emp_edit["year"], errors="coerce").astype("Int64")
                if "amount" in emp_edit.columns:
                    emp_edit["amount"] = pd.to_numeric(emp_edit["amount"], errors="coerce")
                _save_generic_table("EmploymentIncome", emp_edit)
                st.success("✅ Employment Income saved.")
            except Exception as ex:
                st.error(f"Failed to save Employment Income: {ex}")

    st.markdown("---")

    # ========== C) Forecast Runner ==========
    st.markdown("### ▶️ Run Forecast")
    with st.form("forecast_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            years = st.number_input("Years to simulate", min_value=1, max_value=100, value=40, step=1)
            use_macro = st.checkbox("Use Macro Forecast", value=True)
        with col2:
            # Whole-percent inputs: 10 => 10% = 0.10
            manual_growth_pct = st.number_input("Manual Growth (%)", min_value=0.0, value=10.0, step=0.5, format="%.0f")
            manual_infl_pct   = st.number_input("Manual Inflation (%)", min_value=0.0, value=3.0, step=0.5, format="%.0f")
        with col3:
            manual_fx = st.number_input("Manual FX (scalar)", min_value=1.0, value=1.37, step=0.01, format="%.2f")
            start_date = st.text_input("Start date (YYYY-MM-01)", value="", placeholder="optional")

        submitted = st.form_submit_button("Run Forecast", use_container_width=True)

    # ⛔ Only run when the user clicks the button
    if not submitted:
        st.info("Set your assumptions and click **Run Forecast**.")
        return

    # Convert whole % inputs to decimals
    mg = float(manual_growth_pct) / 100.0
    mi = float(manual_infl_pct)   / 100.0
    mx = float(manual_fx)
    start_date_val = start_date.strip() or None

    with st.spinner("Running forecast…"):
        try:
            out = run_forecast(
                db_path=DB_PATH,
                params=ForecastParams(
                    years=int(years),
                    cadence="monthly",
                    start_date=start_date_val,
                    use_macro=bool(use_macro),

                    # Used only when use_macro=False
                    manual_growth=mg,
                    manual_inflation=mi,
                    manual_fx=mx,

                    # Seed from current holdings at market
                    seed_from_holdings=True,
                    holdings_valuation="market",
                ),
                write_to_db=True,
                return_frames=True,
            )
        except Exception as ex:
            st.error(f"Forecast failed: {ex}")
            return

    run_id = out.get("run_id")
    c1, c2 = st.columns(2)
    c1.success(f"✅ Forecast complete (run_id={run_id}).")
    if use_macro:
        c2.caption(f"Assumptions: MacroForecast table • years={int(years)}.")
    else:
        c2.caption(f"Assumptions: manual constants • growth={mg:.4f}, infl={mi:.4f}, fx={mx:.4f} • years={int(years)}.")

    # ---------- Annual results table (summed across portfolios; NEW columns only) ----------
    st.markdown("### 📊 Annual Results (summed across portfolios)")
    raw = _load_annual_results_for_run(run_id)
    if raw.empty:
        st.info("No results to display.")
    else:
        g = (raw.groupby("year", as_index=False)[
            ["value_real","contributions","withdrawals","real_pretax_income","real_taxes_paid","real_after_tax_income"]
        ].sum(numeric_only=True))

        g["real_effective_tax_rate"] = g.apply(
            lambda r: (r["real_taxes_paid"] / r["real_pretax_income"]) if r["real_pretax_income"] else 0.0, axis=1
        )

        display = g[[
            "year",
            "value_real",
            "contributions",
            "withdrawals",
            "real_pretax_income",
            "real_taxes_paid",
            "real_after_tax_income",
            "real_effective_tax_rate",
        ]].copy()

        st.dataframe(display, use_container_width=True, hide_index=True)

        # Export (single sheet)
        _export_excel_ui(display, run_id)
