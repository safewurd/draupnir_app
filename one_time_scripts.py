# one_time_migration_forecast_annual.py
import sqlite3
from pathlib import Path

DB_PATH = Path("data/draupnir.db")

# ---- Configuration -----------------------------------------------------------
# If you're migrating old runs and want to backfill the new columns once,
# set this to True. Otherwise leave False (your engine should populate on next run).
BACKFILL_EXISTING = False

# ---- Helpers ----------------------------------------------------------------
def column_exists(conn, table, col):
    q = "PRAGMA table_info(%s)" % table
    return any(r[1] == col for r in conn.execute(q))

def add_column_safe(conn, table, coldef):
    colname = coldef.split()[0]
    if not column_exists(conn, table, colname):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {coldef}")

def table_exists(conn, table):
    r = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return r is not None

def drop_trigger_if_exists(conn, name):
    conn.execute("DROP TRIGGER IF EXISTS %s" % name)

# ---- Migration ---------------------------------------------------------------
with sqlite3.connect(DB_PATH) as conn:
    conn.execute("PRAGMA foreign_keys = ON;")

    if not table_exists(conn, "forecast_results_annual"):
        raise SystemExit("Table 'forecast_results_annual' not found. Run a forecast once, then re-run this migration.")

    # 1) Add recommended columns (amounts + single rate)
    # Nominal amounts
    add_column_safe(conn, "forecast_results_annual", "nominal_pretax_income REAL")
    add_column_safe(conn, "forecast_results_annual", "nominal_taxes_paid REAL")
    add_column_safe(conn, "forecast_results_annual", "nominal_after_tax_income REAL")
    # Real amounts
    add_column_safe(conn, "forecast_results_annual", "real_pretax_income REAL")
    add_column_safe(conn, "forecast_results_annual", "real_taxes_paid REAL")
    add_column_safe(conn, "forecast_results_annual", "real_after_tax_income REAL")
    # Single effective rate (nominal). Keep/alias for display as needed.
    add_column_safe(conn, "forecast_results_annual", "nominal_effective_tax_rate REAL")

    # 2) If a legacy real_effective_tax_rate exists, keep it in sync (optional convenience)
    has_real_rate = column_exists(conn, "forecast_results_annual", "real_effective_tax_rate")

    # 3) Create compute triggers (re-create to ensure latest logic)
    drop_trigger_if_exists(conn, "fta_ai_compute_derived")  # after insert
    drop_trigger_if_exists(conn, "fta_au_compute_derived")  # after update

    compute_sql_shared = """
        UPDATE forecast_results_annual
        SET
            nominal_after_tax_income = 
                CASE
                    WHEN NEW.nominal_pretax_income IS NOT NULL AND NEW.nominal_taxes_paid IS NOT NULL
                    THEN NEW.nominal_pretax_income - NEW.nominal_taxes_paid
                    ELSE nominal_after_tax_income
                END,
            real_after_tax_income =
                CASE
                    WHEN NEW.real_pretax_income IS NOT NULL AND NEW.real_taxes_paid IS NOT NULL
                    THEN NEW.real_pretax_income - NEW.real_taxes_paid
                    ELSE real_after_tax_income
                END,
            nominal_effective_tax_rate =
                CASE
                    WHEN NEW.nominal_pretax_income IS NOT NULL AND NEW.nominal_pretax_income > 0
                    THEN COALESCE(NEW.nominal_taxes_paid, 0.0) * 1.0 / NULLIF(NEW.nominal_pretax_income, 0)
                    ELSE NULL
                END
            {maybe_real_rate}
        WHERE rowid = NEW.rowid;
    """.strip()

    maybe_real_rate_clause = ""
    if has_real_rate:
        maybe_real_rate_clause = """,
            real_effective_tax_rate = nominal_effective_tax_rate
        """

    conn.executescript(f"""
        CREATE TRIGGER fta_ai_compute_derived
        AFTER INSERT ON forecast_results_annual
        BEGIN
            {compute_sql_shared.format(maybe_real_rate=maybe_real_rate_clause)}
        END;

        CREATE TRIGGER fta_au_compute_derived
        AFTER UPDATE ON forecast_results_annual
        BEGIN
            {compute_sql_shared.format(maybe_real_rate=maybe_real_rate_clause)}
        END;
    """)

    # 4) Optional: backfill existing rows once so current data has derived fields
    if BACKFILL_EXISTING:
        # Compute derived fields for all existing rows
        conn.execute("""
            UPDATE forecast_results_annual
            SET
                nominal_after_tax_income =
                    CASE
                        WHEN nominal_pretax_income IS NOT NULL AND nominal_taxes_paid IS NOT NULL
                        THEN nominal_pretax_income - nominal_taxes_paid
                        ELSE nominal_after_tax_income
                    END,
                real_after_tax_income =
                    CASE
                        WHEN real_pretax_income IS NOT NULL AND real_taxes_paid IS NOT NULL
                        THEN real_pretax_income - real_taxes_paid
                        ELSE real_after_tax_income
                    END,
                nominal_effective_tax_rate =
                    CASE
                        WHEN nominal_pretax_income IS NOT NULL AND nominal_pretax_income > 0
                        THEN COALESCE(nominal_taxes_paid, 0.0) * 1.0 / NULLIF(nominal_pretax_income, 0)
                        ELSE NULL
                    END
        """)
        if has_real_rate:
            conn.execute("""
                UPDATE forecast_results_annual
                SET real_effective_tax_rate = nominal_effective_tax_rate
            """)

    # 5) Show final schema (for a quick sanity check)
    cols = conn.execute("PRAGMA table_info(forecast_results_annual)").fetchall()
    print("forecast_results_annual columns:")
    for cid, name, ctype, notnull, dflt, pk in cols:
        print(f" - {name} {ctype}")

print("Migration complete.")
