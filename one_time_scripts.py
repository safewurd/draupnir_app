# draupnir_core/one_time_tax_migration.py
import sqlite3
from draupnir_core.db_config import get_db_path

def run():
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # --- Control tables ---
    c.execute("""
    CREATE TABLE IF NOT EXISTS tax_profile(
        year INTEGER PRIMARY KEY,
        province TEXT NOT NULL
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS tax_policy(
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
    )""")

    c.execute("""
    CREATE TABLE IF NOT EXISTS tax_distribution_assumptions(
        scope TEXT PRIMARY KEY,
        interest_yield REAL NOT NULL,
        eligible_div_yield REAL NOT NULL,
        noneligible_div_yield REAL NOT NULL,
        last_updated TEXT
    )""")

    # --- Transparency columns in annual table (NULLable; non-breaking) ---
    cols = [
        ("interest_income_nominal", "REAL"),
        ("eligible_dividends_nominal", "REAL"),
        ("noneligible_dividends_nominal", "REAL"),
        ("realized_cap_gains_nominal", "REAL"),  # reserved; may already exist in your engine
        ("province_used", "TEXT"),
    ]
    for col, typ in cols:
        try:
            c.execute(f"ALTER TABLE forecast_results_annual ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass  # already exists

    conn.commit()
    conn.close()
    print("Tax migration complete.")

if __name__ == "__main__":
    run()
