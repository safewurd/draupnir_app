import sqlite3
from draupnir_core.db_config import get_db_path

DB_PATH = get_db_path()

def clear_portfolio_flows(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM portfolio_flows")
        print(f"✅ Cleared portfolio_flows ({cur.rowcount} rows)")
    except sqlite3.OperationalError as e:
        print(f"⚠️ portfolio_flows table not found: {e}")
    conn.commit()
    conn.close()

if __name__ == "__main__":
    print(f"Using DB: {DB_PATH}")
    clear_portfolio_flows(DB_PATH)
