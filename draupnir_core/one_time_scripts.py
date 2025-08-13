# delete_portfolio_3.py

import sqlite3

DB_PATH = "draupnir.db"
PORTFOLIO_ID_TO_DELETE = 3

def delete_portfolio_and_trades(db_path, portfolio_id):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Delete trades first to avoid foreign key constraints (if enabled)
    cur.execute("DELETE FROM trades WHERE portfolio_id = ?", (portfolio_id,))
    trades_deleted = cur.rowcount

    # Delete the portfolio
    cur.execute("DELETE FROM portfolios WHERE portfolio_id = ?", (portfolio_id,))
    portfolios_deleted = cur.rowcount

    conn.commit()
    conn.close()

    print(f"✅ Deleted {trades_deleted} trades and {portfolios_deleted} portfolio row(s) for portfolio_id = {portfolio_id}")

if __name__ == "__main__":
    delete_portfolio_and_trades(DB_PATH, PORTFOLIO_ID_TO_DELETE)

