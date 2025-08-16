import sqlite3
from pathlib import Path

# ✅ Your actual database path (use raw string to avoid backslash escape issues)
DB_PATH = r"C:\Users\ryann\Dropbox\Ryan\data\draupnir.db"
OUTPUT = "draupnir_schema.sql"

def main():
    db = Path(DB_PATH)
    if not db.exists():
        raise FileNotFoundError(f"DB not found at: {db}")

    conn = sqlite3.connect(str(db))
    try:
        with open(OUTPUT, "w", encoding="utf-8") as f:
            f.write("-- Draupnir schema export\n\n")
            for row in conn.execute(
                "SELECT sql FROM sqlite_master WHERE type IN ('table','view','index','trigger') "
                "AND sql IS NOT NULL ORDER BY type, name;"
            ):
                f.write(row[0] + ";\n\n")
        print(f"✅ Schema exported to {Path(OUTPUT).resolve()}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
