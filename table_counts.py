import sqlite3
con = sqlite3.connect(r"data/draupnir.db")
cur = con.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;")
tables = [n for (n,) in cur.fetchall()]
print("Tables and row counts:")
for t in tables:
    cnt = cur.execute(f"SELECT COUNT(*) FROM [{t}]").fetchone()[0]
    print(f"{t}: {cnt}")
