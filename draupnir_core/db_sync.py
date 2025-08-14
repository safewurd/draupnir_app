# draupnir_core/db_sync.py
import os, json, glob, sqlite3, time
from pathlib import Path
import pandas as pd

def _conn(db_path: str):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)

def export_snapshot(db_path: str, seed_dir: str = "data/seed") -> dict:
    seed = Path(seed_dir)
    seed.mkdir(parents=True, exist_ok=True)
    meta = {"db_path": os.path.abspath(db_path), "exported_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    with _conn(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;")
        rows = cur.fetchall()
        tables = [r[0] for r in rows]
        schema_sql = "PRAGMA foreign_keys=OFF;\nBEGIN;\n" + "\n".join(r[1] for r in rows if r[1]) + "\nCOMMIT;\n"
        (seed / "schema.sql").write_text(schema_sql, encoding="utf-8")
        for t in tables:
            df = pd.read_sql_query(f"SELECT * FROM [{t}]", conn)
            df.to_csv(seed / f"{t}.csv", index=False)
    meta["tables"] = tables
    (seed / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta

def import_snapshot(db_path: str, seed_dir: str = "data/seed") -> dict:
    seed = Path(seed_dir)
    schema_file = seed / "schema.sql"
    if not schema_file.exists():
        raise FileNotFoundError(f"Missing {schema_file}. Run export first.")
    with _conn(db_path) as conn:
        conn.executescript("PRAGMA foreign_keys=OFF;")
        conn.executescript(schema_file.read_text(encoding="utf-8"))
        csvs = sorted(glob.glob(str(seed / "*.csv")))
        imported = []
        for path in csvs:
            table = Path(path).stem
            df = pd.read_csv(path)
            if not df.empty:
                cols = ", ".join([f"[{c}]" for c in df.columns])
                qs = ", ".join(["?" for _ in df.columns])
                conn.executemany(f"INSERT INTO [{table}] ({cols}) VALUES ({qs})", df.itertuples(index=False, name=None))
            imported.append({"table": table, "rows": 0 if df.empty else len(df)})
        conn.executescript("PRAGMA foreign_keys=ON;")
    return {"db_path": os.path.abspath(db_path), "imported": imported}
