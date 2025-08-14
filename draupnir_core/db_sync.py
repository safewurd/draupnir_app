# draupnir_core/db_sync.py
from __future__ import annotations

import os
import re
import json
import time
import glob
import sqlite3
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd


def _conn(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


# --- helpers -----------------------------------------------------------------
_CREATE_OBJ_RE = re.compile(
    r'CREATE\s+(?:TEMP|TEMPORARY\s+)?(TABLE|VIEW|INDEX)\s+'
    r'(?:IF\s+NOT\s+EXISTS\s+)?'
    r'(?P<name>(?:"[^"]+"|\[[^\]]+\]|`[^`]+`|\w+))\s*\(',
    re.IGNORECASE,
)

def _ensure_semicolons_between_creates(sql: str) -> str:
    # add ; between consecutive CREATE statements like ")  CREATE TABLE ..."
    return re.sub(r'\)\s*(?=CREATE\s+(?:TABLE|VIEW|INDEX)\b)', ');\n', sql, flags=re.IGNORECASE)

def _strip_control_statements(sql: str) -> str:
    # remove any PRAGMA/BEGIN/COMMIT lines present in schema.sql
    sql = re.sub(r'^\s*PRAGMA\b[^;]*;?\s*', '', sql, flags=re.IGNORECASE | re.MULTILINE)
    sql = re.sub(r'^\s*BEGIN\b[^;]*;?\s*',  '', sql, flags=re.IGNORECASE | re.MULTILINE)
    sql = re.sub(r'^\s*COMMIT\b[^;]*;?\s*', '', sql, flags=re.IGNORECASE | re.MULTILINE)
    return sql

def _add_drop_before_create_table(sql: str) -> str:
    # Insert DROP TABLE IF EXISTS before each CREATE TABLE to make re-import idempotent.
    def repl(m: re.Match) -> str:
        full = m.group(0)
        name = m.group('name')
        return f'DROP TABLE IF EXISTS {name};\n{full}'
    return re.sub(_CREATE_OBJ_RE, repl, sql)

def _final_semicolon(sql: str) -> str:
    return sql if sql.rstrip().endswith(';') else (sql.rstrip() + ';\n')


# --- export ------------------------------------------------------------------
def export_snapshot(db_path: str, seed_dir: str = "data/seed") -> Dict[str, Any]:
    """
    Export schema + all user tables to CSV under data/seed/.
    Writes:
      - data/seed/schema.sql   (tables only; semicolons; DROP TABLE IF EXISTS)
      - data/seed/<table>.csv
      - data/seed/manifest.json
    """
    seed = Path(seed_dir)
    seed.mkdir(parents=True, exist_ok=True)

    if not Path(db_path).exists():
        raise FileNotFoundError(f"DB not found at {db_path}")

    meta: Dict[str, Any] = {
        "db_path_exported_from": os.path.abspath(db_path),
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with _conn(db_path) as conn:
        cur = conn.cursor()
        # pull only user tables (skip sqlite_internal)
        cur.execute("""
            SELECT name, sql
            FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name;
        """)
        rows = cur.fetchall()
        tables: List[str] = [name for name, _ in rows]

        # build schema.sql: DROP TABLE IF EXISTS + CREATE TABLE ...;
        ddl_parts: List[str] = []
        for name, ddl in rows:
            if not ddl:
                continue
            ddl_clean = ddl.strip().rstrip(";") + ";"
            ddl_parts.append(f'DROP TABLE IF EXISTS {name};')
            ddl_parts.append(ddl_clean)

        schema_sql = "\n".join(ddl_parts)
        schema_sql = _ensure_semicolons_between_creates(schema_sql)
        schema_sql = _final_semicolon(schema_sql)
        (seed / "schema.sql").write_text(schema_sql, encoding="utf-8")

        # export tables to CSV
        for t in tables:
            df = pd.read_sql_query(f'SELECT * FROM "{t}"', conn)
            df.to_csv(seed / f"{t}.csv", index=False)

    meta["tables"] = tables
    (seed / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


# --- import ------------------------------------------------------------------
def import_snapshot(db_path: str, seed_dir: str = "data/seed") -> Dict[str, Any]:
    """
    Rebuild the SQLite DB from data/seed/{schema.sql, *.csv}.
    - Strips PRAGMA/BEGIN/COMMIT from schema.sql
    - Ensures semicolons between CREATE statements
    - Adds DROP TABLE IF EXISTS before CREATE TABLE
    - Executes schema with FKs OFF, then loads CSVs
    - If a CSV exists for a table missing from schema.sql, auto-creates a TEXT-column table
    Returns: {"db_path": <abs>, "imported": [{"table":..., "rows":..., "note":...}, ...]}
    """
    seed = Path(seed_dir)
    schema_file = seed / "schema.sql"
    if not schema_file.exists():
        raise FileNotFoundError(f"Missing {schema_file}. Run export first.")

    s = schema_file.read_text(encoding="utf-8")
    s = _strip_control_statements(s)
    s = _ensure_semicolons_between_creates(s)
    s = _add_drop_before_create_table(s)
    s = _final_semicolon(s)

    imported: List[Dict[str, Any]] = []

    with _conn(db_path) as conn:
        # Load schema
        conn.execute("PRAGMA foreign_keys=OFF;")
        conn.executescript(s)
        conn.execute("PRAGMA foreign_keys=ON;")

        # Load CSVs in one transaction
        with conn:
            for csv_path in sorted(seed.glob("*.csv")):
                table = csv_path.stem
                df = pd.read_csv(csv_path)

                # Does schema define this table?
                exists = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                ).fetchone() is not None

                if not exists:
                    if df.empty:
                        imported.append({"table": table, "rows": 0, "note": "skipped (missing schema, empty CSV)"})
                        continue
                    # Create from CSV columns (TEXT by default to avoid type issues)
                    col_defs = ", ".join([f'[{c}] TEXT' for c in df.columns])
                    conn.execute(f'CREATE TABLE [{table}] ({col_defs});')

                if df.empty:
                    imported.append({"table": table, "rows": 0})
                    continue

                cols = ", ".join([f"[{c}]" for c in df.columns])
                qs = ", ".join(["?" for _ in df.columns])
                conn.executemany(
                    f'INSERT INTO [{table}] ({cols}) VALUES ({qs})',
                    df.itertuples(index=False, name=None),
                )
                imported.append({"table": table, "rows": len(df)})

    return {"db_path": os.path.abspath(db_path), "imported": imported}
