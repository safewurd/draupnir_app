# drop_legacy_items.py  (edit list first!)
import sqlite3
from draupnir_core.db_config import get_db_path

DB = get_db_path()
TO_DROP = [
    "Projections", "AfterTaxProjections", "FXRates", "Markets",
    "GlobalSettings", "Trades_old", "Portfolios_old", "IncomeSources",
    # If you’re fully migrated:
    # "Assets", "ProjectionInputs",
    # If you keep only lowercase:
    # "Institutions", "TaxTreatments",
]

conn = sqlite3.connect(DB)
cur = conn.cursor()
for t in TO_DROP:
    try:
        cur.execute(f'DROP TABLE IF EXISTS "{t}"')
        print(f"Dropped: {t}")
    except Exception as e:
        print(f"Skip {t}: {e}")
conn.commit()
conn.close()
