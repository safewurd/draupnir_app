import sqlite3

# === Default configuration values ===
DEFAULTS = {
    "base_currency": "CAD",
    "fx_rate_t0": 1.37,
    "fx_mode": "fixed",              # fixed = use t0 rate, can add "standard" if needed
    "inflation_mode": "standard",    # standard = adjust contributions/withdrawals
    "default_inflation": 0.03,
    "growth_mode": "nominal",
    "market_data_provider": "yahoo",
    "macro_mode": "forecast",        # ✅ new setting
    "last_trades_hash": ""
}

def load_global_settings(conn):
    """
    Loads all settings from the GlobalSettings table into a dictionary.
    Uses defaults if a setting is missing or invalid.
    Converts numeric settings where appropriate.
    """
    settings = DEFAULTS.copy()
    try:
        cur = conn.cursor()
        cur.execute("SELECT setting_name, setting_value FROM GlobalSettings")
        rows = cur.fetchall()
        for name, value in rows:
            if name not in settings:
                continue  # Ignore unknown keys

            if name in ["fx_rate_t0", "default_inflation"]:
                try:
                    settings[name] = float(value)
                except (ValueError, TypeError):
                    print(f"[WARN] Invalid numeric value for {name}, using default.")
            else:
                settings[name] = value
    except sqlite3.Error as e:
        print(f"[WARN] Could not load settings from DB: {e}")

    # Normalize modes
    settings["fx_mode"] = settings.get("fx_mode", "fixed").lower()
    settings["inflation_mode"] = settings.get("inflation_mode", "standard").lower()
    
    # ✅ Normalize macro_mode
    mode = settings.get("macro_mode", "forecast").lower()
    settings["macro_mode"] = "forecast" if mode not in ["forecast", "fixed"] else mode

    return settings

def get_setting(conn, name, default=None):
    """
    Fetch a single setting by name with optional default fallback.
    """
    try:
        cur = conn.cursor()
        cur.execute("SELECT setting_value FROM GlobalSettings WHERE setting_name=?", (name,))
        row = cur.fetchone()
        if row:
            if name in ["fx_rate_t0", "default_inflation"]:
                try:
                    return float(row[0])
                except ValueError:
                    return default if default is not None else DEFAULTS.get(name)
            return row[0]
    except sqlite3.Error:
        pass
    return default if default is not None else DEFAULTS.get(name)

def get_fx_rate_t0(conn):
    """Returns the fx_rate_t0 from settings as a float."""
    return get_setting(conn, "fx_rate_t0", DEFAULTS["fx_rate_t0"])
