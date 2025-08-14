# draupnir_core/db_config.py
from __future__ import annotations
import json, os, shutil
from pathlib import Path
from typing import Optional

# Local-only config (per machine). We'll ignore this in .gitignore.
_LOCAL_CFG = Path("data/db_local.json")
_LOCAL_DEFAULT = Path("data/draupnir.db")

def _find_dropbox_root() -> Optional[Path]:
    """
    Find Dropbox root (Windows/Mac) from info.json.
    Handles personal or business accounts.
    """
    candidates = [
        Path.home() / "Dropbox" / "info.json",
        Path(os.environ.get("APPDATA", "")) / "Dropbox" / "info.json",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Dropbox" / "info.json",
    ]
    for p in candidates:
        if p and p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                # try personal then business
                for key in ("personal", "business"):
                    root = data.get(key, {}).get("path")
                    if root:
                        return Path(root)
            except Exception:
                pass
    # Standard install sometimes is just ~/Dropbox without info.json
    default = Path.home() / "Dropbox"
    return default if default.exists() else None

def _resolve_db_path(cfg_value: str) -> Path:
    """
    Resolve stored cfg string into an absolute Path.
    Supports:
      - absolute paths
      - user paths (~)
      - dropbox://relative/inside/dropbox/draupnir.db
    """
    if cfg_value.startswith("dropbox://"):
        rel = cfg_value.replace("dropbox://", "", 1).lstrip("/\\")
        root = _find_dropbox_root()
        if not root:
            raise FileNotFoundError("Dropbox root not found on this machine.")
        return (root / rel).resolve()
    return Path(cfg_value).expanduser().resolve()

def get_db_path() -> str:
    """
    Return the absolute path to the DB for THIS machine.
    Falls back to ./data/draupnir.db if no local config exists.
    """
    if _LOCAL_CFG.exists():
        try:
            cfg = json.loads(_LOCAL_CFG.read_text(encoding="utf-8"))
            path_str = cfg.get("db_path", "")
            if path_str:
                return str(_resolve_db_path(path_str))
        except Exception:
            pass
    return str(_LOCAL_DEFAULT.resolve())

def set_db_path_local(path_or_dropbox_uri: str, copy_if_needed: bool = True) -> str:
    """
    Save a local-only DB path for THIS machine.
    path_or_dropbox_uri:
      - absolute path like N:\\Dropbox\\Apps\\draupnir\\data\\draupnir.db
      - or dropbox://Apps/draupnir/data/draupnir.db
    If copy_if_needed=True and target doesn't exist, copy the current local DB there once.
    Returns the resolved absolute path that will be used.
    """
    # Ensure data/ exists
    _LOCAL_CFG.parent.mkdir(parents=True, exist_ok=True)

    # Resolve target for validation/copy
    target = _resolve_db_path(path_or_dropbox_uri)
    target.parent.mkdir(parents=True, exist_ok=True)

    # copy current local DB if asked and target file missing but local exists
    if copy_if_needed and not target.exists():
        local = _LOCAL_DEFAULT.resolve()
        if local.exists() and str(local) != str(target):
            shutil.copy2(local, target)

    # Save config
    _LOCAL_CFG.write_text(json.dumps({"db_path": path_or_dropbox_uri}, indent=2), encoding="utf-8")
    return str(target)

def clear_db_path_local() -> None:
    """
    Remove local override so the app uses ./data/draupnir.db on THIS machine.
    """
    try:
        _LOCAL_CFG.unlink()
    except FileNotFoundError:
        pass
