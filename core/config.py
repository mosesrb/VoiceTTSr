"""
VoiceTTSr Configuration Persistence
Atomic writes and robust fallback loading to prevent config corruption.
"""

import json
import os
from typing import Dict, Any


def save_config_atomic(filepath: str, data: Dict[str, Any]) -> None:
    """Save config atomically with temp file write and atomic replace."""
    tmp = filepath + ".tmp"
    dir_name = os.path.dirname(filepath)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    # On Windows, os.replace handles atomic overwrite if on same filesystem
    if os.path.exists(filepath):
        try:
            os.replace(tmp, filepath)
        except OSError:
            # Fallback for systems/filesystems where replace fails on existing destination
            os.remove(filepath)
            os.rename(tmp, filepath)
    else:
        os.rename(tmp, filepath)


def load_config_safe(filepath: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Load config safely with fallback defaults on missing or corrupt files."""
    if not os.path.isfile(filepath):
        return dict(defaults)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            merged = dict(defaults)
            if isinstance(loaded, dict):
                merged.update(loaded)
            return merged
    except Exception:
        return dict(defaults)
