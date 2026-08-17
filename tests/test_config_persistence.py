import os
import json
import tempfile
import pytest


def save_config_atomic(filepath, data):
    """Save config atomically with temp file rename."""
    tmp = filepath + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    if os.path.exists(filepath):
        os.remove(filepath)
    os.rename(tmp, filepath)


def load_config_safe(filepath, defaults):
    """Load config safely with fallback defaults."""
    if not os.path.isfile(filepath):
        return dict(defaults)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            # Merge with defaults
            merged = dict(defaults)
            merged.update(loaded)
            return merged
    except Exception:
        return dict(defaults)


class TestConfigPersistence:
    @pytest.fixture
    def defaults(self):
        return {
            "backend": "xtts",
            "lang": "en",
            "speed": 1.0,
            "temperature": 0.75,
            "out_folder": "Output",
            "active_preset": "Natural"
        }

    def test_safe_load_nonexistent(self, defaults):
        loaded = load_config_safe("non_existent_config.json", defaults)
        assert loaded == defaults

    def test_atomic_save_and_safe_load(self, defaults):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "config.json")
            data = dict(defaults)
            data["speed"] = 1.25
            data["active_preset"] = "Whisper"

            save_config_atomic(cfg_path, data)
            assert os.path.exists(cfg_path)

            loaded = load_config_safe(cfg_path, defaults)
            assert loaded["speed"] == 1.25
            assert loaded["active_preset"] == "Whisper"
            assert loaded["backend"] == "xtts"

    def test_corrupted_config_fallback(self, defaults):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "corrupted.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                f.write("{ INVALID JSON DATA ...")

            loaded = load_config_safe(cfg_path, defaults)
            assert loaded == defaults
