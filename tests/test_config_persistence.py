import os
import json
import tempfile
import pytest

from core.config import save_config_atomic, load_config_safe
from core import save_config_atomic as save_atomic_pkg, load_config_safe as load_safe_pkg


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

    def test_package_exports(self):
        assert save_config_atomic is save_atomic_pkg
        assert load_config_safe is load_safe_pkg

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

    def test_non_dict_json_fallback(self, defaults):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "array.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(["not", "a", "dict"], f)

            loaded = load_config_safe(cfg_path, defaults)
            assert loaded == defaults

    def test_atomic_save_nested_directory_creation(self, defaults):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "nested", "subfolder", "config.json")
            save_config_atomic(cfg_path, defaults)
            assert os.path.exists(cfg_path)
            loaded = load_config_safe(cfg_path, defaults)
            assert loaded == defaults
