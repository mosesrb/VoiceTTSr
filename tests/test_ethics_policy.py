"""
Tests for VoiceTTSr Ethical Use, Terms & Privacy Policy System
"""

import os
import json
import tempfile
import pytest
from ui.components.ethics_dialog import (
    _load_doc_file,
    PRIVACY_POLICY_TEXT,
    _VOICE_ETHICS_PATH,
    _THIRD_PARTY_PATH,
)


class TestEthicsPolicy:
    def test_voice_ethics_doc_exists_and_loads(self):
        """Verify docs/VOICE_ETHICS.md exists and loads content properly."""
        assert os.path.isfile(_VOICE_ETHICS_PATH), f"Missing {_VOICE_ETHICS_PATH}"
        content = _load_doc_file(_VOICE_ETHICS_PATH, "Voice Ethics")
        assert "Voice Ethics & Acceptable Use" in content
        assert "No real person's voice ships as a default" in content

    def test_third_party_notices_doc_exists_and_loads(self):
        """Verify docs/THIRD_PARTY_NOTICES.md exists and loads content properly."""
        assert os.path.isfile(_THIRD_PARTY_PATH), f"Missing {_THIRD_PARTY_PATH}"
        content = _load_doc_file(_THIRD_PARTY_PATH, "Third-Party Notices")
        assert "Third-Party Notices" in content
        assert "Coqui Public Model License" in content

    def test_privacy_policy_doc_exists_and_loads(self):
        """Verify docs/PRIVACY_POLICY.md exists and loads content properly."""
        from ui.components.ethics_dialog import _PRIVACY_PATH
        assert os.path.isfile(_PRIVACY_PATH), f"Missing {_PRIVACY_PATH}"
        content = _load_doc_file(_PRIVACY_PATH, "Privacy Policy")
        assert "Local Privacy Policy" in content
        assert "100% Offline & Local Execution" in content

    def test_privacy_policy_guarantees(self):
        """Verify privacy policy text explicitly defines offline, local execution guarantees."""
        assert "100% Offline & Local Execution" in PRIVACY_POLICY_TEXT
        assert "Zero Telemetry & Zero Cloud Transmission" in PRIVACY_POLICY_TEXT
        assert "uploaded or transmitted to any external server" in PRIVACY_POLICY_TEXT

    def test_missing_doc_fallback(self):
        """Verify missing doc paths return descriptive fallback text rather than crashing."""
        content = _load_doc_file("nonexistent_path/dummy.md", "Fallback Title")
        assert "# Fallback Title" in content
        assert "File not found" in content

    def test_ethics_accepted_config_persistence(self):
        """Verify ethics_accepted boolean serializes and deserializes accurately."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "config.json")
            cfg = {"backend": "xtts", "ethics_accepted": True}
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f)

            with open(cfg_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            assert loaded.get("ethics_accepted") is True
