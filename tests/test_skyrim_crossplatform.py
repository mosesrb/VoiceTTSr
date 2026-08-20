"""
Tests for Skyrim SE Cross-Platform Command Construction & Wine Support
"""

import sys
import pytest
from skyrim_utils import SkyrimConverter


class TestSkyrimCrossPlatform:
    @pytest.fixture
    def converter(self):
        return SkyrimConverter("mock_facefx.exe", "mock_xwma.exe", "mock_fonix.cdf")

    def test_build_command_windows(self, converter, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        cmd = converter._build_command("tools/FaceFX.exe", "Skyrim", "USEnglish")
        assert cmd == ["tools/FaceFX.exe", "Skyrim", "USEnglish"]

    def test_build_command_linux_with_wine(self, converter, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr("shutil.which", lambda bin_name: "/usr/bin/wine" if bin_name == "wine" else None)
        cmd = converter._build_command("tools/FaceFX.exe", "Skyrim", "USEnglish")
        assert cmd == ["/usr/bin/wine", "tools/FaceFX.exe", "Skyrim", "USEnglish"]

    def test_build_command_linux_without_wine(self, converter, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr("shutil.which", lambda bin_name: None)
        with pytest.raises(RuntimeError, match="requires Wine to run on linux"):
            converter._build_command("tools/FaceFX.exe", "Skyrim", "USEnglish")
