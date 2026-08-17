import os
import struct
import tempfile
import pytest
from skyrim_utils import SkyrimConverter


class TestSkyrimUtils:
    @pytest.fixture
    def converter(self):
        return SkyrimConverter("mock_facefx.exe", "mock_xwma.exe", "mock_fonix.cdf")

    def test_sanitize_dialogue_text_normal(self, converter):
        clean = converter.sanitize_dialogue_text("Hello dragonborn! Let's go.")
        assert clean == "Hello dragonborn! Let's go."

    def test_sanitize_dialogue_text_newlines_and_tabs(self, converter):
        raw = "Line 1\nLine 2\r\nLine 3\tTabbed"
        clean = converter.sanitize_dialogue_text(raw)
        assert clean == "Line 1 Line 2 Line 3 Tabbed"

    def test_sanitize_dialogue_text_empty_or_whitespace(self, converter):
        assert converter.sanitize_dialogue_text("") == "..."
        assert converter.sanitize_dialogue_text("   \n\t  ") == "..."
        assert converter.sanitize_dialogue_text(None) == "..."

    def test_pack_fuz_binary_structure(self, converter):
        with tempfile.TemporaryDirectory() as tmpdir:
            lip_path = os.path.join(tmpdir, "test.lip")
            xwm_path = os.path.join(tmpdir, "test.xwm")
            fuz_path = os.path.join(tmpdir, "test.fuz")

            lip_data = b"FACEFX_LIP_DATA_12345"
            xwm_data = b"XWMA_ENCODED_AUDIO_STREAM_67890"

            with open(lip_path, "wb") as f:
                f.write(lip_data)
            with open(xwm_path, "wb") as f:
                f.write(xwm_data)

            result = converter.pack_fuz(lip_path, xwm_path, fuz_path)
            assert os.path.exists(result)

            with open(fuz_path, "rb") as f:
                # Read 12-byte header: 4s II
                magic, version, lip_length = struct.unpack('<4sII', f.read(12))
                assert magic == b'FUZE'
                assert version == 1
                assert lip_length == len(lip_data)

                read_lip = f.read(lip_length)
                assert read_lip == lip_data

                read_xwm = f.read()
                assert read_xwm == xwm_data
