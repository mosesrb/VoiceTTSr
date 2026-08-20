"""
Tests for Voice Profile Deserialization Security Gates
Verifies safetensors loading, weights_only enforcement, and rejection of unsafe unpickling.
"""

import os
import tempfile
import pytest
import torch
import safetensors.torch


class TestWorkerSecurity:
    def test_safetensors_profile_creation_and_loading(self):
        """Verify safetensors tensors load cleanly without unpickling."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = os.path.join(tmpdir, "test_voice.safetensors")
            tensors = {
                "gpt_cond_latent": torch.randn(1, 32, 80),
                "speaker_embedding": torch.randn(1, 512)
            }
            safetensors.torch.save_file(tensors, profile_path)

            loaded = safetensors.torch.load_file(profile_path)
            assert "gpt_cond_latent" in loaded
            assert "speaker_embedding" in loaded
            assert loaded["gpt_cond_latent"].shape == (1, 32, 80)

    def test_legacy_weights_only_safe_pth(self):
        """Verify pure tensor dictionaries load safely under weights_only=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = os.path.join(tmpdir, "legacy_voice.pth")
            tensors = {
                "gpt_cond_latent": torch.randn(1, 32, 80),
                "speaker_embedding": torch.randn(1, 512)
            }
            torch.save(tensors, profile_path)

            # Loading with weights_only=True must succeed for pure numerical tensors
            loaded = torch.load(profile_path, weights_only=True)
            assert "gpt_cond_latent" in loaded
            assert "speaker_embedding" in loaded

    def test_unsafe_pickle_payload_rejected_by_weights_only(self):
        """Verify non-numerical executable objects are rejected by weights_only=True."""
        class MaliciousPayload:
            def __reduce__(self):
                return (os.system, ("echo unsafe",))

        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = os.path.join(tmpdir, "exploit.pth")
            # Save object via raw torch.save (pickle)
            torch.save({"payload": MaliciousPayload()}, profile_path)

            # weights_only=True must raise an exception and prevent execution
            with pytest.raises(Exception):
                torch.load(profile_path, weights_only=True)
