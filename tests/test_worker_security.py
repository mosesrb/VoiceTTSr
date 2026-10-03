"""
Tests for Voice Profile and Model Checkpoint Deserialization Security Gates
Verifies safetensors loading, weights_only enforcement, and rejection of unsafe unpickling across workers.
"""

import os
import sys
import types
import tempfile
import pytest
import torch
import safetensors.torch
import rvc_worker


class _MockDictionary:
    """Module-level mock fairseq Dictionary for safe_globals serialization tests."""
    pass

_MockDictionary.__module__ = "fairseq.data.dictionary"
_MockDictionary.__name__ = "Dictionary"
_MockDictionary.__qualname__ = "Dictionary"


class _HostilePayload:
    """Module-level simulated malicious payload attempting arbitrary system command execution."""
    def __reduce__(self):
        return (os.system, ("echo hostile_execution_attempt",))


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
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = os.path.join(tmpdir, "exploit.pth")
            # Save object via raw torch.save (pickle)
            torch.save({"payload": _HostilePayload()}, profile_path)

            # weights_only=True must raise an exception and prevent execution
            with pytest.raises(Exception):
                torch.load(profile_path, weights_only=True)

    def test_rvc_verify_safe_checkpoint_valid(self):
        """Verify rvc_worker.verify_rvc_checkpoint accepts valid tensor dictionaries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "valid_rvc.pth")
            valid_weights = {
                "weight": {"enc_p.emb_phone.weight": torch.randn(256, 192)},
                "config": [192, 192, 768, 2, 6, 3, 0],
                "f0": 1,
                "version": "v2"
            }
            torch.save(valid_weights, model_path)
            cpt = rvc_worker.verify_rvc_checkpoint(model_path)
            assert isinstance(cpt, dict)
            assert "weight" in cpt
            assert "config" in cpt

    def test_rvc_verify_safe_checkpoint_existing_baselines(self):
        """Verify pre-existing baseline RVC models pass verify_rvc_checkpoint."""
        baseline = os.path.join("rvc_models", "female_baseline.pth")
        if os.path.isfile(baseline):
            cpt = rvc_worker.verify_rvc_checkpoint(baseline)
            assert isinstance(cpt, dict)
            assert "model" in cpt

    def test_rvc_verify_safe_checkpoint_rejects_malicious_payload(self):
        """Verify rvc_worker.verify_rvc_checkpoint rejects hostile pickle payloads."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exploit_path = os.path.join(tmpdir, "hostile_model.pth")
            torch.save({"exploit": _HostilePayload()}, exploit_path)

            with pytest.raises(ValueError, match="Security rejection.*failed safe tensor verification"):
                rvc_worker.verify_rvc_checkpoint(exploit_path)

    def test_rvc_verify_safe_checkpoint_missing_file(self):
        """Verify rvc_worker.verify_rvc_checkpoint raises FileNotFoundError for nonexistent paths."""
        with pytest.raises(FileNotFoundError):
            rvc_worker.verify_rvc_checkpoint("nonexistent_rvc_model.pth")

    def test_rvc_secure_torch_load_default(self):
        """Verify rvc_worker patches torch.load to default to weights_only=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exploit_path = os.path.join(tmpdir, "default_exploit.pth")
            torch.save({"payload": _HostilePayload()}, exploit_path)

            # Omitting weights_only argument must still reject the exploit
            with pytest.raises(Exception):
                torch.load(exploit_path, map_location="cpu")

    def test_hubert_safe_globals_allows_dictionary_blocks_hostile(self):
        """Verify Hubert loader allowlists Dictionary safely while blocking hostile classes."""
        if hasattr(torch.serialization, "add_safe_globals"):
            DictClass = sys.modules["fairseq.data.dictionary"].Dictionary

            with tempfile.TemporaryDirectory() as tmpdir:
                safe_path = os.path.join(tmpdir, "safe_dict.pt")
                hostile_path = os.path.join(tmpdir, "hostile_dict.pt")

                torch.save({"dict": DictClass(), "weight": torch.randn(10, 10)}, safe_path)
                torch.save({"dict": DictClass(), "hostile": _HostilePayload()}, hostile_path)

                # Safe Dictionary should load cleanly under weights_only=True
                loaded = torch.load(safe_path, map_location="cpu", weights_only=True)
                assert "dict" in loaded
                assert "weight" in loaded

                # Hostile payload must be rejected
                with pytest.raises(Exception):
                    torch.load(hostile_path, map_location="cpu", weights_only=True)

    def test_hubert_base_pt_loads_with_weights_only(self):
        """Verify actual hubert_base.pt loads cleanly under weights_only=True."""
        hubert_path = os.path.join("rvc_models", "hubert_base.pt")
        if not os.path.isfile(hubert_path):
            pytest.skip("hubert_base.pt not found on disk")
        ckpt = torch.load(hubert_path, map_location="cpu", weights_only=True)
        assert isinstance(ckpt, dict)
        assert "model" in ckpt or "args" in ckpt
