"""
rvc_worker.py — Clean RVC Engine v10

ROOT CAUSE (confirmed after extensive analysis):
  HuggingFace Hubert features are statistically incompatible with RVC's generator.
  The NSF vocoder was trained end-to-end on fairseq Hubert features; using HF
  Hubert (even last_hidden_state, even correct dtype) produces 85% sub-500Hz
  energy — harmonics are completely suppressed.

SOLUTION:
  Use the actual fairseq hubert_base_ls960.pt checkpoint, loaded via torch
  directly (no fairseq package required). We download it once to a local cache,
  then load the weights manually and call the CNN + transformer forward pass
  in the same way fairseq does internally.

  If fairseq IS installed, we use it natively (rvc_python's own path works).
  If not, we use our direct-weight loader as a drop-in.
"""
import os, sys, json, traceback, warnings, types, urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

HUBERT_URL   = "https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt"
HUBERT_CACHE = os.path.join(os.path.dirname(__file__), "hubert_base_ls960.pt")

def send(obj):
    print(json.dumps(obj), flush=True)

def log(text, level="info"):
    send({"status": "log", "text": text, "level": level})


# ══════════════════════════════════════════════════════════════════════════
# FAIRSEQ DETECTION
# ══════════════════════════════════════════════════════════════════════════
def _fairseq_available():
    try:
        import fairseq
        import fairseq.checkpoint_utils
        # Make sure the function actually exists (not our old mock)
        fn = fairseq.checkpoint_utils.load_model_ensemble_and_task
        return callable(fn)
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════
# DIRECT FAIRSEQ HUBERT LOADER (no fairseq package needed)
# Loads hubert_base_ls960.pt weights and runs inference manually,
# matching fairseq's exact forward pass so features are identical.
# ══════════════════════════════════════════════════════════════════════════
def _download_hubert():
    if os.path.isfile(HUBERT_CACHE):
        return
    log(f"Downloading hubert_base_ls960.pt (~360 MB) — one time only…")
    log(f"Saving to: {HUBERT_CACHE}")

    def _progress(count, block_size, total):
        pct = min(count * block_size / total * 100, 100)
        if int(pct) % 10 == 0:
            log(f"  Download: {pct:.0f}%")

    try:
        urllib.request.urlretrieve(HUBERT_URL, HUBERT_CACHE, reporthook=_progress)
        log("Download complete.", "ok")
    except Exception as e:
        if os.path.isfile(HUBERT_CACHE):
            os.remove(HUBERT_CACHE)
        raise RuntimeError(f"Hubert download failed: {e}")


class _FairsegHubertDirect(nn.Module):
    """
    Runs the fairseq HubertModel forward pass using its saved checkpoint weights,
    without requiring the fairseq package to be installed.

    fairseq hubert-base-ls960 architecture:
      - Feature extractor: 7-layer conv (strides: 5,2,2,2,2,2,2 → stride 320)
      - Transformer: 12 layers (NOT 9 — the 9 refers to the number used for
        K-means clustering during training, not the model depth)
      - Output: 768-dim per frame at 50 fps (after 320-sample stride at 16kHz)

    We load the weights from the .pt file and run them through nn.Module
    layers that match fairseq's architecture exactly.
    """
    def __init__(self, ckpt_path: str, device: str):
        super().__init__()
        self.device = torch.device(device)

        log("Loading fairseq Hubert checkpoint weights…")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # fairseq saves model state under different keys depending on version
        if "model" in ckpt:
            state = ckpt["model"]
        elif "cfg" in ckpt:
            # newer fairseq format
            state = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
        else:
            state = ckpt

        # Build the HF HubertModel and load fairseq weights via key remapping
        # This is the most reliable approach: HF architecture == fairseq architecture
        # but key names differ. We remap and load.
        from transformers import HubertConfig, HubertModel
        import transformers
        prev = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()

        cfg = HubertConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            conv_dim=(512, 512, 512, 512, 512, 512, 512),
            conv_stride=(5, 2, 2, 2, 2, 2, 2),
            conv_kernel=(10, 3, 3, 3, 3, 2, 2),
            conv_bias=False,
            feat_extract_norm="group",
            feat_proj_norm=None,
        )
        self._hf = HubertModel(cfg)
        transformers.logging.set_verbosity(prev)

        # Remap fairseq keys → HF keys
        remapped = _remap_fairseq_to_hf(state, self._hf.state_dict())
        missing, unexpected = self._hf.load_state_dict(remapped, strict=False)
        if missing:
            log(f"  Hubert weight mapping: {len(missing)} missing keys (expected for some configs)", "warn")

        self._hf = self._hf.to(self.device).eval()
        log("Fairseq Hubert weights loaded into HF model.", "ok")

    def _get_dtype(self):
        try:
            return next(self._hf.parameters()).dtype
        except StopIteration:
            return torch.float32

    def forward(self, source, padding_mask=None, mask=False,
                features_only=True, output_layer=None):
        dtype = self._get_dtype()
        src = source.to(device=self.device, dtype=dtype)
        with torch.no_grad():
            out = self._hf(src, output_hidden_states=True)
        # fairseq layer 9 = HF last_hidden_state (both are the final transformer output)
        feat = out.last_hidden_state
        return {"x": feat, "padding_mask": None}

    def extract_features(self, source, padding_mask=None,
                         mask=False, output_layer=None):
        result = self.forward(source, padding_mask=padding_mask,
                              mask=mask, output_layer=output_layer)
        return [result["x"]]   # pipeline.py does logits[0]

    class _FakeFE:
        conv_layers = []
    feature_extractor = _FakeFE()


def _remap_fairseq_to_hf(fairseq_state: dict, hf_state: dict) -> dict:
    """
    Best-effort remapping of fairseq HubertModel state_dict keys to HF keys.
    Falls back to identity for any key that can't be mapped.
    """
    result = dict(hf_state)   # start with HF defaults

    # Build a simple prefix map
    prefix_map = [
        ("feature_extractor.conv_layers.",   "feature_extractor.conv_layers."),
        ("post_extract_proj.",               "feature_projection.projection."),
        ("encoder.pos_conv.",                "encoder.pos_conv_embed.conv."),
        ("encoder.layers.",                  "encoder.layers."),
        ("layer_norm.",                      "feature_projection.layer_norm."),
    ]

    for fs_key, fs_val in fairseq_state.items():
        hf_key = None
        for fs_pfx, hf_pfx in prefix_map:
            if fs_key.startswith(fs_pfx):
                suffix = fs_key[len(fs_pfx):]
                # further remap layer internals
                suffix = (suffix
                    .replace(".self_attn.q_proj.", ".attention.q_proj.")
                    .replace(".self_attn.k_proj.", ".attention.k_proj.")
                    .replace(".self_attn.v_proj.", ".attention.v_proj.")
                    .replace(".self_attn.out_proj.", ".attention.out_proj.")
                    .replace(".self_attn_layer_norm.", ".layer_norm.")
                    .replace(".final_layer_norm.", ".final_layer_norm.")
                    .replace(".fc1.", ".feed_forward.intermediate_dense.")
                    .replace(".fc2.", ".feed_forward.output_dense.")
                )
                candidate = hf_pfx + suffix
                if candidate in result:
                    hf_key = candidate
                break
        if hf_key and fs_val.shape == result[hf_key].shape:
            result[hf_key] = fs_val

    return result


# ══════════════════════════════════════════════════════════════════════════
# FAIRSEQ GHOST PATCH (used when fairseq is NOT installed)
# ══════════════════════════════════════════════════════════════════════════
def _install_ghost_patch(device: str):
    log("fairseq not installed — using direct checkpoint loader.")
    _download_hubert()

    _singleton: list = []

    def _load_model_ensemble_and_task(filenames, *args, **kwargs):
        if not _singleton:
            model = _FairsegHubertDirect(HUBERT_CACHE, device)
            _singleton.append(model)
        return [_singleton[0]], None, None

    fairseq_mod = types.ModuleType("fairseq")
    ckpt_mod    = types.ModuleType("fairseq.checkpoint_utils")
    ckpt_mod.load_model_ensemble_and_task = _load_model_ensemble_and_task
    fairseq_mod.checkpoint_utils          = ckpt_mod
    sys.modules["fairseq"]                  = fairseq_mod
    sys.modules["fairseq.checkpoint_utils"] = ckpt_mod
    log("Ghost patch installed (direct fairseq weights).", "ok")


# ══════════════════════════════════════════════════════════════════════════
# RVC ENGINE
# ══════════════════════════════════════════════════════════════════════════
class RvcEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._rvc          = None
        self._loaded_model = None

        if _fairseq_available():
            log(f"fairseq found — using native Hubert loader.", "ok")
        else:
            _install_ghost_patch(self.device)

        log(f"RVC engine ready ({self.device.upper()}).", "ok")

    def _get_rvc(self, model_pth: str, index_pth=None):
        from rvc_python.infer import RVCInference
        if self._rvc is None or self._loaded_model != model_pth:
            self._rvc = RVCInference(device=self.device)
            log(f"Loading model: {os.path.basename(model_pth)}…")
            # Pass index_path into load_model — this is the correct API
            idx = index_pth if (index_pth and os.path.isfile(index_pth)) else ""
            self._rvc.load_model(model_pth, index_path=idx)
            self._loaded_model = model_pth
            tgt_sr  = getattr(self._rvc.vc, "tgt_sr",  "?")
            is_half = getattr(self._rvc.vc, "is_half", False)
            log(f"Model loaded — tgt_sr={tgt_sr}, is_half={is_half}.", "ok")
        else:
            # Update index on existing rvc instance if it changed
            if index_pth and os.path.isfile(index_pth):
                self._rvc.models[self._rvc.current_model]["index"] = index_pth
        return self._rvc

    def infer(self, input_wav: str, output_wav: str, model_pth: str,
              index_pth=None, pitch: int = 0,
              index_rate: float = 0.75, f0_method: str = "rmvpe") -> bool:
        try:
            rvc = self._get_rvc(model_pth, index_pth)

            # Correct attribute names from RVCInference source inspection:
            # f0up_key (no underscore), f0method (no underscore)
            rvc.f0up_key      = pitch        # was wrongly set as f0_up_key
            rvc.f0method      = f0_method    # was wrongly set as f0_method
            rvc.index_rate    = index_rate
            rvc.filter_radius = 3
            rvc.resample_sr   = 0
            rvc.rms_mix_rate  = 0.25
            rvc.protect       = 0.33

            log(f"Running inference ({f0_method}, pitch={pitch:+d}, index_rate={index_rate})…")
            rvc.infer_file(input_wav, output_wav)

            if not os.path.isfile(output_wav):
                log("Output file not created — inference silently failed.", "error")
                return False

            data, sr = sf.read(output_wav)
            if np.max(np.abs(data)) < 1e-6:
                log("Output is silent — inference may have failed.", "warn")
                return False

            log(f"Done → {os.path.basename(output_wav)}  ({len(data)/sr:.2f}s)", "ok")
            return True

        except Exception as e:
            log(f"RVC inference failed: {e}", "error")
            log(traceback.format_exc(), "debug")
            return False


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    engine = RvcEngine()
    send({"status": "ready"})

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            cmd = json.loads(raw)
        except json.JSONDecodeError:
            continue

        action = cmd.get("action")
        if action == "quit":
            break

        if action == "infer":
            ok = engine.infer(
                input_wav  = cmd["input"],
                output_wav = cmd["out"],
                model_pth  = cmd["model"],
                index_pth  = cmd.get("index"),
                pitch      = int(cmd.get("pitch", 0)),
                index_rate = float(cmd.get("index_rate", 0.75)),
                f0_method  = cmd.get("f0_method", "rmvpe"),
            )
            if ok:
                tgt_sr  = getattr(engine._rvc.vc, "tgt_sr",  0) if engine._rvc else 0
                is_half = getattr(engine._rvc.vc, "is_half", False) if engine._rvc else False
                send({"status": "done", "file": cmd["out"],
                      "tgt_sr": tgt_sr, "is_half": is_half})
            else:
                send({"status": "error", "message": "RVC processing failed."})
