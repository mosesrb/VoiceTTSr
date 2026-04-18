# Copyright (c) 2026 mosesrb (Moses Bharshankar). Licensed under GNU GPL-v3.
"""
chatterbox_worker.py  —  VoiceTTSr subprocess worker for Chatterbox TTS
v1.2 — patches internal n_timesteps to cap diffusion steps (1000 -> 40)
"""

import sys, json, os, traceback

def _send(obj: dict):
    print(json.dumps(obj), flush=True)

def _log(text, level="info"):
    _send({"status": "log", "level": level, "text": text})

_log("Chatterbox worker: importing dependencies…", "warn")

try:
    import torch
    import torchaudio as ta
    from chatterbox.tts import ChatterboxTTS
    _log("Dependencies imported OK.", "ok")
except ImportError as e:
    _send({"status": "error", "message": f"Import failed: {e}\nRun: pip install chatterbox-tts torchaudio"})
    sys.exit(1)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
_log(f"Using device: {DEVICE}", "warn")

_log("Loading Chatterbox model (first run downloads ~1.5 GB)…", "warn")
try:
    MODEL = ChatterboxTTS.from_pretrained(device=DEVICE)
    _log("Chatterbox model loaded!", "ok")
except Exception as e:
    _send({"status": "error", "message": f"Model load failed: {e}"})
    sys.exit(1)

# ── Discover where n_timesteps lives and how to patch it ──────────────────
def _patch_steps(n: int):
    """
    Chatterbox uses a flow-matching decoder. The step count is stored as
    n_timesteps on the internal flow/decoder model. We try every known
    attribute path used across versions of the library.
    Returns True if patched, False if not found (will still run, just slow).
    """
    targets = [
        # v0.1.x paths
        lambda m: setattr(m.flow, 'n_timesteps', n),
        lambda m: setattr(m.decoder, 'n_timesteps', n),
        lambda m: setattr(m.flow.estimator, 'n_timesteps', n),
        lambda m: setattr(m.s3gen, 'n_timesteps', n),
        lambda m: setattr(m.s3gen.flow, 'n_timesteps', n),
        lambda m: setattr(m.t3_cfg, 'n_timesteps', n),
        # also try the synthesizer sub-object
        lambda m: setattr(m.synthesizer, 'n_timesteps', n) if hasattr(m, 'synthesizer') else None,
    ]
    patched = False
    for fn in targets:
        try:
            fn(MODEL)
            patched = True
        except (AttributeError, TypeError):
            pass
    # Deep scan — walk all submodules looking for n_timesteps attribute
    if not patched:
        for name, module in MODEL.named_modules() if hasattr(MODEL, 'named_modules') else []:
            if hasattr(module, 'n_timesteps'):
                try:
                    module.n_timesteps = n
                    patched = True
                    _log(f"Patched n_timesteps={n} on {name}", "info")
                except Exception:
                    pass
        # Also check direct attributes of MODEL
        for attr in dir(MODEL):
            if attr.startswith('_'): continue
            try:
                obj = getattr(MODEL, attr)
                if hasattr(obj, 'n_timesteps'):
                    obj.n_timesteps = n
                    patched = True
                    _log(f"Patched n_timesteps={n} on MODEL.{attr}", "info")
            except Exception:
                pass
    return patched

# Apply default step cap immediately after load
_DEFAULT_STEPS = 40
_patched = _patch_steps(_DEFAULT_STEPS)
if _patched:
    _log(f"Diffusion steps capped to {_DEFAULT_STEPS} (was 1000).", "ok")
else:
    _log("Could not find n_timesteps to patch — generation may be slow (1000 steps).", "warn")

_send({"status": "ready"})

# ── in-memory ref cache ────────────────────────────────────────────────────
_cached_ref_path = None
_cached_cond     = None

def _resolve_cond(audio_prompt_path):
    global _cached_ref_path, _cached_cond
    try:
        if audio_prompt_path == _cached_ref_path and _cached_cond is not None:
            _log("Using cached voice conditioning.", "info")
            return _cached_cond, None
        _log(f"Computing conditioning: {os.path.basename(audio_prompt_path)}", "info")
        cond = MODEL.get_conditioning(audio_prompt_path)
        _cached_ref_path = audio_prompt_path
        _cached_cond     = cond
        return cond, None
    except AttributeError:
        return None, audio_prompt_path

# ── command loop ───────────────────────────────────────────────────────────
for raw in sys.stdin:
    raw = raw.strip()
    if not raw:
        continue
    try:
        cmd = json.loads(raw)
    except json.JSONDecodeError:
        _log(f"Bad JSON: {raw}", "error")
        continue

    action = cmd.get("action", "")

    if action == "quit":
        _log("Worker shutting down.", "info")
        break

    elif action == "save_profile":
        refs     = cmd.get("refs", [])
        out_path = cmd.get("out_path", "cb_profile.cbprof")
        try:
            ref = next((r for r in refs if os.path.isfile(r)), None)
            if not ref:
                raise ValueError("No valid reference WAV found.")
            _log(f"Building profile from: {os.path.basename(ref)}", "info")
            try:
                cond = MODEL.get_conditioning(ref)
                torch.save({"version": 1, "ref_name": os.path.basename(ref), "cond": cond}, out_path)
                _cached_ref_path = ref
                _cached_cond     = cond
                _log(f"Profile saved → {os.path.basename(out_path)}", "ok")
                _send({"status": "done_save", "file": out_path})
            except AttributeError:
                torch.save({"version": 0, "ref_path": ref, "ref_name": os.path.basename(ref)}, out_path)
                _log(f"Profile saved (ref-path mode) → {os.path.basename(out_path)}", "ok")
                _send({"status": "done_save", "file": out_path})
        except Exception:
            err = traceback.format_exc()
            _log(err, "error")
            _send({"status": "error", "message": err})

    elif action == "load_profile":
        profile_path = cmd.get("profile_path", "")
        try:
            if not os.path.isfile(profile_path):
                raise FileNotFoundError(f"Profile not found: {profile_path}")
            data = torch.load(profile_path, map_location=DEVICE, weights_only=False)
            if data.get("version", 0) >= 1:
                _cached_cond     = data["cond"]
                _cached_ref_path = profile_path
                _log(f"Profile loaded: {data.get('ref_name','?')} (conditioning)", "ok")
            else:
                _cached_ref_path = None
                _cached_cond     = None
                _log(f"Profile loaded: {data.get('ref_name','?')} (ref-path mode)", "ok")
            _send({"status": "done_load", "ref_name": data.get("ref_name", "?")})
        except Exception:
            err = traceback.format_exc()
            _log(err, "error")
            _send({"status": "error", "message": err})

    elif action == "generate":
        text         = cmd.get("text", "")
        refs         = cmd.get("refs", [])
        profile_path = cmd.get("profile_path")
        out_path     = cmd.get("out", "output.wav")
        exaggeration = float(cmd.get("exaggeration", 0.5))
        cfg_weight   = float(cmd.get("cfg_weight",   0.5))
        temperature  = float(cmd.get("temperature",  0.8))
        max_steps    = int(cmd.get("max_steps",      _DEFAULT_STEPS))
        post_process = cmd.get("post_process", False)

        try:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

            # Apply requested step count
            _patch_steps(max_steps)

            gen_kw = dict(exaggeration=exaggeration, cfg_weight=cfg_weight, temperature=temperature)

            # Resolve source
            use_cond = None
            use_path = None

            if profile_path and os.path.isfile(profile_path):
                try:
                    data = torch.load(profile_path, map_location=DEVICE, weights_only=False)
                    if data.get("version", 0) >= 1:
                        use_cond = data["cond"]
                        _log(f"Profile: {data.get('ref_name','?')}", "info")
                    else:
                        use_path = data.get("ref_path", "")
                        if not os.path.isfile(use_path): use_path = None
                except Exception as e:
                    _log(f"Profile load failed, using refs: {e}", "warn")

            if use_cond is None and use_path is None:
                ref = next((r for r in refs if os.path.isfile(r)), None)
                if ref:
                    use_cond, use_path = _resolve_cond(ref)

            # Generate
            if use_cond is not None:
                try:
                    wav = MODEL.generate(text, conditioning=use_cond, **gen_kw)
                except TypeError:
                    wav = MODEL.generate(text, **gen_kw)
            elif use_path:
                wav = MODEL.generate(text, audio_prompt_path=use_path, **gen_kw)
            else:
                _log("No reference — using default voice.", "warn")
                wav = MODEL.generate(text, **gen_kw)

            # Post-process
            if post_process:
                try:
                    import numpy as np
                    from scipy.signal import butter, lfilter
                    samples = wav.squeeze().cpu().numpy().astype(np.float32)
                    b, a = butter(2, 80 / (MODEL.sr / 2), btype="high")
                    samples = lfilter(b, a, samples)
                    peak = np.abs(samples).max() + 1e-9
                    samples = samples * (10 ** (-1.0 / 20.0) / peak)
                    wav = torch.from_numpy(samples).unsqueeze(0)
                except Exception as e:
                    _log(f"Post-process skipped: {e}", "warn")

            ta.save(out_path, wav, MODEL.sr)
            duration = wav.shape[-1] / MODEL.sr
            _send({"status": "done", "file": out_path, "duration": duration})
            _log(f"Saved → {os.path.basename(out_path)} ({duration:.1f}s)", "ok")

        except Exception:
            err = traceback.format_exc()
            _log(err, "error")
            _send({"status": "error", "message": err})

    else:
        _log(f"Unknown action: {action}", "warn")
        _send({"status": "error", "message": f"Unknown action: {action}"})
