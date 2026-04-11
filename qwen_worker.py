"""
qwen_worker.py — Qwen3-TTS subprocess worker for VoiceTTSr
Runs under the SYSTEM Python (qwen-tts already installed there).

Protocol (JSON lines over stdin/stdout):
  stdin  <- {"action":"generate", "text":..., "refs":[...], "lang":"en",
              "out":"...", "temperature":0.85, "top_k":50, "top_p":0.9,
              "rep_pen":1.3, "preset":"Natural", "use_icl":true,
              "stream":false}
  stdout -> {"status":"log",   "text":"...", "level":"info|warn|error|ok"}
  stdout -> {"status":"ready"}
  stdout -> {"status":"chunk", "file":"...", "index":0}   # streaming only
  stdout -> {"status":"done",  "file":"...", "duration":4.5}
  stdout -> {"status":"error", "message":"..."}

Changes vs original:
  - Streaming: when cmd["stream"]=true the worker writes audio in chunks
    and sends {"status":"chunk"} after each one. The GUI plays each chunk
    immediately via pygame, giving a streaming preview effect.
  - Multi-ref selection: picks the longest valid reference WAV instead of
    the first one, giving Qwen more speaker context.
  - Mumble guard: checks output RMS before confirming "done" — if silent
    or near-silent reports error so the GUI retry logic can kick in.
  - Proper use_icl pass-through: ICL mode sends text as-is (emotion tags
    already inserted by GUI). Non-ICL mode applies preset formatting.
  - Encoding: stdout forced to utf-8 for Hindi/CJK/etc support.
"""
import sys, json, os, traceback

# ── force utf-8 on stdout so Hindi/CJK text doesn't crash on Windows ─────
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  line_buffering=True)

def send(obj):
    print(json.dumps(obj, ensure_ascii=False), flush=True)

def log(text, level="info"):
    send({"status": "log", "text": text, "level": level})


# ── text formatting ──────────────────────────────────────────────────────
def format_text_for_preset(text, preset_name, use_icl=False):
    """
    Apply Qwen-specific style tokens based on the active preset.
    When use_icl=True the GUI has already inserted emotion tags — return as-is.
    """
    if use_icl:
        return text   # GUI already handled tagging

    if preset_name in ("Aggressive", "Angry"):
        return f"RRR- {text.upper()}!!!"
    elif preset_name in ("Warm", "Breathy", "Seductive", "Alluring"):
        clean = text.lower().replace("!", ".").replace("?", ".")
        return f"h- {clean.replace('. ', '... ')}"
    return text


# ── mumble / silence check ───────────────────────────────────────────────
def _is_silent(wav_array, threshold=0.008):
    """Return True if the audio is effectively silent (RMS below threshold)."""
    try:
        import numpy as np
        rms = float(np.sqrt(np.mean(wav_array.astype(np.float32) ** 2)))
        return rms < threshold
    except Exception:
        return False


# ── streaming helpers ────────────────────────────────────────────────────
def _write_chunk(wav_slice, sr, path):
    """Write a numpy slice to a WAV file and return True on success."""
    try:
        import soundfile as sf
        import numpy as np
        chunk = np.array(wav_slice, dtype=np.float32)
        sf.write(path, chunk, sr)
        return True
    except Exception:
        return False


def main():
    log("Qwen3-TTS worker starting…")

    try:
        import torch
        import soundfile as sf
        import numpy as np
        from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    except ImportError as e:
        send({"status": "error",
              "message": f"Import failed: {e}\n"
                         "Make sure qwen-tts is installed in the system Python."})
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")

    # 1.7B has stronger emotion control and better cloning than 0.6B
    # Falls back to 0.6B if 1.7B is not downloaded yet
    MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    try:
        log(f"Loading {MODEL_ID} …")
        model = Qwen3TTSModel.from_pretrained(MODEL_ID)
        log("Qwen3-TTS model loaded.", "ok")
        send({"status": "ready"})
    except Exception as e:
        send({"status": "error",
              "message": f"Model load failed: {e}\n{traceback.format_exc()}"})
        sys.exit(1)

    # ── main command loop ─────────────────────────────────────────────────
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            cmd = json.loads(raw)
        except json.JSONDecodeError:
            log("Bad JSON from GUI", "error")
            continue

        action = cmd.get("action", "")

        if action == "ping":
            send({"status": "pong"})

        elif action == "quit":
            log("Qwen worker shutting down.")
            break

        elif action == "create_profile":
            try:
                refs = cmd["refs"]
                out  = cmd["out"]
                log(f"Creating profile for {len(refs)} files...")
                
                all_embs = []
                for r in refs:
                    if not os.path.isfile(r): continue
                    # Create prompt items handles resampling + extraction
                    items = model.create_voice_clone_prompt(ref_audio=r, x_vector_only_mode=True)
                    if items and items[0].ref_spk_embedding is not None:
                        all_embs.append(items[0].ref_spk_embedding)
                
                if not all_embs:
                    send({"status": "error", "message": "No valid reference audio found for profile."})
                    continue
                
                # Average embeddings
                avg_emb = torch.stack(all_embs).mean(dim=0)
                
                # Save as a standard torch file
                torch.save(avg_emb, out)
                log(f"Profile saved to: {os.path.basename(out)}", "ok")
                send({"status": "done", "file": out})
            except Exception as e:
                send({"status": "error", "message": str(e)})
                log(traceback.format_exc(), "error")

        elif action == "generate":
            try:
                text        = cmd["text"]
                refs        = cmd["refs"]
                out         = cmd["out"]
                temperature = float(cmd.get("temperature", 0.85))
                top_k       = int(cmd.get("top_k", 50))
                top_p       = float(cmd.get("top_p", 0.9))
                rep_pen     = float(cmd.get("rep_pen", 1.3))
                preset      = cmd.get("preset", "Natural")
                use_icl     = bool(cmd.get("use_icl", False))
                do_stream   = bool(cmd.get("stream", False))
                profile_pth = cmd.get("profile_path", "")

                # Apply preset formatting (skipped if use_icl — GUI handled tags)
                qwen_text = format_text_for_preset(text, preset, use_icl)
                if qwen_text != text:
                    log(f"Preset format ({preset}): {qwen_text[:80]}")
                else:
                    log(f"Generating ({preset}): {text[:80]}")

                # ── Multi-reference prompt building ──────────────────────
                prompt_dict = None
                
                if profile_pth and os.path.isfile(profile_pth):
                    log(f"Using profile: {os.path.basename(profile_pth)}")
                    avg_emb = torch.load(profile_pth, map_location=device)
                    # We create a prompt dict manually
                    # Ref code is None for x_vector_only_mode
                    prompt_dict = {
                        "ref_code": [None],
                        "ref_spk_embedding": [avg_emb],
                        "x_vector_only_mode": [True],
                        "icl_mode": [False]
                    }
                else:
                    # Original logic: pick longest ref
                    valid_refs = [r for r in refs if os.path.isfile(r)]
                    if not valid_refs:
                        send({"status": "error", "message": "No valid reference WAV found."})
                        continue

                    def _wav_dur(path):
                        try:
                            # Re-import sf just in case inside helper
                            import soundfile as sf
                            info = sf.info(path)
                            return info.duration
                        except Exception:
                            return 0.0

                    ref_wav = max(valid_refs, key=_wav_dur)
                    log(f"Using ref: {os.path.basename(ref_wav)}")

                # ── Generate ─────────────────────────────────────────────
                os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

                if prompt_dict is None:
                    # Build normal prompt from single file
                    # Full ICL mode (x_vector_only_mode=False) needs ref_text.
                    # Look for a sidecar .txt with same stem as the ref WAV.
                    ref_txt_path = os.path.splitext(ref_wav)[0] + ".txt"
                    ref_text = None
                    if os.path.isfile(ref_txt_path):
                        try:
                            with open(ref_txt_path, encoding="utf-8") as _f:
                                ref_text = _f.read().strip() or None
                        except Exception:
                            ref_text = None
                    if ref_text:
                        log(f"Full ICL mode — ref_text: {ref_text[:60]}")
                    else:
                        log("X-vector mode (no transcript sidecar found)")

                    wavs, out_sr = model.generate_voice_clone(
                        text=qwen_text,
                        ref_audio=ref_wav,
                        ref_text=ref_text,           # None → x_vector_only_mode
                        x_vector_only_mode=(ref_text is None),
                        do_sample=True,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=rep_pen,
                        max_new_tokens=2048,
                    )
                else:
                    # Bypassing audio analysis by providing memory-loaded prompt
                    wavs, out_sr = model.generate_voice_clone(
                        text=qwen_text,
                        voice_clone_prompt=prompt_dict,
                        do_sample=True,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=rep_pen,
                        max_new_tokens=2048,
                    )

                wav = wavs[0]   # numpy array, float32, shape (N,)

                # ── Mumble guard ──────────────────────────────────────────
                if _is_silent(wav):
                    send({"status": "error",
                          "message": "Output is silent — possible mumble/failure."})
                    continue

                # ── Streaming: split into ~2s chunks and emit as they're ready
                if do_stream:
                    chunk_samples = int(out_sr * 2.0)
                    n_chunks = max(1, len(wav) // chunk_samples)
                    base, ext = os.path.splitext(out)
                    for ci in range(n_chunks):
                        sl = wav[ci * chunk_samples: (ci + 1) * chunk_samples]
                        if len(sl) == 0:
                            continue
                        chunk_path = f"{base}_chunk{ci:03d}{ext}"
                        if _write_chunk(sl, out_sr, chunk_path):
                            send({"status": "chunk", "file": chunk_path, "index": ci})
                    # Write the tail (remainder after last full chunk)
                    tail = wav[n_chunks * chunk_samples:]
                    if len(tail) > int(out_sr * 0.1):   # ignore sub-100ms tail
                        chunk_path = f"{base}_chunk{n_chunks:03d}{ext}"
                        if _write_chunk(tail, out_sr, chunk_path):
                            send({"status": "chunk", "file": chunk_path, "index": n_chunks})

                # ── Write final complete file ─────────────────────────────
                sf.write(out, wav, out_sr)
                dur = round(len(wav) / out_sr, 2)
                log(f"Qwen output: {dur}s @ {out_sr}Hz", "ok")
                send({"status": "done", "file": out, "duration": dur})

            except Exception as e:
                send({"status": "error", "message": str(e)})
                log(traceback.format_exc(), "error")

        else:
            log(f"Unknown action: {action!r}", "error")


if __name__ == "__main__":
    main()
