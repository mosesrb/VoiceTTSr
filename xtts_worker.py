"""
xtts_worker.py — XTTS v2 subprocess worker for VoiceTTSr
Run inside xtts-env-py310 (transformers==4.36.2, TTS==0.22.0)

Protocol (JSON lines over stdin/stdout):
  stdin  ← {"action":"generate", "text":..., "refs":[...], "lang":"en", "out":"...", "speed":1.0}
  stdout → {"status":"log",  "text":"...", "level":"info|warn|error"}
  stdout → {"status":"ready"}
  stdout → {"status":"done", "file":"...", "duration":4.5}
  stdout → {"status":"error","message":"..."}
"""
import sys, json, os, wave

# ── send helpers ──────────────────────────────────────────────────────────────
def send(obj):
    print(json.dumps(obj), flush=True)

def log(text, level="info"):
    send({"status": "log", "text": text, "level": level})


def normalize_wav(path):
    """Ensure WAV is Mono 24kHz; fix in-place if needed. Returns True if ok."""
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(path)
        if audio.frame_rate != 24000 or audio.channels != 1 or audio.sample_width != 2:
            log(f"Normalizing → Mono 24kHz: {os.path.basename(path)}", "warn")
            audio = (audio
                     .set_channels(1)
                     .set_frame_rate(24000)
                     .set_sample_width(2))
            audio.export(path, format="wav")
        return True
    except Exception as e:
        log(f"Could not normalize {path}: {e}", "error")
        return False


def post_process_audio(path):
    """Option A: Pro-audio enhancer for output."""
    try:
        from pydub import AudioSegment, effects
        log(f"Applying pro-audio post-processing: {os.path.basename(path)}")
        audio = AudioSegment.from_file(path)
        
        # 1. Subtle High-pass filter (75Hz instead of 100Hz)
        # 75Hz removes low-end "mud" without making the voice sound like an old radio.
        audio = audio.high_pass_filter(75)
        
        # 2. Final Normalization to -1.5dB
        # This ensures it's loud and clear without EVER clipping (the VLC distortion issue).
        audio = effects.normalize(audio, headroom=1.5)

        audio.export(path, format="wav")
        return True
    except Exception as e:
        log(f"Post-processing failed: {e}", "error")
        return False


def clean_references(paths):
    """Option B: Clean up reference clips."""
    count = 0
    try:
        from pydub import AudioSegment, effects
        from pydub.silence import split_on_silence
        
        for p in paths:
            if not os.path.isfile(p): continue
            audio = AudioSegment.from_file(p)
            
            # Trim silence from start/end
            chunks = split_on_silence(audio, min_silence_len=400, silence_thresh=-45, keep_silence=100)
            if chunks:
                audio = chunks[0]
                for chunk in chunks[1:]:
                    audio += chunk
            
            # Normalize to -1dB
            audio = effects.normalize(audio, headroom=1.0)
            
            # Fix format
            audio = (audio.set_channels(1)
                          .set_frame_rate(24000)
                          .set_sample_width(2))
            
            audio.export(p, format="wav")
            count += 1
        return count
    except Exception as e:
        log(f"Clean refs failed: {e}", "error")
        return count


def get_duration(path):
    try:
        with wave.open(path) as wf:
            return round(wf.getnframes() / wf.getframerate(), 2)
    except Exception:
        return 0.0


def main():
    log("XTTS v2 worker starting…")
    try:
        import torch
        from TTS.api import TTS
    except ImportError as e:
        send({"status": "error", "message": f"Import failed: {e}"})
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")

    try:
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        log("XTTS v2 model loaded successfully.", "ok")
        send({"status": "ready"})
    except Exception as e:
        send({"status": "error", "message": f"Model load failed: {e}"})
        sys.exit(1)

    # ── main command loop ─────────────────────────────────────────────────────
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
            log("Worker shutting down.")
            break

        elif action == "save_profile":
            try:
                refs = cmd["refs"]
                out_path = cmd["out_path"]
                valid = [r for r in refs if normalize_wav(r)]
                if not valid:
                    send({"status": "error", "message": "No valid reference WAVs."})
                    continue
                import torch
                gpt_cond, spk_embed = tts.synthesizer.tts_model.get_conditioning_latents(audio_path=valid)
                torch.save({"gpt_cond_latent": gpt_cond, "speaker_embedding": spk_embed}, out_path)
                send({"status": "done_save", "file": out_path})
            except Exception as e:
                import traceback
                send({"status": "error", "message": str(e)})
                log(traceback.format_exc(), "error")

        elif action == "generate":
            try:
                text   = cmd["text"]
                refs   = cmd["refs"]
                lang   = cmd.get("lang", "en")
                out    = cmd["out"]
                speed  = float(cmd.get("speed", 1.0))
                profile_path = cmd.get("profile_path")
                post_process = cmd.get("post_process", False)

                if profile_path and os.path.isfile(profile_path):
                    import torch
                    log(f"Generating from profile {os.path.basename(profile_path)} (speed={speed:.2f}): {text[:60]}…")
                    data = torch.load(profile_path, map_location=device)
                    gpt_cond = data["gpt_cond_latent"]
                    spk_embed = data["speaker_embedding"]
                    
                    wav = tts.synthesizer.tts_model.inference(
                        text=text,
                        language=lang,
                        gpt_cond_latent=gpt_cond,
                        speaker_embedding=spk_embed,
                        temperature=float(cmd.get("temperature", 0.55)),
                        length_penalty=float(cmd.get("rep_pen", 5.0)),
                        repetition_penalty=float(cmd.get("rep_pen", 5.0)),
                        top_k=int(cmd.get("top_k", 50)),
                        top_p=float(cmd.get("top_p", 0.85)),
                        speed=speed,
                        enable_text_splitting=False
                    )["wav"]
                    import soundfile as sf
                    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
                    sf.write(out, wav, 24000)
                else:
                    # Normalize all refs (only if no profile provided)
                    valid = [r for r in refs if normalize_wav(r)]
                    if not valid:
                        send({"status": "error", "message": "No valid reference WAVs after normalization."})
                        continue

                    log(f"Generating ({len(valid)} ref(s), speed={speed:.2f}): {text[:60]}…")

                    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

                    tts.tts_to_file(
                        text=text,
                        speaker_wav=valid,
                        language=lang,
                        file_path=out,
                        speed=speed,
                        split_sentences=False,
                    )

                if post_process:
                    post_process_audio(out)

                dur = get_duration(out)
                send({"status": "done", "file": out, "duration": dur})

            except Exception as e:
                import traceback
                send({"status": "error", "message": str(e)})
                log(traceback.format_exc(), "error")

        elif action == "enhance_refs":
            try:
                refs = cmd["refs"]
                count = clean_references(refs)
                send({"status": "done_enhance", "count": count})
            except Exception as e:
                send({"status": "error", "message": str(e)})

        elif action == "post_process":
            try:
                path = cmd["file"]
                post_process_audio(path)
                send({"status": "done_post"})
            except Exception as e:
                send({"status": "error", "message": str(e)})

        else:
            log(f"Unknown action: {action!r}", "error")


if __name__ == "__main__":
    main()
