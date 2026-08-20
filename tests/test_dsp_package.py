import os
import wave
import tempfile
import numpy as np
import pytest
from dsp import normalize_to_wav, get_audio_duration, analyze_audio_file, apply_presence_eq, apply_highpass


def create_synthetic_wav(path, duration=3.0, sr=24000, volume=0.3):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Speech-like harmonic blend: 150Hz pitch + vocal formants
    signal = (
        0.3 * np.sin(2 * np.pi * 150 * t) +
        0.2 * np.sin(2 * np.pi * 300 * t) +
        0.15 * np.sin(2 * np.pi * 1200 * t) +
        0.15 * np.sin(2 * np.pi * 2500 * t) +
        0.1 * np.sin(2 * np.pi * 3500 * t)
    ) * volume
    # Apply envelope with quiet start/end (0.2s head/tail)
    head_silence = int(sr * 0.2)
    tail_silence = int(sr * 0.3)
    signal[:head_silence] = 0.0
    signal[-tail_silence:] = 0.0

    samples = (np.clip(signal, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


class TestDspPackage:
    def test_audio_duration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "test_dur.wav")
            create_synthetic_wav(wav_path, duration=2.5, sr=24000)
            dur = get_audio_duration(wav_path)
            assert dur == 2.5

    def test_normalize_to_wav(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            in_wav = os.path.join(tmpdir, "in_48k.wav")
            out_wav = os.path.join(tmpdir, "out_24k.wav")
            create_synthetic_wav(in_wav, duration=1.0, sr=48000)

            result = normalize_to_wav(in_wav, out_wav, target_sr=24000)
            assert os.path.exists(result)

            with wave.open(result, "rb") as wf:
                assert wf.getframerate() == 24000
                assert wf.getnchannels() == 1
                assert wf.getsampwidth() == 2

    def test_analyze_audio_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "healthy.wav")
            create_synthetic_wav(wav_path, duration=3.0, volume=0.4)

            report = analyze_audio_file(wav_path)
            assert report.status in ["OK", "WARN"]
            assert report.score >= 50
            assert report.duration_sec == 3.0
            assert report.clipping_pct == 0.0

    def test_presence_eq_and_highpass(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            in_wav = os.path.join(tmpdir, "raw.wav")
            hp_wav = os.path.join(tmpdir, "hp.wav")
            eq_wav = os.path.join(tmpdir, "eq.wav")

            create_synthetic_wav(in_wav, duration=1.0, volume=0.5)

            apply_highpass(in_wav, hp_wav, cutoff_hz=80)
            assert os.path.exists(hp_wav)

            apply_presence_eq(in_wav, eq_wav, gain_db=2.0)
            assert os.path.exists(eq_wav)

    def test_xtts_post_process_audio(self):
        from xtts_worker import post_process_audio
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "synth.wav")
            create_synthetic_wav(wav_path, duration=1.0, volume=0.6)
            res = post_process_audio(wav_path)
            assert res is True
            assert os.path.exists(wav_path)

