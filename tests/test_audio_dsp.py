import os
import wave
import tempfile
import numpy as np
import pytest


def create_test_wav(path, duration_sec=1.0, sample_rate=24000, freq=440.0, dc_offset=0.0, volume=0.5):
    """Helper to generate a clean synthetic WAV file for DSP testing."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    samples = np.sin(2 * np.pi * freq * t) * volume + dc_offset
    # Clip to -1.0 to 1.0
    samples = np.clip(samples, -1.0, 1.0)
    int_samples = (samples * 32767).astype(np.int16)

    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())


class TestAudioDspMetrics:
    def test_clean_audio_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "clean_440hz.wav")
            create_test_wav(wav_path, duration_sec=1.5, volume=0.5)

            with wave.open(wav_path, 'rb') as wf:
                sr = wf.getframerate()
                nframes = wf.getnframes()
                raw = wf.readframes(nframes)

            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(samples ** 2)))
            dc = float(np.mean(samples))
            peak = float(np.abs(samples).max())

            # Sine wave RMS for amplitude 0.5 is 0.5 / sqrt(2) ≈ 0.3535
            assert 0.34 < rms < 0.36
            assert abs(dc) < 0.001
            assert 0.49 < peak <= 0.51

    def test_dc_offset_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "dc_offset.wav")
            create_test_wav(wav_path, duration_sec=1.0, dc_offset=0.15, volume=0.3)

            with wave.open(wav_path, 'rb') as wf:
                raw = wf.readframes(wf.getnframes())

            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            dc = float(np.mean(samples))
            assert abs(dc - 0.15) < 0.01

    def test_clipping_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "clipped.wav")
            # Over-drive sine wave so it gets clipped at boundaries
            create_test_wav(wav_path, duration_sec=1.0, volume=2.0)

            with wave.open(wav_path, 'rb') as wf:
                raw = wf.readframes(wf.getnframes())

            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            clip_pct = float(np.mean(np.abs(samples) >= 0.99) * 100)
            assert clip_pct > 10.0
