"""
Audio Health & Spectral Quality Analyzer
Calculates acoustic metrics: RMS energy, peak clipping, DC offset, SNR, spectral distribution, and muffled voice flags.
"""

import os
import wave
from dataclasses import dataclass, field
from typing import List
import numpy as np
from numpy.fft import rfft, rfftfreq


@dataclass
class AudioHealthReport:
    """Comprehensive acoustic quality analysis report for a single audio file."""
    filepath: str
    filename: str
    status: str              # 'OK', 'WARN', 'ERR'
    tag: str                 # 'ok', 'warn', 'err'
    score: int               # 0 - 100
    duration_sec: float
    rms_pct: float
    peak_dbfs: float
    clipping_pct: float
    snr_db: float
    is_muffled: bool
    dc_offset: float
    issues: List[str] = field(default_factory=list)


def analyze_audio_file(
    wav_path: str,
    min_dur: float = 1.0,
    max_dur: float = 30.0,
    noise_thr: float = 0.03,
    clip_thr: float = 0.98
) -> AudioHealthReport:
    """
    Perform deep acoustic health and quality analysis on a WAV audio file.
    """
    fname = os.path.basename(wav_path)
    score = 100
    issues: List[str] = []

    try:
        with wave.open(wav_path, "rb") as wf:
            sr = wf.getframerate()
            nch = wf.getnchannels()
            sw = wf.getsampwidth()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)

        dtype = np.int16 if sw == 2 else np.int32
        samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        if nch > 1:
            samples = samples.reshape(-1, nch).mean(axis=1)
        scale = float(2 ** (sw * 8 - 1))
        samples /= scale
        dur = len(samples) / sr
        peak = float(np.abs(samples).max())
        rms = float(np.sqrt(np.mean(samples ** 2)))
        peak_dbfs = 20 * np.log10(peak + 1e-9)
        clip_pct = float(np.mean(np.abs(samples) >= clip_thr) * 100)

        # 1. Duration
        if dur < min_dur:
            issues.append(f"too short ({dur:.2f}s)")
            score -= 40
        elif dur > max_dur:
            issues.append(f"too long ({dur:.0f}s)")
            score -= 15

        # 2. RMS / Silence
        if rms < 0.01:
            issues.append(f"near-silent (RMS={rms:.4f})")
            score -= 50
        elif rms < 0.02:
            issues.append("very low level")
            score -= 20

        # 3. Peak Clipping
        if clip_pct > 1.0:
            issues.append(f"heavy clipping ({clip_pct:.1f}%)")
            score -= 30
        elif clip_pct > 0.1:
            issues.append(f"some clipping ({clip_pct:.2f}%)")
            score -= 10

        # 4. Flat-top Clipping
        max_val = np.abs(samples).max()
        flat_runs = int(np.sum(np.abs(samples) >= max_val * 0.999))
        if flat_runs > 10 and clip_pct < 0.1:
            issues.append(f"flat-top clips ({flat_runs} smp)")
            score -= 10

        # 5. DC Offset
        dc = float(np.mean(samples))
        if abs(dc) > 0.05:
            issues.append(f"DC offset ({dc:+.3f})")
            score -= 15
        elif abs(dc) > 0.02:
            issues.append(f"mild DC ({dc:+.3f})")
            score -= 5

        # 6. SNR (Signal-to-Noise Ratio)
        noise_len = min(int(sr * 0.15), len(samples) // 6, 4096)
        snr = 0.0
        if noise_len > 64:
            noise_rms = float(np.sqrt(np.mean(samples[:noise_len] ** 2))) + 1e-9
            snr = 20 * np.log10(rms / noise_rms)
            if snr < 10:
                issues.append(f"very noisy (SNR {snr:.0f}dB)")
                score -= 25
            elif snr < 20:
                issues.append(f"noisy (SNR {snr:.0f}dB)")
                score -= 10

        # 7. Muffling — Spectral energy above 2 kHz
        chunk = min(8192, len(samples))
        F = np.abs(rfft(samples[:chunk] * np.hanning(chunk)))
        freqs = rfftfreq(chunk, 1 / sr)
        hi = float(np.sum(F[freqs >= 2000] ** 2))
        tot_spec = float(np.sum(F ** 2)) + 1e-9
        hi_ratio = hi / tot_spec
        muffled_flag = hi_ratio < 0.08
        if hi_ratio < 0.04:
            issues.append(f"severely muffled ({hi_ratio*100:.0f}%>2kHz)")
            score -= 30
        elif hi_ratio < 0.08:
            issues.append(f"muffled ({hi_ratio*100:.0f}%>2kHz)")
            score -= 15

        # 8. Background Noise (Tail RMS)
        tail_len = min(int(sr * 0.3), len(samples) // 4)
        tail_rms = float(np.sqrt(np.mean(samples[-tail_len:] ** 2)))
        if tail_rms >= noise_thr:
            issues.append(f"noisy tail (RMS={tail_rms:.3f})")
            score -= 10

        score = max(0, score)
        if score >= 75:
            status = "OK"
            tag = "ok"
        elif score >= 45:
            status = "WARN"
            tag = "warn"
        else:
            status = "ERR"
            tag = "err"

        return AudioHealthReport(
            filepath=wav_path,
            filename=fname,
            status=status,
            tag=tag,
            score=score,
            duration_sec=round(dur, 2),
            rms_pct=round(rms * 100, 2),
            peak_dbfs=round(peak_dbfs, 1),
            clipping_pct=round(clip_pct, 2),
            snr_db=round(snr, 1),
            is_muffled=muffled_flag,
            dc_offset=round(dc, 4),
            issues=issues
        )

    except Exception as e:
        return AudioHealthReport(
            filepath=wav_path,
            filename=fname,
            status="ERR",
            tag="err",
            score=0,
            duration_sec=0.0,
            rms_pct=0.0,
            peak_dbfs=-99.0,
            clipping_pct=0.0,
            snr_db=0.0,
            is_muffled=False,
            dc_offset=0.0,
            issues=[f"Read error: {e}"]
        )
