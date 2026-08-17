"""
Audio Normalization Utilities
Handles conversions to standardized 16-bit PCM Mono WAV files at specified sample rates.
"""

import os
import wave
from pydub import AudioSegment


def normalize_to_wav(input_path: str, output_path: str = None, target_sr: int = 24000, channels: int = 1, sample_width: int = 2) -> str:
    """
    Standardize an audio file to uncompressed 16-bit PCM WAV.
    
    :param input_path: Source audio file (.wav, .mp3, .ogg, .flac, etc.)
    :param output_path: Destination path. If None, overwrites or creates adjacent .wav.
    :param target_sr: Target sample rate (e.g., 24000 for XTTS, 44100 for Skyrim, 16000 for ASR).
    :param channels: Target channel count (1 for Mono, 2 for Stereo).
    :param sample_width: Byte width per sample (2 = 16-bit PCM).
    :return: Output file path.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input audio file not found: {input_path}")

    if output_path is None:
        output_path = input_path

    # Check if already compliant
    if input_path.lower().endswith(".wav") and os.path.abspath(input_path) == os.path.abspath(output_path):
        try:
            with wave.open(input_path, "rb") as wf:
                if (wf.getnchannels() == channels and
                    wf.getsampwidth() == sample_width and
                    wf.getframerate() == target_sr and
                    wf.getcomptype() == "NONE"):
                    return input_path
        except Exception:
            pass

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(channels).set_frame_rate(target_sr).set_sample_width(sample_width)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    audio.export(output_path, format="wav")
    return output_path


def get_audio_duration(path: str) -> float:
    """Return the duration of a WAV file in seconds."""
    try:
        with wave.open(path, "rb") as wf:
            return round(wf.getnframes() / wf.getframerate(), 2)
    except Exception:
        try:
            audio = AudioSegment.from_file(path)
            return round(len(audio) / 1000.0, 2)
        except Exception:
            return 0.0
