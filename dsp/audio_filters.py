"""
Audio Filtering & Equalization DSP
Provides high-pass filtering, presence EQ boost, and noise removal functions.
"""

import os
from pydub import AudioSegment, effects


def apply_highpass(input_path: str, output_path: str = None, cutoff_hz: int = 80) -> str:
    """Apply a high-pass filter to remove low-frequency rumble / DC hum."""
    if output_path is None:
        output_path = input_path

    audio = AudioSegment.from_file(input_path)
    filtered = audio.high_pass_filter(cutoff_hz)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    filtered.export(output_path, format="wav")
    return output_path


def apply_presence_eq(input_path: str, output_path: str = None, gain_db: float = 2.5) -> str:
    """
    Apply presence boost to enhance vocal clarity in dialogue.
    Blends a subtle high-shelf boost with original audio.
    """
    if output_path is None:
        output_path = input_path

    audio = AudioSegment.from_file(input_path)
    # Extract highs and overlay with gain
    highs = audio.high_pass_filter(3000)
    highs = highs + gain_db
    enhanced = audio.overlay(highs)
    # Normalize volume
    enhanced = effects.normalize(enhanced)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    enhanced.export(output_path, format="wav")
    return output_path
