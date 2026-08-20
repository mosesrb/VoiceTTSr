"""
VoiceTTSr UI Theme & Styling System
Defines color palettes, preset dictionaries, and High-DPI display initialization.
"""

import sys

def enable_high_dpi() -> None:
    """Enable High-DPI awareness on Windows to prevent blurry text on high-res displays."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                import ctypes
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


# ── Color Palette ────────────────────────────────────────────────────────────
DARK_BG  = "#0f0f13"
PANEL_BG = "#16161e"
CARD_BG  = "#1e1e2a"
BORDER   = "#2a2a3a"
ACCENT   = "#7c6af7"
ACCENT2  = "#5dcaa5"
ACCENT3  = "#f7a26a"
TEXT_PRI = "#e8e6f0"
TEXT_SEC = "#8885a0"
TEXT_MUT = "#55536a"
DANGER   = "#e24b4a"
WARNING  = "#ef9f27"


# ── XTTS v2 Presets (tuned for stable faithful cloning) ──────────────────────
XTTS_PRESETS = {
    "Natural":      (0.55, 1.00, 5.0, 50, 0.85, "Balanced everyday voice"),
    "Warm":         (0.45, 0.92, 6.0, 40, 0.80, "Soft, intimate — podcasts/narration"),
    "Crisp":        (0.35, 1.05, 7.0, 30, 0.75, "Clear & precise — announcements"),
    "Expressive":   (0.75, 0.95, 4.0, 70, 0.92, "Emotional range — storytelling"),
    "Fast Draft":   (0.50, 1.40, 5.0, 50, 0.85, "Quick generation, slightly rougher"),
    "Slow & Clear": (0.40, 0.75, 6.5, 35, 0.78, "Accessibility / tutorials"),
    "Deep":         (0.42, 0.88, 6.0, 38, 0.78, "Rich, authoritative tone"),
    "Cinematic":    (0.45, 0.88, 8.5, 40, 0.80, "Steady, weighty & dramatic — trailers"),
}

# ── Qwen3-TTS Presets (tuned for expressiveness & emotion acting) ─────────────
QWEN_PRESETS = {
    "Natural":    (0.55, 1.00, 5.0, 50,  0.85, "Clean, faithful clone — no formatting"),
    "Warm":       (0.45, 0.92, 6.0, 40,  0.80, "Soft intimate — h- format auto-applied"),
    "Breathy":    (0.68, 0.90, 4.5, 60,  0.90, "Soft, airy — ASMR / relaxation"),
    "Seductive":  (0.78, 0.85, 5.5, 65,  0.95, "Slow, breathy, intimate — teasing tone"),
    "Alluring":   (0.82, 0.90, 5.0, 75,  0.92, "Intimate & highly expressive"),
    "Deep":       (0.42, 0.88, 6.0, 38,  0.78, "Rich, authoritative tone"),
    "Expressive": (0.75, 0.95, 4.0, 70,  0.92, "Emotional range — storytelling"),
    "Hyper-Real": (0.92, 1.05, 4.2, 80,  0.98, "Natural imperfections, vocal fry"),
    "Aggressive": (0.95, 1.05, 3.5, 85,  0.95, "Loud, harsh — RRR- format auto-applied"),
}

# ── Chatterbox Presets (exaggeration, cfg_weight, temperature, description) ────
CHATTERBOX_PRESETS = {
    "Natural":      (0.50, 0.50, 0.80, "Balanced everyday voice"),
    "Warm":         (0.40, 0.40, 0.70, "Soft, intimate — podcasts/narration"),
    "Expressive":   (0.80, 0.50, 0.85, "Emotional range — storytelling"),
    "Dramatic":     (1.20, 0.50, 0.90, "High emotion — cinematic/trailers"),
    "Calm":         (0.35, 0.40, 0.60, "Slow, measured — tutorials/docs"),
    "Energetic":    (0.90, 0.60, 0.90, "Upbeat, punchy — ads/promos"),
    "Whisper":      (0.30, 0.30, 0.65, "Soft, subdued — ASMR-adjacent"),
    "Max Drama":    (1.50, 0.50, 0.95, "Maximum expressiveness — experimental"),
}

# ── RVC v2 Presets ────────────────────────────────────────────────────────────
RVC_PRESETS = {
    "Natural":       (0,   0.40, "rmvpe",   "Clean reskin, faithful to source"),
    "Character+":    (0,   0.75, "rmvpe",   "Strong index — heavy voice texture"),
    "Subtle":        (0,   0.20, "rmvpe",   "Light touch — barely-there reskin"),
    "Pitch Down":    (-4,  0.40, "rmvpe",   "4 semitones lower — deeper voice"),
    "Pitch Up":      (4,   0.40, "rmvpe",   "4 semitones higher — lighter voice"),
    "Feminine":      (6,   0.50, "rmvpe",   "Shift up for female-sounding output"),
    "Masculine":     (-6,  0.50, "rmvpe",   "Shift down for male-sounding output"),
    "Harvest":       (0,   0.40, "harvest", "Harvest F0 — smoother on noisy audio"),
    "PM Fast":       (0,   0.30, "pm",      "Fastest F0 — lower quality, quick preview"),
}

# ── Qwen3-TTS Emotion → Text Prefix/Suffix Map ──────────────────────────────
QWEN_EMOTION_TAGS = {
    "joy":       ("",                          " :)",   "light upbeat suffix"),
    "love":      ("h- ",                       "",      "soft intimate prefix"),
    "sadness":   ("h- ",                       "...",   "soft + trailing ellipsis"),
    "fear":      ("h- ",                       "",      "soft hushed prefix"),
    "anger":     ("",                          "!!!",   "strong exclamation"),
    "disgust":   ("",                          ".",     "flat period — deadpan"),
    "surprise":  ("",                          "!",     "exclamation"),
    "neutral":   ("",                          "",      "no modification"),
}

# Legacy merged dictionary
PRESETS = {**XTTS_PRESETS, **QWEN_PRESETS, **CHATTERBOX_PRESETS}
