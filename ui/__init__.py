"""
VoiceTTSr UI Package
Modular UI components, themes, presets, and high-DPI scaling helpers.
"""

from ui.theme import (
    DARK_BG,
    PANEL_BG,
    CARD_BG,
    BORDER,
    ACCENT,
    ACCENT2,
    ACCENT3,
    TEXT_PRI,
    TEXT_SEC,
    TEXT_MUT,
    DANGER,
    WARNING,
    XTTS_PRESETS,
    QWEN_PRESETS,
    CHATTERBOX_PRESETS,
    RVC_PRESETS,
    QWEN_EMOTION_TAGS,
    PRESETS,
    enable_high_dpi,
)

from ui.components.ethics_dialog import (
    show_first_run_agreement,
    show_policy_viewer,
    PRIVACY_POLICY_TEXT,
)

__all__ = [
    "DARK_BG",
    "PANEL_BG",
    "CARD_BG",
    "BORDER",
    "ACCENT",
    "ACCENT2",
    "ACCENT3",
    "TEXT_PRI",
    "TEXT_SEC",
    "TEXT_MUT",
    "DANGER",
    "WARNING",
    "XTTS_PRESETS",
    "QWEN_PRESETS",
    "CHATTERBOX_PRESETS",
    "RVC_PRESETS",
    "QWEN_EMOTION_TAGS",
    "PRESETS",
    "enable_high_dpi",
    "show_first_run_agreement",
    "show_policy_viewer",
    "PRIVACY_POLICY_TEXT",
]
