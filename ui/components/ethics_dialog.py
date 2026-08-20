"""
VoiceTTSr Ethical Use, Terms & Privacy Agreement System
Provides the first-launch agreement modal and persistent in-app policy viewer.
"""

import os
import tkinter as tk
from tkinter import ttk, scrolledtext
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
)

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_VOICE_ETHICS_PATH = os.path.join(_BASE_DIR, "docs", "VOICE_ETHICS.md")
_PRIVACY_PATH = os.path.join(_BASE_DIR, "docs", "PRIVACY_POLICY.md")
_THIRD_PARTY_PATH = os.path.join(_BASE_DIR, "docs", "THIRD_PARTY_NOTICES.md")

PRIVACY_POLICY_TEXT = """# VoiceTTSr — Local Privacy Policy & Data Guarantee

> **Certified 100% Offline & Local Execution Architecture**  
> *Last Updated: August 2026*

---

## 1. 100% Offline & Local Execution

VoiceTTSr is designed from the ground up as an entirely local, self-contained desktop studio. All text-to-speech synthesis, neural voice cloning latents, flow-matching diffusion, and digital signal processing execute strictly on your personal computer hardware.

---

## 2. Zero Telemetry & Zero Cloud Transmission

* **No Telemetry**: VoiceTTSr collects zero analytics, zero telemetry, zero metrics, and zero crash dumps.
* **No Cloud Transmission**: Your voice reference audio, generated output files, prompt text, custom models, and speaker embeddings are **NEVER** uploaded or transmitted to any external server or third party.
* **Full Offline Operation**: Once initial technical models are downloaded, VoiceTTSr functions with all internet and network connections completely disabled.

---

## 3. Local File Safety & Storage

* **Local Storage**: Voice profiles (`.safetensors`) and generated audio files (`.wav`, `.fuz`) reside exclusively in your local application directories or your configured output folders.
* **Recycle Bin Protection**: File deletion actions in the studio route safely through your operating system's Recycle Bin / Trash via `send2trash` rather than permanently deleting data without recovery options.
* **Transparent Configuration**: Studio settings (`voicecloner_config.json`) are stored locally on your disk in human-readable JSON format.

---

## 4. User Responsibility & Biometric Voice Data

Because voice embeddings and cloned profiles represent biometric characteristics of individuals:

* You are solely responsible for securing your local files and reference recordings.
* You must ensure that any audio recorded or stored on your system complies with all applicable privacy, consent, and biometric data protection regulations in your jurisdiction.
"""


import re

def _load_doc_file(filepath: str, fallback_title: str) -> str:
    """Safely read a markdown documentation file or return fallback text."""
    if os.path.isfile(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"# {fallback_title}\n\nCould not load file: {e}"
    return f"# {fallback_title}\n\nFile not found at: {filepath}"


def _setup_text_tags(widget: tk.Text):
    """Configure modern typography and color tags for Tkinter Text widget."""
    widget.tag_configure("h1", font=("Segoe UI", 15, "bold"), foreground=ACCENT, spacing1=16, spacing3=8)
    widget.tag_configure("h2", font=("Segoe UI", 12, "bold"), foreground=ACCENT3, spacing1=14, spacing3=6)
    widget.tag_configure("h3", font=("Segoe UI", 10, "bold"), foreground=ACCENT2, spacing1=10, spacing3=4)
    widget.tag_configure("p", font=("Segoe UI", 10), foreground=TEXT_PRI, spacing1=2, spacing3=4, lmargin1=4, lmargin2=4)
    widget.tag_configure("quote", font=("Segoe UI", 9, "italic"), foreground=TEXT_SEC, lmargin1=16, lmargin2=16, spacing1=4, spacing3=4)
    widget.tag_configure("bullet", font=("Segoe UI", 10), foreground=TEXT_PRI, lmargin1=14, lmargin2=28, spacing1=2, spacing3=2)
    widget.tag_configure("bullet_dot", font=("Segoe UI", 10, "bold"), foreground=ACCENT)
    widget.tag_configure("num_dot", font=("Segoe UI", 10, "bold"), foreground=ACCENT3)
    widget.tag_configure("bold", font=("Segoe UI", 10, "bold"), foreground="#ffffff")
    widget.tag_configure("code", font=("Consolas", 9), foreground="#ff7b72", background="#21262d")
    widget.tag_configure("hr", font=("Segoe UI", 4), foreground=BORDER, spacing1=8, spacing3=8)


def _insert_formatted_line(widget: tk.Text, line_text: str, base_tag: str):
    """Insert a line of text while parsing inline **bold** and `code` tokens."""
    pattern = re.compile(r'(\*\*[^*]+\*\*|`[^`]+`)')
    tokens = pattern.split(line_text)
    for token in tokens:
        if not token:
            continue
        if token.startswith("**") and token.endswith("**") and len(token) >= 4:
            widget.insert("end", token[2:-2], (base_tag, "bold"))
        elif token.startswith("`") and token.endswith("`") and len(token) >= 2:
            widget.insert("end", f" {token[1:-1]} ", (base_tag, "code"))
        else:
            widget.insert("end", token, (base_tag,))
    widget.insert("end", "\n", (base_tag,))


def render_markdown_to_text(widget: tk.Text, markdown_text: str):
    """Parse and render markdown text with rich styling and typography into a Text widget."""
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    _setup_text_tags(widget)

    lines = markdown_text.splitlines()
    for line in lines:
        s = line.strip()
        if not s:
            widget.insert("end", "\n")
            continue
        if s.startswith("# "):
            widget.insert("end", s[2:] + "\n", ("h1",))
        elif s.startswith("## "):
            widget.insert("end", s[3:] + "\n", ("h2",))
        elif s.startswith("### "):
            widget.insert("end", s[4:] + "\n", ("h3",))
        elif s.startswith("---") or s.startswith("==="):
            widget.insert("end", "─" * 70 + "\n", ("hr",))
        elif s.startswith("> "):
            _insert_formatted_line(widget, "│  " + s[2:], "quote")
        elif s.startswith("- ") or s.startswith("* "):
            widget.insert("end", " • ", ("bullet", "bullet_dot"))
            _insert_formatted_line(widget, s[2:], "bullet")
        elif re.match(r'^\d+\.\s', s):
            parts = s.split(".", 1)
            num_prefix = parts[0] + "."
            content = parts[1].strip() if len(parts) > 1 else ""
            widget.insert("end", f" {num_prefix} ", ("bullet", "num_dot"))
            _insert_formatted_line(widget, content, "bullet")
        else:
            _insert_formatted_line(widget, s, "p")

    widget.configure(state="disabled")


def show_first_run_agreement(parent: tk.Tk, on_accept, on_decline) -> tk.Toplevel:
    """
    Display the First-Launch Ethical Use, Terms & Privacy Agreement modal.
    Blocks parent until user accepts or declines.
    """
    dlg = tk.Toplevel(parent)
    dlg.title("VoiceTTSr — Ethical Use & Privacy Agreement")
    dlg.geometry("720x620")
    dlg.minsize(650, 560)
    dlg.configure(bg=DARK_BG)
    dlg.transient(parent)
    dlg.grab_set()

    # Center over parent window
    try:
        dlg.update_idletasks()
        px = parent.winfo_rootx() + (parent.winfo_width() // 2) - 360
        py = parent.winfo_rooty() + (parent.winfo_height() // 2) - 310
        dlg.geometry(f"+{max(10, px)}+{max(10, py)}")
    except Exception:
        pass

    # Header
    hdr = tk.Frame(dlg, bg=DARK_BG)
    hdr.pack(fill="x", padx=24, pady=(20, 10))

    tk.Label(
        hdr,
        text="VoiceTTSr Ethical Use & Privacy Agreement",
        font=("Segoe UI", 16, "bold"),
        bg=DARK_BG,
        fg=ACCENT,
    ).pack(anchor="w")

    tk.Label(
        hdr,
        text="Please review the ethical principles, consent requirements, and privacy terms before proceeding.",
        font=("Segoe UI", 10),
        bg=DARK_BG,
        fg=TEXT_SEC,
    ).pack(anchor="w", pady=(4, 0))

    tk.Frame(dlg, bg=BORDER, height=1).pack(fill="x", padx=24, pady=10)

    # Content Container
    content = tk.Frame(dlg, bg=DARK_BG)
    content.pack(fill="both", expand=True, padx=24, pady=5)

    # Card 1: Local Privacy
    c1 = tk.Frame(content, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
    c1.pack(fill="x", pady=6, ipady=4)
    tk.Label(
        c1,
        text="🔒  100% Offline & Local Privacy Guarantee",
        font=("Segoe UI", 11, "bold"),
        bg=CARD_BG,
        fg=ACCENT2,
    ).pack(anchor="w", padx=14, pady=(8, 2))
    tk.Label(
        c1,
        text="All voice models, audio synthesis, and cloned profiles run strictly on your local PC hardware.\nZero telemetry, zero audio, and zero prompts are ever sent to external cloud servers.",
        font=("Segoe UI", 9),
        bg=CARD_BG,
        fg=TEXT_PRI,
        justify="left",
    ).pack(anchor="w", padx=14, pady=(0, 8))

    # Card 2: Voice Cloning Consent & Responsibility
    c2 = tk.Frame(content, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
    c2.pack(fill="x", pady=6, ipady=4)
    tk.Label(
        c2,
        text="🎙️  Voice Cloning Consent & Ethical Use",
        font=("Segoe UI", 11, "bold"),
        bg=CARD_BG,
        fg=ACCENT3,
    ).pack(anchor="w", padx=14, pady=(8, 2))
    tk.Label(
        c2,
        text="You agree to only clone voices that you own or have explicit, informed permission to replicate.\nUnauthorized deepfakes, fraud, defamation, harassment, and deceptive impersonation are strictly prohibited.",
        font=("Segoe UI", 9),
        bg=CARD_BG,
        fg=TEXT_PRI,
        justify="left",
    ).pack(anchor="w", padx=14, pady=(0, 8))

    # Card 3: Upstream Licensing
    c3 = tk.Frame(content, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
    c3.pack(fill="x", pady=5, ipady=3)
    tk.Label(
        c3,
        text="⚖️  Upstream Model Licensing & Watermarking",
        font=("Segoe UI", 10, "bold"),
        bg=CARD_BG,
        fg=TEXT_SEC,
    ).pack(anchor="w", padx=14, pady=(6, 2))
    tk.Label(
        c3,
        text="• Coqui XTTS v2 is governed by the Coqui Public Model License (CPML — non-commercial use only).\n• Chatterbox TTS embeds an imperceptible audio watermark (Perth watermarker) by design.\n• Bethesda Skyrim modding utilities require user-sourced Creation Kit assets.",
        font=("Segoe UI", 8),
        bg=CARD_BG,
        fg=TEXT_PRI,
        justify="left",
    ).pack(anchor="w", padx=14, pady=(0, 6))

    # Card 4: First-Run Model Download Notice
    c4 = tk.Frame(content, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
    c4.pack(fill="x", pady=5, ipady=3)
    tk.Label(
        c4,
        text="⚡  First-Launch Model Downloads & Initialization",
        font=("Segoe UI", 10, "bold"),
        bg=CARD_BG,
        fg=WARNING,
    ).pack(anchor="w", padx=14, pady=(6, 2))
    tk.Label(
        c4,
        text="On first launch or engine switch, required AI model weights are downloaded locally.\nInitial download times vary based on your internet connection. Once downloaded, all synthesis runs 100% offline.",
        font=("Segoe UI", 8),
        bg=CARD_BG,
        fg=TEXT_PRI,
        justify="left",
    ).pack(anchor="w", padx=14, pady=(0, 6))

    tk.Frame(dlg, bg=BORDER, height=1).pack(fill="x", padx=24, pady=10)

    # Footer Buttons
    footer = tk.Frame(dlg, bg=DARK_BG)
    footer.pack(fill="x", padx=24, pady=(5, 18))

    def _open_full():
        show_policy_viewer(dlg)

    btn_view = tk.Button(
        footer,
        text="📖  Read Full Policies",
        font=("Segoe UI", 9),
        bg=PANEL_BG,
        fg=TEXT_PRI,
        activebackground=CARD_BG,
        activeforeground=TEXT_PRI,
        relief="flat",
        padx=12,
        pady=6,
        command=_open_full,
    )
    btn_view.pack(side="left")

    def _handle_decline():
        dlg.destroy()
        if on_decline:
            on_decline()

    def _handle_accept():
        dlg.destroy()
        if on_accept:
            on_accept()

    btn_accept = tk.Button(
        footer,
        text="✓  I Accept & Continue",
        font=("Segoe UI", 10, "bold"),
        bg=ACCENT,
        fg=DARK_BG,
        activebackground=ACCENT2,
        activeforeground=DARK_BG,
        relief="flat",
        padx=18,
        pady=6,
        command=_handle_accept,
    )
    btn_accept.pack(side="right", padx=(8, 0))

    btn_decline = tk.Button(
        footer,
        text="✕  Decline & Exit",
        font=("Segoe UI", 9),
        bg=PANEL_BG,
        fg=DANGER,
        activebackground=CARD_BG,
        activeforeground=DANGER,
        relief="flat",
        padx=14,
        pady=6,
        command=_handle_decline,
    )
    btn_decline.pack(side="right")

    dlg.protocol("WM_DELETE_WINDOW", _handle_decline)
    return dlg


def show_policy_viewer(parent: tk.Tk, initial_tab: int = 0) -> tk.Toplevel:
    """
    Display the full, scrollable Policy & Ethics viewer modal with tabbed sections.
    """
    win = tk.Toplevel(parent)
    win.title("VoiceTTSr — Terms, Ethics & Privacy Policy Viewer")
    win.geometry("820x680")
    win.minsize(700, 500)
    win.configure(bg=DARK_BG)
    win.transient(parent)

    # Header
    hdr = tk.Frame(win, bg=DARK_BG)
    hdr.pack(fill="x", padx=24, pady=(18, 10))

    tk.Label(
        hdr,
        text="⚖️  VoiceTTSr Legal, Ethics & Privacy Documentation",
        font=("Segoe UI", 14, "bold"),
        bg=DARK_BG,
        fg=ACCENT,
    ).pack(side="left")

    tk.Frame(win, bg=BORDER, height=1).pack(fill="x", padx=24, pady=(0, 10))

    # Tabs Frame
    tab_bar = tk.Frame(win, bg=DARK_BG)
    tab_bar.pack(fill="x", padx=24, pady=(0, 6))

    # Viewer Text Area
    text_frame = tk.Frame(win, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
    text_frame.pack(fill="both", expand=True, padx=24, pady=(0, 12))

    viewer = scrolledtext.ScrolledText(
        text_frame,
        wrap="word",
        font=("Segoe UI", 10),
        bg=CARD_BG,
        fg=TEXT_PRI,
        insertbackground=TEXT_PRI,
        relief="flat",
        padx=14,
        pady=12,
    )
    viewer.pack(fill="both", expand=True)

    tabs = [
        ("🎙️ Voice Ethics & Use", _load_doc_file(_VOICE_ETHICS_PATH, "Voice Ethics & Acceptable Use")),
        ("🔒 Privacy Policy", _load_doc_file(_PRIVACY_PATH, "Local Privacy Policy")),
        ("⚖️ Third-Party Notices", _load_doc_file(_THIRD_PARTY_PATH, "Third-Party Notices & Licenses")),
    ]

    tab_buttons = []

    def _select_tab(idx: int):
        for i, b in enumerate(tab_buttons):
            if i == idx:
                b.configure(bg=ACCENT, fg=DARK_BG, font=("Segoe UI", 9, "bold"))
            else:
                b.configure(bg=PANEL_BG, fg=TEXT_SEC, font=("Segoe UI", 9))
        
        render_markdown_to_text(viewer, tabs[idx][1])

    for idx, (label, _) in enumerate(tabs):
        btn = tk.Button(
            tab_bar,
            text=label,
            font=("Segoe UI", 9),
            bg=PANEL_BG,
            fg=TEXT_SEC,
            relief="flat",
            padx=12,
            pady=4,
            command=lambda i=idx: _select_tab(i),
        )
        btn.pack(side="left", padx=(0, 6))
        tab_buttons.append(btn)

    _select_tab(initial_tab)

    # Footer
    footer = tk.Frame(win, bg=DARK_BG)
    footer.pack(fill="x", padx=24, pady=(0, 16))

    btn_close = tk.Button(
        footer,
        text="Close",
        font=("Segoe UI", 9),
        bg=PANEL_BG,
        fg=TEXT_PRI,
        activebackground=CARD_BG,
        activeforeground=TEXT_PRI,
        relief="flat",
        padx=16,
        pady=5,
        command=win.destroy,
    )
    btn_close.pack(side="right")

    return win
