"""
VoiceTTSr GUI  —  v1.6.0 (subprocess worker architecture)
XTTS v2 + Qwen3-TTS run in isolated Python envs via subprocess workers.
No ML imports in this file — zero version conflicts.
Run: python voice_cloner_gui_2.py
GUI deps: pip install pydub pygame numpy
"""

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading, queue, subprocess, os, glob, json, math, struct, wave
from pathlib import Path
from datetime import datetime
import sys, re
from skyrim_utils import SkyrimConverter

# ── pydub lazy import (only used for ref WAV normalization in the GUI) ──────
def _import_pydub():
    from pydub import AudioSegment
    return AudioSegment


# ── constants ──────────────────────────────────────────────────────────────
TARGET_SR       = 24000
TARGET_CHANNELS = 1
SAMPLE_WIDTH    = 2
MIN_DUR_SEC     = 3   # min 3s for refs (6s was too strict, caused silent empty-list rumble)
CONFIG_FILE     = "voicecloner_config.json"
VERSION         = "1.6.0"

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

# ── XTTS v2 Presets (tuned for stable faithful cloning) ───────────────────
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

# ── Qwen3-TTS Presets (tuned for expressiveness & emotion acting) ──────────
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

# ── RVC Presets (pitch, index_rate, f0_method, description) ──────────────
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

# ── Qwen3-TTS emotion → text prefix/suffix map ───────────────────────────
# IMPORTANT: Bracket tags like [breath] [sigh] only work in the full Qwen3
# 8B Instruct model. The 0.6B Base model used here speaks them literally.
# Instead we use the h- prefix (soft/intimate) and speed/punctuation cues
# that the Base model actually responds to via its training data patterns.
QWEN_EMOTION_TAGS = {
    #            (prefix,                     suffix,  description)
    "joy":       ("",                          " :)",   "light upbeat suffix"),
    "love":      ("h- ",                       "",      "soft intimate prefix"),
    "sadness":   ("h- ",                       "...",   "soft + trailing ellipsis"),
    "fear":      ("h- ",                       "",      "soft hushed prefix"),
    "anger":     ("",                          "!!!",   "strong exclamation"),
    "disgust":   ("",                          ".",     "flat period — deadpan"),
    "surprise":  ("",                          "!",     "exclamation"),
    "neutral":   ("",                          "",      "no modification"),
}

# Legacy alias so any older code that still references PRESETS doesn't crash
PRESETS = {**XTTS_PRESETS, **QWEN_PRESETS}

# ── paths to the two isolated Python envs ─────────────────────────────────
# XTTS: uses its own venv (transformers==4.36.2, TTS==0.22.0)
# Qwen:  uses system Python where qwen-tts + transformers==4.57+ are installed
# Auto-locate Python 3.10 based on environment or system fallback
_BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
_XTTS_PYTHON = os.path.join(_BASE_DIR, "xtts-env-py310", "Scripts", "python.exe")
_QWEN_ENV    = os.path.join(_BASE_DIR, "qwen-env-py310", "Scripts", "python.exe")
_RVC_PYTHON  = os.path.join(_BASE_DIR, "rvc-env", "Scripts", "python.exe")

# Fallback to system python if venvs aren't found
_QWEN_PYTHON = _QWEN_ENV if os.path.isfile(_QWEN_ENV) else sys.executable


# ══════════════════════════════════════════════════════════════════════════
class _TtsWorker:
    """Subprocess wrapper for xtts_worker.py / qwen_worker.py.
    Communicates via JSON lines over stdin/stdout.
    Stays alive between generations (model stays in VRAM).
    """
    def __init__(self, python_exe, script, on_log, on_status):
        self._python     = python_exe
        self._script     = script
        self._on_log     = on_log       # callable(text, level)
        self._on_status  = on_status    # callable(status_text, color)
        self._on_chunk   = None         # callable(path) — streaming audio chunk
        self._proc       = None
        self._ready_evt  = threading.Event()
        self._resp_queue = queue.Queue()
        self._starting   = False

    def is_alive(self):
        return self._proc is not None and self._proc.poll() is None

    def start(self, on_ready, on_error):
        """Spawn worker subprocess. Calls on_ready() or on_error(msg) when done."""
        if self.is_alive():
            on_ready(); return
        if not os.path.isfile(self._python):
            on_error(f"Python not found: {self._python}\nRun setup_qwen_env.bat to create the Qwen env."); return
        if not os.path.isfile(self._script):
            on_error(f"Worker script not found: {self._script}"); return

        self._ready_evt.clear()
        self._starting = True
        # Set PYTHONIOENCODING=utf-8 so workers can print Hindi/CJK/etc
        # without crashing on Windows cp1252 console encoding.
        worker_env = os.environ.copy()
        worker_env["PYTHONIOENCODING"] = "utf-8"
        self._proc = subprocess.Popen(
            [self._python, "-u", self._script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            encoding="utf-8",
            env=worker_env,
        )
        # reader thread
        t = threading.Thread(target=self._reader_loop, daemon=True)
        t.start()
        # watcher thread
        threading.Thread(target=self._wait_ready,
                         args=(on_ready, on_error), daemon=True).start()

    def _reader_loop(self):
        for raw in self._proc.stdout:
            raw = raw.strip()
            if not raw: continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                # plain text fallback
                self._on_log(raw, "info"); continue

            s = msg.get("status", "")
            if s == "ready":
                self._ready_evt.set()
            elif s == "log":
                lvl = msg.get("level", "info")
                self._on_log(msg.get("text", ""), lvl)
            elif s == "chunk":
                if self._on_chunk and msg.get("file"):
                    self._on_chunk(msg["file"])
            else:
                self._resp_queue.put(msg)

    def _wait_ready(self, on_ready, on_error):
        ok = self._ready_evt.wait(timeout=180)
        self._starting = False
        if ok:
            on_ready()
        else:
            rc = self._proc.poll()
            on_error(f"Worker failed to start (exit code={rc}).")

    def send(self, cmd: dict):
        if self.is_alive():
            try:
                self._proc.stdin.write(json.dumps(cmd) + "\n")
                self._proc.stdin.flush()
            except BrokenPipeError:
                self._on_log("Worker pipe broken.", "error")

    def get_response(self, timeout=180):
        try:
            return self._resp_queue.get(timeout=timeout)
        except queue.Empty:
            return {"status": "error", "message": "Response timeout."}

    def stop(self):
        if self.is_alive():
            try:
                self.send({"action": "quit"})
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()


# ══════════════════════════════════════════════════════════════════════════
class VoiceClonerApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title(f"VoiceTTSr Studio — v{VERSION}")
        self.geometry("1700x940")
        self.minsize(1200, 740)
        self.configure(bg=DARK_BG)

        # subprocess workers (one per backend)
        self._xtts_worker = _TtsWorker(
            python_exe=_XTTS_PYTHON,
            script="xtts_worker.py",
            on_log=self._worker_log,
            on_status=self._set_status,
        )
        self._qwen_worker = _TtsWorker(
            python_exe=_QWEN_PYTHON,
            script="qwen_worker.py",
            on_log=self._worker_log,
            on_status=self._set_status,
        )
        self._rvc_worker = _TtsWorker(
            python_exe=_RVC_PYTHON,
            script="rvc_worker.py",
            on_log=self._worker_log,
            on_status=self._set_status,
        )
        # legacy shims — parts of the UI still check these
        self._tts_model      = None   # True once XTTS worker is ready
        self._qwen_model     = None   # True once Qwen worker is ready
        self._tts_lock       = threading.Lock()
        self._jobs_active    = 0
        self._voice_profile  = None
        self._profile_path   = None
        self._active_preset  = tk.StringVar(value="")
        self._backend_var    = tk.StringVar(value="xtts")
        self._bert_director  = tk.BooleanVar(value=False)
        self._bert_pipeline  = None
        self._load_config()
        self._qwen_voice_design = tk.BooleanVar(value=self.config_data.get("qwen_voicedesign", False))
        self._qwen_emotion_tags = tk.BooleanVar(value=self.config_data.get("qwen_emotion_tags", False))
        self._qwen_retry_mumble = tk.BooleanVar(value=self.config_data.get("qwen_retry_mumble", True))
        self._qwen_stream       = tk.BooleanVar(value=self.config_data.get("qwen_stream", False))
        self._qwen_profile_pth  = None
        self._qwen_profile_var  = tk.StringVar(value="")
        self._xtts_audio_pro    = tk.BooleanVar(value=self.config_data.get("xtts_audio_pro", True))
        self._rvc_enabled       = tk.BooleanVar(value=self.config_data.get("rvc_enabled", False))
        self._rvc_auto_var      = tk.BooleanVar(value=self.config_data.get("rvc_auto", False))
        self._rvc_auto_scope    = tk.StringVar(value=self.config_data.get("rvc_auto_scope", "per-job"))
        self._rvc_model_var     = tk.StringVar(value=self.config_data.get("rvc_model", ""))
        self._rvc_pitch_var     = tk.IntVar(value=self.config_data.get("rvc_pitch", 0))
        self._rvc_index_var     = tk.DoubleVar(value=self.config_data.get("rvc_index", 0.75))
        self._rvc_method_var    = tk.StringVar(value=self.config_data.get("rvc_method", "rmvpe"))

        # ── Skyrim SE settings ──
        self._skyrim_enabled    = tk.BooleanVar(value=self.config_data.get("skyrim_enabled", False))
        self._skyrim_plugin     = tk.StringVar(value=self.config_data.get("skyrim_plugin", "PlayerVoice.esp"))
        self._skyrim_voice_type = tk.StringVar(value=self.config_data.get("skyrim_voice_type", "MalePC"))
        self._skyrim_facefx_path= tk.StringVar(value=self.config_data.get("skyrim_facefx_path", os.path.join("tools", "FaceFXWrapper.exe")))
        self._skyrim_xwma_path  = tk.StringVar(value=self.config_data.get("skyrim_xwma_path", os.path.join("tools", "xWMAEncode.exe")))
        self._skyrim_fonix_path = tk.StringVar(value=self.config_data.get("skyrim_fonix_path", os.path.join("tools", "FonixData.cdf")))

        # generation stop flag
        self._stop_generation = threading.Event()

        # ── Batch settings (session-only) ──
        self._naming_mode       = tk.StringVar(value="Normal")
        self._batch_out_enabled = tk.BooleanVar(value=False)
        self._batch_out_folder  = tk.StringVar(value="")
        self._batch_size_var    = tk.IntVar(value=self.config_data.get("batch_size", 10))

        # granular control flags
        self._skip_job_event    = threading.Event()
        self._skip_batch_event  = threading.Event()
        self._stop_batch_event  = threading.Event()

        # playback state
        self._pygame_ok      = False
        self._current_file   = None
        self._playing        = False
        self._play_thread    = None

        # auto-create separate reference folders
        os.makedirs("references/xtts", exist_ok=True)
        os.makedirs("references/qwen",  exist_ok=True)
        os.makedirs("output",           exist_ok=True)

        self._init_pygame()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── pygame audio init ──────────────────────────────────────────────────
    def _init_pygame(self):
        try:
            import pygame
            pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=1024)
            self._pygame_ok = True
        except Exception:
            self._pygame_ok = False

    # ── config ─────────────────────────────────────────────────────────────
    def _load_config(self):
        self.config_data = {}
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE) as f:
                    self.config_data = json.load(f)
            except Exception:
                pass

    def _save_config(self):
        try:
            d = self.config_data
            # per-backend folders
            d["xtts_ref_folder"]     = self.xtts_ref_folder_var.get()
            d["qwen_ref_folder"]     = self.qwen_ref_folder_var.get()
            d["out_folder"]          = self.out_folder_var.get()
            d["language"]            = self._lang_map.get(self.lang_var.get(), "en")
            # xtts settings
            d["xtts_speed"]          = self.xtts_speed_var.get()
            d["xtts_temperature"]    = self.xtts_temp_var.get()
            d["xtts_rep_pen"]        = self.xtts_rep_pen_var.get()
            d["xtts_top_k"]          = self.xtts_top_k_var.get()
            d["xtts_top_p"]          = self.xtts_top_p_var.get()
            # qwen settings
            d["qwen_temperature"]    = self.qwen_temp_var.get()
            d["qwen_rep_pen"]        = self.qwen_rep_pen_var.get()
            d["qwen_top_k"]          = self.qwen_top_k_var.get()
            d["qwen_top_p"]          = self.qwen_top_p_var.get()
            d["backend"]             = self._backend_var.get()
            d["qwen_voicedesign"]    = self._qwen_voice_design.get()
            d["qwen_emotion_tags"]   = self._qwen_emotion_tags.get()
            d["qwen_retry_mumble"]   = self._qwen_retry_mumble.get()
            d["qwen_stream"]         = self._qwen_stream.get()
            d["xtts_audio_pro"]      = self._xtts_audio_pro.get()
            d["rvc_enabled"]         = self._rvc_enabled.get()
            d["rvc_model"]           = self._rvc_model_var.get()
            d["rvc_pitch"]           = self._rvc_pitch_var.get()
            d["rvc_index"]           = self._rvc_index_var.get()
            d["rvc_method"]          = self._rvc_method_var.get()
            d["rvc_auto"]             = self._rvc_auto_var.get()
            d["rvc_auto_scope"]       = self._rvc_auto_scope.get()
            # skyrim settings
            d["skyrim_enabled"]       = self._skyrim_enabled.get()
            d["skyrim_plugin"]        = self._skyrim_plugin.get()
            d["skyrim_voice_type"]    = self._skyrim_voice_type.get()
            d["skyrim_facefx_path"]   = self._skyrim_facefx_path.get()
            d["skyrim_xwma_path"]     = self._skyrim_xwma_path.get()
            d["skyrim_fonix_path"]    = self._skyrim_fonix_path.get()
            if self._profile_path:
                d["profile_path"] = self._profile_path
            with open(CONFIG_FILE, "w") as f:
                json.dump(d, f, indent=2)
        except Exception:
            pass

    # ── UI skeleton ────────────────────────────────────────────────────────
    def _build_ui(self):
        self._apply_styles()

        # ── header ──
        hdr = tk.Frame(self, bg=DARK_BG)
        hdr.pack(fill="x", padx=24, pady=(18, 0))
        self._title_lbl = tk.Label(hdr, text="VoiceTTSr",
                                    font=("Courier New", 22, "bold"),
                                    bg=DARK_BG, fg=ACCENT)
        self._title_lbl.pack(side="left")
        self._subtitle_lbl = tk.Label(hdr, text="  XTTS v2 voice cloning studio",
                                       font=("Courier New", 11),
                                       bg=DARK_BG, fg=TEXT_SEC)
        self._subtitle_lbl.pack(side="left", pady=4)
        # Audio Analyzer button in header
        self._btn(hdr, "🔬 Audio Analyzer", self._open_audio_analyzer,
                  small=True).pack_configure(side="right", padx=(0, 12))

        self._status_dot = tk.Label(hdr, text="●", font=("Arial", 13),
                                     bg=DARK_BG, fg=TEXT_MUT)
        self._status_dot.pack(side="right", padx=(0, 4))
        self._status_lbl = tk.Label(hdr, text="idle",
                                     font=("Courier New", 10), bg=DARK_BG, fg=TEXT_MUT)
        self._status_lbl.pack(side="right")

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=24, pady=10)

        # ── three-column body ──
        body = tk.Frame(self, bg=DARK_BG)
        body.pack(fill="both", expand=True, padx=24, pady=(0, 16))
        body.columnconfigure(0, weight=2)
        body.columnconfigure(1, weight=7)   # Maximized weight for jobs panel
        body.columnconfigure(2, weight=1)   # Minimal weight for log panel
        body.rowconfigure(0, weight=1)

        # left scrollable
        lo = tk.Frame(body, bg=DARK_BG)
        lo.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        lo.rowconfigure(0, weight=1); lo.columnconfigure(0, weight=1)
        lc = tk.Canvas(lo, bg=DARK_BG, highlightthickness=0)
        ls = ttk.Scrollbar(lo, orient="vertical", command=lc.yview)
        lc.configure(yscrollcommand=ls.set)
        ls.grid(row=0, column=1, sticky="ns")
        lc.grid(row=0, column=0, sticky="nsew")
        left = tk.Frame(lc, bg=DARK_BG)
        lw   = lc.create_window((0, 0), window=left, anchor="nw")
        left.bind("<Configure>", lambda e: lc.configure(scrollregion=lc.bbox("all")))
        lc.bind("<Configure>",   lambda e: lc.itemconfig(lw, width=e.width))
        # hover-based localized scrolling
        def _on_left_mousewheel(event):
            lc.yview_scroll(int(-1*(event.delta/120)), "units")
        lc.bind("<Enter>", lambda e: self.bind_all("<MouseWheel>", _on_left_mousewheel))
        lc.bind("<Leave>", lambda e: self.unbind_all("<MouseWheel>"))

        # middle scrollable
        mo = tk.Frame(body, bg=DARK_BG)
        mo.grid(row=0, column=1, sticky="nsew", padx=(0, 10))
        mo.rowconfigure(0, weight=1); mo.columnconfigure(0, weight=1)
        mc = tk.Canvas(mo, bg=DARK_BG, highlightthickness=0)
        msc= ttk.Scrollbar(mo, orient="vertical", command=mc.yview)
        mc.configure(yscrollcommand=msc.set)
        msc.grid(row=0, column=1, sticky="ns")
        mc.grid(row=0, column=0, sticky="nsew")
        mid = tk.Frame(mc, bg=DARK_BG)
        mw  = mc.create_window((0, 0), window=mid, anchor="nw")
        mid.bind("<Configure>", lambda e: mc.configure(scrollregion=mc.bbox("all")))
        mc.bind("<Configure>",  lambda e: mc.itemconfig(mw, width=e.width))

        # middle column mousewheel scroll
        def _on_mid_mousewheel(event):
            mc.yview_scroll(int(-1*(event.delta/120)), "units")
        mc.bind("<Enter>",  lambda e: self.bind_all("<MouseWheel>", _on_mid_mousewheel))
        mc.bind("<Leave>",  lambda e: self.unbind_all("<MouseWheel>"))
        mid.bind("<Enter>", lambda e: self.bind_all("<MouseWheel>", _on_mid_mousewheel))
        mid.bind("<Leave>", lambda e: self.unbind_all("<MouseWheel>"))

        # right log
        right = tk.Frame(body, bg=DARK_BG)
        right.grid(row=0, column=2, sticky="nsew")

        # build panels
        self._build_backend(left)
        self._build_folders(left)
        self._build_converter(left)
        self._build_presets(left)
        self._build_settings(left)
        self._build_rvc_panel(left)
        self._build_skyrim_panel(left)
        self._build_voice_profile(left)
        self._build_qwen_profile(left)

        self._build_jobs(mid)
        self._build_output_player(mid)

        self._build_log(right)

    def _apply_styles(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("Vertical.TScrollbar", background=CARD_BG,
                    troughcolor=PANEL_BG, borderwidth=0, arrowsize=12)
        s.configure("TProgressbar", troughcolor=CARD_BG,
                    background=ACCENT, borderwidth=0, thickness=4)

    # ── widget helpers ─────────────────────────────────────────────────────
    def _card(self, parent, title, accent_color=None):
        outer = tk.Frame(parent, bg=CARD_BG, bd=0,
                         highlightbackground=accent_color or BORDER,
                         highlightthickness=1)
        outer.pack(fill="x", pady=(0, 10))
        tk.Label(outer, text=title, font=("Courier New", 9, "bold"),
                 bg=CARD_BG, fg=accent_color or TEXT_SEC
                 ).pack(anchor="w", padx=12, pady=(10, 4))
        inner = tk.Frame(outer, bg=CARD_BG)
        inner.pack(fill="x", padx=12, pady=(0, 10))
        return inner

    def _btn(self, parent, text, cmd, accent=False, small=False, color=None, fg_col=None):
        fg = fg_col or ("#ffffff" if accent else TEXT_PRI)
        bg = color  or (ACCENT  if accent else CARD_BG)
        ab = ACCENT2 if accent else BORDER
        
        if accent:
            f = ("Courier New", 8, "bold") if small else ("Courier New", 9, "bold")
        else:
            f = ("Courier New", 8) if small else ("Courier New", 9)

        b  = tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg,
                       activebackground=ab, activeforeground=DARK_BG,
                       relief="flat", bd=0, padx=10, pady=4, font=f,
                       cursor="hand2")
        b.pack(side="left")
        return b

    def _slider_row(self, parent, label, var, from_, to, resolution, hint=""):
        row = tk.Frame(parent, bg=CARD_BG)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=label, width=17, anchor="w",
                 font=("Courier New", 9), bg=CARD_BG, fg=TEXT_SEC).pack(side="left")
        val_lbl = tk.Label(row, text=f"{var.get():.2f}", width=5,
                           font=("Courier New", 9), bg=CARD_BG, fg=ACCENT)
        val_lbl.pack(side="right")
        def _upd(v, lbl=val_lbl):
            lbl.config(text=f"{float(v):.2f}")
            self._active_preset.set("")
        tk.Scale(row, from_=from_, to=to, resolution=resolution,
                 variable=var, orient="horizontal",
                 bg=CARD_BG, fg=TEXT_PRI, troughcolor=PANEL_BG,
                 highlightthickness=0, bd=0, sliderlength=14,
                 showvalue=False, command=_upd
                 ).pack(side="left", fill="x", expand=True, padx=(0, 6))
        if hint:
            tk.Label(parent, text=hint, font=("Courier New", 7),
                     bg=CARD_BG, fg=TEXT_MUT).pack(anchor="w")
        return val_lbl

    # ══ LEFT PANEL ══════════════════════════════════════════════════════════

    # ── backend ─────────────────────────────────────────────────────────────
    def _build_backend(self, parent):
        c = self._card(parent, "ENGINE")
        row = tk.Frame(c, bg=CARD_BG); row.pack(fill="x")

        def _pick(val):
            self._backend_var.set(val)
            is_xtts = val == "xtts"
            self._btn_xtts.config(bg=ACCENT  if is_xtts else CARD_BG,
                                   fg=DARK_BG if is_xtts else TEXT_SEC)
            self._btn_qwen.config(bg=ACCENT3 if not is_xtts else CARD_BG,
                                   fg=DARK_BG if not is_xtts else TEXT_SEC)
            self._subtitle_lbl.config(
                text="  XTTS v2 voice cloning studio" if is_xtts
                else "  Qwen3-TTS voice cloning studio")

            # swap voice profile card (XTTS only)
            if is_xtts:
                self._profile_card_frame.pack(fill="x", pady=(0, 10))
                self._qwen_profile_card_frame.pack_forget()
                self._rvc_panel_frame.pack(fill="x", pady=(0, 10))
            else:
                self._profile_card_frame.pack_forget()
                self._qwen_profile_card_frame.pack(fill="x", pady=(0, 10))
                self._rvc_panel_frame.pack_forget()

            # swap Qwen power mode toggle
            if not is_xtts:
                self._qwen_vd_frame.pack(fill="x", pady=(8, 0))
            else:
                self._qwen_vd_frame.pack_forget()

            # swap settings panels
            if is_xtts:
                self._qwen_settings_frame.pack_forget()
                self._xtts_settings_frame.pack(fill="x")
                self._enhance_refs_frame.pack(fill="x", pady=(8, 0))
            else:
                self._xtts_settings_frame.pack_forget()
                self._qwen_settings_frame.pack(fill="x")
                self._enhance_refs_frame.pack_forget()

            # rebuild preset buttons for active backend
            self._rebuild_preset_grid(val)

            # update job mood dropdowns
            self._update_job_mood_values()

        self._btn_xtts = self._btn(row, "XTTS v2  (stable)",
                                    lambda: _pick("xtts"), accent=True, small=True)
        tk.Frame(row, width=8, bg=CARD_BG).pack(side="left")
        self._btn_qwen = self._btn(row, "Qwen3-TTS  (higher quality)",
                                    lambda: _pick("qwen"), small=True)

        info = tk.Frame(c, bg=CARD_BG); info.pack(fill="x", pady=(6, 0))
        tk.Label(info, text="XTTS v2: 17 langs · 4 GB · proven stable",
                 font=("Courier New", 7), bg=CARD_BG, fg=TEXT_MUT).pack(anchor="w")
        tk.Label(info,
                 text="Qwen3-TTS: 10 langs · 6-8 GB · better quality"
                      "  needs: pip install transformers==4.37.2 soundfile accelerate",
                 font=("Courier New", 7), bg=CARD_BG, fg=TEXT_MUT,
                 wraplength=420, justify="left").pack(anchor="w")

        load_row = tk.Frame(c, bg=CARD_BG); load_row.pack(fill="x", pady=(8, 0))
        self._load_model_btn = self._btn(load_row, "Load Model",
                                          self._load_model, accent=True)
        self._model_lbl = tk.Label(load_row, text="not loaded",
                                    font=("Courier New", 8), bg=CARD_BG, fg=TEXT_MUT)
        self._model_lbl.pack(side="left", padx=8)

        # Qwen Power Mode toggle (Qwen-only)
        self._qwen_vd_frame = tk.Frame(c, bg=CARD_BG)
        vd_btn = tk.Checkbutton(self._qwen_vd_frame, text="High-Emotion Power Mode (ICL)",
                                 variable=self._qwen_voice_design,
                                 bg=CARD_BG, fg=ACCENT3, selectcolor=DARK_BG,
                                 activebackground=CARD_BG, activeforeground=ACCENT3,
                                 font=("Courier New", 8, "bold"), bd=0, highlightthickness=0)
        vd_btn.pack(side="left")
        tk.Label(self._qwen_vd_frame,
                 text=" — enables ICL emotion acting (may cause mumbling on some voices)",
                 font=("Courier New", 7), bg=CARD_BG, fg=TEXT_MUT).pack(side="left")

        saved = self.config_data.get("backend", "xtts")
        self._backend_var.set(saved)
        self.after(0, lambda: _pick(saved))

    # ── folders ─────────────────────────────────────────────────────────────
    def _build_folders(self, parent):
        c = self._card(parent, "FOLDERS")

        # backward-compat: if old config has 'ref_folder', migrate it
        old_ref = self.config_data.get("ref_folder", "")

        # Smart XTTS default: if references/xtts is empty/missing but old 'references' has WAVs, use that
        xtts_default = self.config_data.get("xtts_ref_folder", "")
        if not xtts_default:
            if old_ref and os.path.isdir(old_ref) and glob.glob(os.path.join(old_ref, "*.wav")):
                xtts_default = old_ref  # migrate legacy path
            else:
                xtts_default = "references/xtts"

        self.xtts_ref_folder_var = tk.StringVar(value=xtts_default)
        self.qwen_ref_folder_var = tk.StringVar(
            value=self.config_data.get("qwen_ref_folder", "references/qwen"))
        self.out_folder_var = tk.StringVar(
            value=self.config_data.get("out_folder", "output"))
        # keep legacy ref_folder_var pointing at active backend
        self.ref_folder_var = self.xtts_ref_folder_var  # updated in _pick()

        def _browse(var, title):
            p = filedialog.askdirectory(title=title)
            if p: var.set(p)

        for label, var, title in [
            ("XTTS References", self.xtts_ref_folder_var, "XTTS Reference folder (6-30 WAV files)"),
            ("Qwen References", self.qwen_ref_folder_var, "Qwen Reference folder (can use many more files)"),
            ("Output",          self.out_folder_var,      "Output folder"),
        ]:
            r = tk.Frame(c, bg=CARD_BG); r.pack(fill="x", pady=2)
            tk.Label(r, text=label, width=14, anchor="w",
                     font=("Courier New", 9), bg=CARD_BG, fg=TEXT_SEC).pack(side="left")
            tk.Entry(r, textvariable=var, bg=PANEL_BG, fg=TEXT_PRI,
                     insertbackground=TEXT_PRI, relief="flat", bd=4,
                     font=("Courier New", 9)).pack(side="left", fill="x",
                                                   expand=True, padx=(0, 4))
            self._btn(r, "Browse", lambda v=var, t=title: _browse(v, t), small=True)

        tk.Label(c, text="Tip: Qwen works well with 10-30 diverse reference clips.",
                 font=("Courier New", 7), bg=CARD_BG, fg=TEXT_MUT).pack(anchor="w", pady=(4, 0))

        # Enhance Refs button (Option B) - visible for XTTS
        self._enhance_refs_frame = tk.Frame(c, bg=CARD_BG)
        self._enhance_refs_frame.pack(fill="x", pady=(8, 0))
        self._btn(self._enhance_refs_frame, "✨ Clean & Enhance XTTS References (Option B)", 
                  self._enhance_xtts_refs, accent=True)
        tk.Label(self._enhance_refs_frame, 
                 text=" Trims silence & normalizes loudness of source clips.",
                 font=("Courier New", 7), bg=CARD_BG, fg=TEXT_MUT).pack(side="left", padx=8)

    # ── converter ───────────────────────────────────────────────────────────
    def _build_converter(self, parent):
        c = self._card(parent, "MP3 → WAV CONVERTER")
        self._conv_lbl = tk.Label(c, text="No MP3s scanned yet",
                                   font=("Courier New", 8), bg=CARD_BG, fg=TEXT_MUT)
        self._conv_lbl.pack(anchor="w", pady=(0, 6))
        btns = tk.Frame(c, bg=CARD_BG); btns.pack(fill="x")
        self._btn(btns, "Scan",        self._scan_mp3s,    small=True)
        tk.Frame(btns, width=8, bg=CARD_BG).pack(side="left")
        self._btn(btns, "Convert All", self._convert_mp3s, accent=True, small=True)
        self._conv_progress = ttk.Progressbar(c, mode="determinate", style="TProgressbar")
        self._conv_progress.pack(fill="x", pady=(6, 0))

    def _scan_mp3s(self):
        folder = self.ref_folder_var.get()
        mp3s   = glob.glob(os.path.join(folder, "*.mp3")) + glob.glob(os.path.join(folder, "*.MP3"))
        self._conv_lbl.config(text=f"{len(mp3s)} MP3(s) in '{folder}'",
                               fg=WARNING if mp3s else TEXT_MUT)

    def _convert_mp3s(self):
        folder = self.ref_folder_var.get()
        mp3s   = glob.glob(os.path.join(folder, "*.mp3")) + glob.glob(os.path.join(folder, "*.MP3"))
        if not mp3s:
            messagebox.showinfo("No MP3s", "No MP3 files found.")
            return
        threading.Thread(target=self._run_conversion, args=(mp3s,), daemon=True).start()

    def _run_conversion(self, mp3s):
        AudioSegment = _import_pydub()
        total = len(mp3s); conv = skip = fail = 0
        self._log(f"Converting {total} MP3(s)…", WARNING)
        self._conv_progress["maximum"] = total; self._conv_progress["value"] = 0
        for i, mp3 in enumerate(mp3s, 1):
            p = Path(mp3); wav_out = p.with_suffix(".wav")
            if wav_out.exists():
                self._log(f"  skip  {p.name}", TEXT_MUT); skip += 1
            else:
                try:
                    audio = (AudioSegment.from_file(p)
                             .set_channels(TARGET_CHANNELS)
                             .set_frame_rate(TARGET_SR)
                             .set_sample_width(SAMPLE_WIDTH))
                    audio.export(wav_out, format="wav")
                    dur = len(audio)/1000
                    self._log(f"  ok  {p.name}  {dur:.1f}s"
                              + (" ⚠short" if dur < MIN_DUR_SEC else ""), ACCENT2)
                    conv += 1
                except Exception as e:
                    self._log(f"  fail  {p.name}  {e}", DANGER); fail += 1
            self.after(0, lambda v=i: self._conv_progress.configure(value=v))
        self._log(f"Done — {conv} converted, {skip} skipped, {fail} failed", ACCENT2)
        self.after(0, lambda: self._conv_lbl.config(
            text=f"Converted {conv} | Skipped {skip} | Failed {fail}", fg=ACCENT2))

    # ── presets ──────────────────────────────────────────────────────────────
    def _build_presets(self, parent):
        c = self._card(parent, "VOICE PRESETS")
        self._preset_grid_frame = tk.Frame(c, bg=CARD_BG)
        self._preset_grid_frame.pack(fill="x", pady=(0, 8))
        self._preset_btns = {}
        # Initial build for xtts
        self._rebuild_preset_grid("xtts")

        det_row = tk.Frame(c, bg=CARD_BG); det_row.pack(fill="x")
        self._btn(det_row, "Auto-Detect from WAVs", self._auto_detect_preset, small=True)
        tk.Frame(det_row, width=10, bg=CARD_BG).pack(side="left")
        self._preset_lbl = tk.Label(det_row, text="no preset selected",
                                     font=("Courier New", 8), bg=CARD_BG, fg=TEXT_MUT)
        self._preset_lbl.pack(side="left")

        bert_row = tk.Frame(c, bg=CARD_BG); bert_row.pack(fill="x", pady=(4, 0))
        self._bert_btn = tk.Button(bert_row, text="Auto-Director (BERT): OFF",
                                   font=("Courier New", 8, "bold"),
                                   bg=PANEL_BG, fg=TEXT_MUT,
                                   activebackground=ACCENT, activeforeground=DARK_BG,
                                   relief="flat", bd=0, padx=10, pady=4, cursor="hand2",
                                   command=self._toggle_bert_director)
        self._bert_btn.pack(side="left")
        tk.Label(c,
                 text="BERT Auto-Director analyses your text and assigns the best mood preset per line.",
                 font=("Courier New", 7), bg=CARD_BG, fg=TEXT_MUT,
                 wraplength=420, justify="left").pack(anchor="w", pady=(4, 0))

    def _rebuild_preset_grid(self, backend):
        """Rebuild the preset button grid for the given backend."""
        for w in self._preset_grid_frame.winfo_children():
            w.destroy()
        self._preset_btns = {}
        active_presets = XTTS_PRESETS if backend == "xtts" else QWEN_PRESETS
        cols = 4
        for i, (name, vals) in enumerate(active_presets.items()):
            btn = tk.Button(self._preset_grid_frame, text=name, font=("Courier New", 8),
                            bg=PANEL_BG, fg=TEXT_SEC,
                            activebackground=ACCENT, activeforeground=DARK_BG,
                            relief="flat", bd=0, padx=6, pady=5, cursor="hand2",
                            command=lambda n=name: self._apply_preset(n))
            btn.grid(row=i//cols, column=i%cols, padx=3, pady=3, sticky="ew")
            self._preset_btns[name] = btn
        for col in range(cols):
            self._preset_grid_frame.columnconfigure(col, weight=1)
        # Re-apply active preset highlight if there is one
        active = self._active_preset.get()
        if active in self._preset_btns:
            self._preset_btns[active].config(bg=ACCENT, fg=DARK_BG)

    def _apply_preset(self, name, silent=False):
        backend = self._backend_var.get()
        active_presets = XTTS_PRESETS if backend == "xtts" else QWEN_PRESETS
        if name not in active_presets: return
        temp, speed, rep, topk, topp, desc = active_presets[name]
        if backend == "xtts":
            self.xtts_temp_var.set(temp); self.xtts_speed_var.set(speed)
            self.xtts_rep_pen_var.set(rep); self.xtts_top_k_var.set(topk); self.xtts_top_p_var.set(topp)
            for lbl, val in [(self._lbl_xtts_temp, temp), (self._lbl_xtts_speed, speed),
                             (self._lbl_xtts_rep,  rep),  (self._lbl_xtts_topk,  topk),
                             (self._lbl_xtts_topp, topp)]:
                lbl.config(text=f"{val:.2f}")
        else:
            self.qwen_temp_var.set(temp)
            self.qwen_rep_pen_var.set(rep); self.qwen_top_k_var.set(topk); self.qwen_top_p_var.set(topp)
            for lbl, val in [(self._lbl_qwen_temp, temp),
                             (self._lbl_qwen_rep,  rep),  (self._lbl_qwen_topk, topk),
                             (self._lbl_qwen_topp, topp)]:
                lbl.config(text=f"{val:.2f}")
        self._active_preset.set(name)
        for n, btn in self._preset_btns.items():
            btn.config(bg=ACCENT if n == name else PANEL_BG,
                       fg=DARK_BG if n == name else TEXT_SEC)
        self._preset_lbl.config(text=f"active: {name} — {desc}", fg=ACCENT2)
        if not silent:
            self._log(f"Preset [{backend.upper()}]: {name} — {desc}", ACCENT)

    def _auto_detect_preset(self):
        valid_wavs, _ = self._get_ref_wavs()
        if not valid_wavs:
            messagebox.showwarning("No WAVs", "Convert MP3s first.")
            return
        threading.Thread(target=self._run_auto_detect,
                         args=(valid_wavs,), daemon=True).start()

    def _run_auto_detect(self, wav_files):
        self._log("Analysing reference audio…", WARNING)
        try:
            AudioSegment = _import_pydub()
            import numpy as np
            pv_list, rate_list, eng_list = [], [], []
            for wav in wav_files:
                try:
                    seg     = AudioSegment.from_file(wav).set_channels(1)
                    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
                    sr      = seg.frame_rate
                    if samples.max() != 0: samples /= np.abs(samples).max()
                    eng_list.append(float(np.sqrt(np.mean(samples**2))))
                    fl = int(sr*0.025); hop = int(sr*0.010)
                    frames  = [samples[i:i+fl] for i in range(0, len(samples)-fl, hop)]
                    rms_arr = np.array([np.sqrt(np.mean(f**2)) for f in frames])
                    pv_list.append(float(np.std(rms_arr)))
                    rate_list.append(float(np.mean(np.abs(np.diff(np.sign(samples))))/2))
                except Exception as e:
                    self._log(f"  skip {os.path.basename(wav)}: {e}", TEXT_MUT)
            if not eng_list:
                self._log("Could not analyse any WAV files.", DANGER); return
            pv   = float(np.mean(pv_list))
            rate = float(np.mean(rate_list))
            eng  = float(np.mean(eng_list))
            pv_n   = min(pv/0.08,  1.0)
            rate_n = min(rate/0.15, 1.0)
            eng_n  = min(eng/0.25,  1.0)
            self._log(f"  features → pv={pv_n:.2f}  rate={rate_n:.2f}  eng={eng_n:.2f}", TEXT_MUT)
            feat_temp  = 0.4 + pv_n  * 0.4
            feat_speed = 0.75 + rate_n * 0.65
            backend = self._backend_var.get()
            active_presets = XTTS_PRESETS if backend == "xtts" else QWEN_PRESETS
            best, best_d = "Natural", float("inf")
            for name, vals in active_presets.items():
                d = math.sqrt((vals[0]-feat_temp)**2 + (vals[1]-feat_speed)**2)
                if d < best_d: best_d = d; best = name
            if eng_n < 0.25 and pv_n > 0.5:  best = "Breathy" if "Breathy" in active_presets else best
            elif eng_n > 0.75 and rate_n > 0.6: best = "Crisp" if "Crisp" in active_presets else best
            elif eng_n > 0.80 and pv_n > 0.70: best = "Aggressive" if "Aggressive" in active_presets else best
            elif eng_n < 0.30 and rate_n < 0.35: best = "Warm"
            self._log(f"Auto-detected preset: {best} [{backend.upper()}]", ACCENT2)
            self.after(0, lambda: self._apply_preset(best, silent=True))
        except Exception as e:
            self._log(f"Auto-detect failed: {e}", DANGER)
    def _toggle_bert_director(self):
        new_state = not self._bert_director.get()
        self._bert_director.set(new_state)
        if new_state:
            self._bert_btn.config(text="Auto-Director (BERT): ON", bg=ACCENT, fg=DARK_BG)
            self._log("Auto-Director (BERT) enabled. It will tag new jobs based on sentiment.", ACCENT)
            if not self._bert_pipeline:
                threading.Thread(target=self._load_bert, daemon=True).start()
        else:
            self._bert_btn.config(text="Auto-Director (BERT): OFF", bg=PANEL_BG, fg=TEXT_MUT)
            self._log("Auto-Director disabled.", TEXT_MUT)

    def _load_bert(self):
        self._log("Spawning BERT emotion brain…", WARNING)
        try:
            from transformers import pipeline
            self._bert_pipeline = pipeline("text-classification",
                                          model="bhadresh-savani/distilbert-base-uncased-emotion",
                                          device=-1) # -1 for CPU to save VRAM for TTS
            self._log("BERT emotion brain ready.", ACCENT2)
        except Exception as e:
            self._log(f"BERT load failed: {e}", DANGER)
            self.after(0, lambda: self._bert_director.set(False))
            self.after(0, lambda: self._bert_btn.config(text="BERT: FAILED", bg=DANGER))

    def _run_job_bert(self, text, style_var):
        try:
            res = self._bert_pipeline(text)[0]
            emotion = res["label"]
            preset = self._get_preset_for_emotion(emotion)
            self.after(0, lambda: style_var.set(preset))
        except:
            pass

    def _get_preset_for_emotion(self, emotion):
        """Backend-aware BERT emotion → preset mapping."""
        backend = self._backend_var.get()
        if backend == "qwen":
            mapping = {
                "sadness":  "Warm",
                "joy":      "Expressive",
                "love":     "Seductive",
                "anger":    "Aggressive",
                "fear":     "Breathy",
                "surprise": "Hyper-Real",
                "disgust":  "Aggressive",
            }
        else:  # xtts
            mapping = {
                "sadness":  "Warm",
                "joy":      "Expressive",
                "love":     "Warm",
                "anger":    "Expressive",
                "fear":     "Cinematic",
                "surprise": "Expressive",
                "disgust":  "Expressive",
            }
        result = mapping.get(emotion, "Natural")
        self._log(f"  [BERT] {emotion} → {result} ({backend.upper()} mode)", ACCENT3)
        return result

    # ── voice settings ────────────────────────────────────────────────────────
    def _build_settings(self, parent):
        cd = self.config_data
        # Maps display name -> XTTS language code
        self._lang_map = {
            "English":    "en",
            "Hindi":      "hi",
            "Spanish":    "es",
            "French":     "fr",
            "German":     "de",
            "Italian":    "it",
            "Portuguese": "pt",
            "Polish":     "pl",
            "Turkish":    "tr",
            "Russian":    "ru",
            "Dutch":      "nl",
            "Czech":      "cs",
            "Arabic":     "ar",
            "Chinese":    "zh-cn",
            "Japanese":   "ja",
            "Korean":     "ko",
        }
        self._lang_map_rev = {v: k for k, v in self._lang_map.items()}
        saved_code = cd.get("language", "en")
        saved_name = self._lang_map_rev.get(saved_code, "English")
        self.lang_var = tk.StringVar(value=saved_name)

        # ── XTTS Settings ──────────────────────────────────────────────────
        self.xtts_temp_var    = tk.DoubleVar(value=cd.get("xtts_temperature", 0.55))
        self.xtts_speed_var   = tk.DoubleVar(value=cd.get("xtts_speed",       1.00))
        self.xtts_rep_pen_var = tk.DoubleVar(value=cd.get("xtts_rep_pen",     5.0))
        self.xtts_top_k_var   = tk.IntVar(   value=cd.get("xtts_top_k",       50))
        self.xtts_top_p_var   = tk.DoubleVar(value=cd.get("xtts_top_p",       0.85))

        # ── Qwen Settings ──────────────────────────────────────────────────
        self.qwen_temp_var    = tk.DoubleVar(value=cd.get("qwen_temperature", 0.55))
        self.qwen_rep_pen_var = tk.DoubleVar(value=cd.get("qwen_rep_pen",     5.0))
        self.qwen_top_k_var   = tk.IntVar(   value=cd.get("qwen_top_k",       50))
        self.qwen_top_p_var   = tk.DoubleVar(value=cd.get("qwen_top_p",       0.85))

        # Legacy aliases so any old code path doesn't crash
        self.temp_var    = self.xtts_temp_var
        self.speed_var   = self.xtts_speed_var
        self.rep_pen_var = self.xtts_rep_pen_var
        self.top_k_var   = self.xtts_top_k_var
        self.top_p_var   = self.xtts_top_p_var

        c = self._card(parent, "VOICE SETTINGS")

        # language row (shared)
        lr = tk.Frame(c, bg=CARD_BG); lr.pack(fill="x", pady=3)
        tk.Label(lr, text="Language", width=17, anchor="w",
                 font=("Courier New", 9), bg=CARD_BG, fg=TEXT_SEC).pack(side="left")
        lang_names = list(self._lang_map.keys())
        ttk.Combobox(lr, textvariable=self.lang_var, values=lang_names, width=12,
                     state="readonly", font=("Courier New", 9)).pack(side="left")

        # ── XTTS panel ─────────────────────────────────────────────────────
        self._xtts_settings_frame = tk.Frame(c, bg=CARD_BG)
        self._lbl_xtts_temp  = self._slider_row(self._xtts_settings_frame, "Temperature",    self.xtts_temp_var,    0.10, 1.00, 0.05, "lower = less sharp")
        self._lbl_xtts_speed = self._slider_row(self._xtts_settings_frame, "Speed",          self.xtts_speed_var,   0.50, 2.00, 0.05, "0.5=slow · 1.0=normal · 2.0=fast")
        self._lbl_xtts_rep   = self._slider_row(self._xtts_settings_frame, "Repetition pen.",self.xtts_rep_pen_var, 1.00, 10.0, 0.50, "higher = fewer repeated sounds")
        self._lbl_xtts_topk  = self._slider_row(self._xtts_settings_frame, "Top-K",          self.xtts_top_k_var,   1,    100,  1,    "lower = more focused output")
        self._lbl_xtts_topp  = self._slider_row(self._xtts_settings_frame, "Top-P",          self.xtts_top_p_var,   0.10, 1.00, 0.05, "lower = more focused output")
        tk.Label(self._xtts_settings_frame,
                 text="Tip: 6–10 clean reference clips gives best XTTS results",
                 font=("Courier New", 7), bg=CARD_BG, fg=TEXT_MUT,
                 wraplength=400).pack(anchor="w", pady=(4, 0))

        # Option A toggle
        tk.Frame(self._xtts_settings_frame, bg=BORDER, height=1).pack(fill="x", pady=8)
        self._xtts_pro_cb = tk.Checkbutton(self._xtts_settings_frame, text="Pro-Audio Post-Process (Option A)",
                                          variable=self._xtts_audio_pro,
                                          bg=CARD_BG, fg=ACCENT, selectcolor=DARK_BG,
                                          activebackground=CARD_BG, activeforeground=ACCENT,
                                          font=("Courier New", 8, "bold"), bd=0, highlightthickness=0)
        self._xtts_pro_cb.pack(anchor="w")
        tk.Label(self._xtts_settings_frame,
                 text=" — applies high-pass filter, limiting & loudness normalization for clarity.",
                 font=("Courier New", 7), bg=CARD_BG, fg=TEXT_MUT).pack(anchor="w", pady=(0, 4))

        # ── Qwen panel ─────────────────────────────────────────────────────
        self._qwen_settings_frame = tk.Frame(c, bg=CARD_BG)
        self._lbl_qwen_temp  = self._slider_row(self._qwen_settings_frame, "Temperature",    self.qwen_temp_var,    0.10, 1.00, 0.05, "lower = less sharp")
        self._lbl_qwen_rep   = self._slider_row(self._qwen_settings_frame, "Repetition pen.",self.qwen_rep_pen_var, 1.00, 10.0, 0.50, "higher = fewer repeated sounds")
        self._lbl_qwen_topk  = self._slider_row(self._qwen_settings_frame, "Top-K",          self.qwen_top_k_var,   1,    100,  1,    "lower = more focused output")
        self._lbl_qwen_topp  = self._slider_row(self._qwen_settings_frame, "Top-P",          self.qwen_top_p_var,   0.10, 1.00, 0.05, "lower = more focused output")
        tk.Label(self._qwen_settings_frame,
                 text="Tip: Qwen ignores speed. Use 10-30 diverse refs for best cloning.",
                 font=("Courier New", 7), bg=CARD_BG, fg=TEXT_MUT,
                 wraplength=400).pack(anchor="w", pady=(4, 0))

        # ── Qwen feature toggles ───────────────────────────────────────
        tk.Frame(self._qwen_settings_frame, bg=BORDER, height=1).pack(fill="x", pady=8)

        etag_row = tk.Frame(self._qwen_settings_frame, bg=CARD_BG)
        etag_row.pack(fill="x", pady=(0, 4))
        tk.Checkbutton(etag_row, text="Auto Emotion Tags",
                       variable=self._qwen_emotion_tags,
                       bg=CARD_BG, fg=ACCENT2, selectcolor=DARK_BG,
                       activebackground=CARD_BG, activeforeground=ACCENT2,
                       font=("Courier New", 8, "bold"), bd=0,
                       highlightthickness=0).pack(side="left")
        tk.Label(etag_row,
                 text=" — BERT inserts [sigh]/[breath]/[haha] tags into text",
                 font=("Courier New", 7), bg=CARD_BG, fg=TEXT_MUT).pack(side="left")

        retry_row = tk.Frame(self._qwen_settings_frame, bg=CARD_BG)
        retry_row.pack(fill="x", pady=(0, 4))
        tk.Checkbutton(retry_row, text="Retry on Mumble",
                       variable=self._qwen_retry_mumble,
                       bg=CARD_BG, fg=ACCENT2, selectcolor=DARK_BG,
                       activebackground=CARD_BG, activeforeground=ACCENT2,
                       font=("Courier New", 8, "bold"), bd=0,
                       highlightthickness=0).pack(side="left")
        tk.Label(retry_row,
                 text=" — auto re-generate if output is garbled or silent",
                 font=("Courier New", 7), bg=CARD_BG, fg=TEXT_MUT).pack(side="left")

        stream_row = tk.Frame(self._qwen_settings_frame, bg=CARD_BG)
        stream_row.pack(fill="x", pady=(0, 4))
        tk.Checkbutton(stream_row, text="Streaming Preview",
                       variable=self._qwen_stream,
                       bg=CARD_BG, fg=ACCENT2, selectcolor=DARK_BG,
                       activebackground=CARD_BG, activeforeground=ACCENT2,
                       font=("Courier New", 8, "bold"), bd=0,
                       highlightthickness=0).pack(side="left")
        tk.Label(stream_row,
                 text=" — play audio chunks as Qwen generates (worker must support it)",
                 font=("Courier New", 7), bg=CARD_BG, fg=TEXT_MUT).pack(side="left")

        score_row = tk.Frame(self._qwen_settings_frame, bg=CARD_BG)
        score_row.pack(fill="x")
        self._btn(score_row, "📊 Score Qwen Refs", self._score_qwen_refs, small=True, accent=True)
        tk.Frame(score_row, width=6, bg=CARD_BG).pack(side="left")
        self._prune_btn = self._btn(score_row, "🗑 Prune Poor Refs",
                                    self._prune_poor_refs, small=True,
                                    color="#2a1218", fg_col=DANGER)
        self._prune_btn.config(state="disabled")
        self._qwen_ref_score_lbl = tk.Label(score_row, text="",
                                             font=("Courier New", 7),
                                             bg=CARD_BG, fg=TEXT_MUT)
        self._qwen_ref_score_lbl.pack(side="left", padx=6)

        # default: show xtts panel
        self._xtts_settings_frame.pack(fill="x")

    def _update_job_mood_values(self):
        """Called when backend switches: update all job mood dropdowns to show the active backend's presets."""
        if not hasattr(self, "_job_mood_combos"):
            return
        backend = self._backend_var.get()
        active_presets = XTTS_PRESETS if backend == "xtts" else QWEN_PRESETS
        new_vals = ["[Auto]"] + list(active_presets.keys())
        for cb in self._job_mood_combos:
            try:
                cb.config(values=new_vals)
                if cb.get() not in new_vals:
                    cb.set("[Auto]")
            except Exception:
                pass

    # ── RVC Post-Processing Panel (Option C) ──────────────────────────────────
    def _build_rvc_panel(self, parent):
        self._rvc_panel_frame = tk.Frame(parent, bg=DARK_BG)
        # visibility handled by _pick()
        
        c = self._card(self._rvc_panel_frame, "RVC VOICE RE-SKIN (Option C)", ACCENT3)
        row1 = tk.Frame(c, bg=CARD_BG); row1.pack(fill="x", pady=(0, 4))
        tk.Checkbutton(row1, text="Enable RVC (HD Voice Layer)", variable=self._rvc_enabled,
                       bg=CARD_BG, fg=ACCENT3, selectcolor=DARK_BG,
                       activebackground=CARD_BG, activeforeground=ACCENT3,
                       font=("Courier New", 9, "bold"), bd=0, highlightthickness=0).pack(side="left")
        
        self._btn(row1, "Load RVC Env", self._load_rvc_worker, small=True)
        self._rvc_status_lbl = tk.Label(row1, text="not loaded", font=("Courier New", 7), bg=CARD_BG, fg=TEXT_MUT)
        self._rvc_status_lbl.pack(side="left", padx=8)

        # Model selection
        row2 = tk.Frame(c, bg=CARD_BG); row2.pack(fill="x", pady=4)
        tk.Label(row2, text="Model (.pth)", width=14, anchor="w",
                 font=("Courier New", 9), bg=CARD_BG, fg=TEXT_SEC).pack(side="left")
        self._rvc_cb = ttk.Combobox(row2, textvariable=self._rvc_model_var, width=25, state="readonly", font=("Courier New", 9))
        self._rvc_cb.pack(side="left", padx=(0, 4))
        self._btn(row2, "🔄", self._refresh_rvc_models, small=True)

        # Pitch
        self._slider_row(c, "Pitch Transpose", self._rvc_pitch_var, -12, 12, 1, "0=same · -12=oct down · +12=oct up")
        
        # Index Rate
        self._slider_row(c, "Index Strength", self._rvc_index_var, 0.0, 1.0, 0.05, "Higher = more character texture")

        # F0 Method
        row3 = tk.Frame(c, bg=CARD_BG); row3.pack(fill="x", pady=4)
        tk.Label(row3, text="F0 Method", width=14, anchor="w",
                 font=("Courier New", 9), bg=CARD_BG, fg=TEXT_SEC).pack(side="left")
        ttk.Combobox(row3, textvariable=self._rvc_method_var, values=["rmvpe", "pm", "harvest"], width=10, state="readonly", font=("Courier New", 9)).pack(side="left")

        # ── RVC Presets ──
        tk.Frame(c, bg=BORDER, height=1).pack(fill="x", pady=(8, 4))
        tk.Label(c, text="PRESETS", font=("Courier New", 8, "bold"),
                 bg=CARD_BG, fg=TEXT_SEC).pack(anchor="w")
        preset_grid = tk.Frame(c, bg=CARD_BG)
        preset_grid.pack(fill="x", pady=(4, 0))
        self._rvc_preset_btns = {}
        cols = 3
        for i, (name, vals) in enumerate(RVC_PRESETS.items()):
            desc = vals[3]
            btn = tk.Button(preset_grid, text=name, font=("Courier New", 8),
                            bg=PANEL_BG, fg=TEXT_SEC,
                            activebackground=ACCENT3, activeforeground=DARK_BG,
                            relief="flat", bd=0, padx=6, pady=4, cursor="hand2",
                            command=lambda n=name: self._apply_rvc_preset(n))
            btn.grid(row=i//cols, column=i%cols, padx=3, pady=2, sticky="ew")
            btn.bind("<Enter>", lambda e, d=desc, b=btn: b.config(fg=ACCENT3))
            btn.bind("<Leave>", lambda e, b=btn, n=name: b.config(
                fg=DARK_BG if b.cget("bg")==ACCENT3 else TEXT_SEC))
            self._rvc_preset_btns[name] = btn
        for col in range(cols):
            preset_grid.columnconfigure(col, weight=1)

        # ── Auto-RVC Director ──
        tk.Frame(c, bg=BORDER, height=1).pack(fill="x", pady=(8, 4))
        auto_row = tk.Frame(c, bg=CARD_BG); auto_row.pack(fill="x")
        self._rvc_auto_cb = tk.Checkbutton(
            auto_row, text="Auto-RVC Director",
            variable=self._rvc_auto_var,
            bg=CARD_BG, fg=ACCENT3, selectcolor=DARK_BG,
            activebackground=CARD_BG, activeforeground=ACCENT3,
            font=("Courier New", 8, "bold"), bd=0, highlightthickness=0)
        self._rvc_auto_cb.pack(side="left")
        tk.Label(auto_row, text=" — BERT emotion + intensity → auto preset",
                 font=("Courier New", 7), bg=CARD_BG, fg=TEXT_MUT).pack(side="left")

        scope_row = tk.Frame(c, bg=CARD_BG); scope_row.pack(fill="x", pady=(4, 0))
        tk.Label(scope_row, text="Apply:", font=("Courier New", 8),
                 bg=CARD_BG, fg=TEXT_SEC).pack(side="left")
        for val, lbl in [("per-job", "Per-job (each line)"), ("global", "Global (all lines)")]:
            tk.Radiobutton(scope_row, text=lbl, variable=self._rvc_auto_scope, value=val,
                           bg=CARD_BG, fg=TEXT_SEC, selectcolor=DARK_BG,
                           activebackground=CARD_BG, activeforeground=ACCENT3,
                           font=("Courier New", 8), bd=0, highlightthickness=0
                           ).pack(side="left", padx=(8, 0))

        self._rvc_auto_lbl = tk.Label(c, text="",
                                       font=("Courier New", 7), bg=CARD_BG, fg=ACCENT3,
                                       wraplength=360, justify="left")
        self._rvc_auto_lbl.pack(anchor="w", pady=(2, 0))

        self._refresh_rvc_models()

    def _apply_rvc_preset(self, name):
        if name not in RVC_PRESETS: return
        pitch, index_rate, f0_method, desc = RVC_PRESETS[name]
        self._rvc_pitch_var.set(pitch)
        self._rvc_index_var.set(index_rate)
        self._rvc_method_var.set(f0_method)
        # Highlight active preset button
        for n, btn in self._rvc_preset_btns.items():
            btn.config(bg=ACCENT3 if n == name else PANEL_BG,
                       fg=DARK_BG if n == name else TEXT_SEC)
        self._log(f"RVC Preset: {name} — {desc}", ACCENT3)

    def _refresh_rvc_models(self):
        models = glob.glob("rvc_models/*.pth")
        names = [os.path.basename(m) for m in models]
        self._rvc_cb.config(values=names)
        if names and not self._rvc_model_var.get():
            self._rvc_model_var.set(names[0])

    def _auto_rvc_preset_for(self, text: str) -> tuple:
        """
        Return (pitch, index_rate, f0_method) for the given text using:
          A) BERT emotion   → base preset
          B) Intensity score (caps, punctuation, word count) → index_rate tweak

        Falls back to "Natural" preset if BERT is not loaded.
        """
        # ── A: BERT emotion → base preset ──────────────────────────────
        emotion = "neutral"
        if self._bert_pipeline:
            try:
                res = self._bert_pipeline(text[:512])[0]
                emotion = res["label"].lower()
            except Exception:
                pass

        emotion_to_rvc = {
            "joy":      "Pitch Up",
            "love":     "Feminine",
            "sadness":  "Subtle",
            "fear":     "Subtle",
            "anger":    "Character+",
            "disgust":  "Masculine",
            "surprise": "Pitch Up",
        }
        preset_name = emotion_to_rvc.get(emotion, "Natural")
        pitch, index_rate, f0_method, _ = RVC_PRESETS[preset_name]

        # ── B: Intensity score → index_rate fine-tune ──────────────────
        # Metrics: ALL_CAPS ratio, exclamation marks, question marks, length
        words      = text.split()
        caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 1) / max(len(words), 1)
        exclaims   = text.count("!") + text.count("!!")
        questions  = text.count("?")
        ellipsis   = text.count("…") + text.count("...")
        intensity  = min(1.0, caps_ratio * 2 + exclaims * 0.15 + questions * 0.05 + ellipsis * 0.05)

        # Blend: high intensity → push index_rate toward character texture
        # low intensity (soft/sad) → pull toward subtle
        if emotion in ("anger", "disgust", "surprise"):
            index_rate = min(0.85, index_rate + intensity * 0.3)
        elif emotion in ("sadness", "fear"):
            index_rate = max(0.10, index_rate - intensity * 0.15)
        else:
            index_rate = min(0.75, index_rate + intensity * 0.15)

        index_rate = round(index_rate, 2)
        return pitch, index_rate, f0_method, preset_name, emotion, intensity

    def _apply_global_auto_rvc(self):
        """Analyse all job texts together, pick one RVC preset for all."""
        all_texts = [var.get().strip() for var, *_ in self._job_entries if var.get().strip()]
        if not all_texts:
            return
        combined = " ".join(all_texts)
        pitch, index_rate, f0_method, preset_name, emotion, intensity =             self._auto_rvc_preset_for(combined)
        self._rvc_pitch_var.set(pitch)
        self._rvc_index_var.set(index_rate)
        self._rvc_method_var.set(f0_method)
        msg = (f"Auto-RVC (global): {preset_name} "
               f"[emotion={emotion}, intensity={intensity:.2f}] "
               f"→ pitch={pitch:+d}, index={index_rate:.2f}, method={f0_method}")
        self._log(msg, ACCENT3)
        self.after(0, lambda: self._rvc_auto_lbl.config(
            text=f"Global: {preset_name} · {emotion} · intensity {intensity:.0%}"))
        # Highlight preset button
        for n, btn in self._rvc_preset_btns.items():
            btn.config(bg=ACCENT3 if n == preset_name else PANEL_BG,
                       fg=DARK_BG if n == preset_name else TEXT_SEC)

    def _load_rvc_worker(self):
        if self._rvc_worker.is_alive():
            messagebox.showinfo("RVC", "RVC worker already running."); return
        if not os.path.isfile(_RVC_PYTHON):
            messagebox.showerror("No RVC env", "Run setup_rvc_env.bat first.")
            return
        
        self._log("Starting RVC worker (rvc-env)…", WARNING)
        self._set_status("loading RVC…", WARNING)
        
        def on_ready():
            self._log("RVC worker ready!", ACCENT3)
            self._set_status("RVC ready", ACCENT3)
            self.after(0, lambda: self._rvc_status_lbl.config(text="online", fg=ACCENT3))
            # Ask worker for model info if a model is already selected
            m = self._rvc_model_var.get()
            if m:
                self._rvc_worker.send({"action": "model_info",
                                       "model": os.path.join("rvc_models", m)})
        
        def on_error(msg):
            self._log(f"RVC start failed: {msg}", DANGER)
            self._set_status("RVC error", DANGER)
        
        self._rvc_worker.start(on_ready, on_error)

    # ── voice profile ─────────────────────────────────────────────────────────
    def _build_voice_profile(self, parent):
        self._profile_card_frame = tk.Frame(parent, bg=DARK_BG)
        self._profile_card_frame.pack(fill="x", pady=(0, 10))

        outer = tk.Frame(self._profile_card_frame, bg=CARD_BG, bd=0,
                         highlightbackground=BORDER, highlightthickness=1)
        outer.pack(fill="x")
        tk.Label(outer, text="VOICE PROFILE  —  save once, clone forever",
                 font=("Courier New", 9, "bold"), bg=CARD_BG, fg=TEXT_SEC
                 ).pack(anchor="w", padx=12, pady=(10, 4))
        c = tk.Frame(outer, bg=CARD_BG); c.pack(fill="x", padx=12, pady=(0, 10))

        self._profile_var = tk.StringVar(value=self.config_data.get("profile_path", ""))
        pr = tk.Frame(c, bg=CARD_BG); pr.pack(fill="x", pady=(0, 6))
        tk.Label(pr, text="Profile (.pth)", width=14, anchor="w",
                 font=("Courier New", 9), bg=CARD_BG, fg=TEXT_SEC).pack(side="left")
        tk.Entry(pr, textvariable=self._profile_var, bg=PANEL_BG, fg=TEXT_PRI,
                 insertbackground=TEXT_PRI, relief="flat", bd=4,
                 font=("Courier New", 9)).pack(side="left", fill="x",
                                               expand=True, padx=(0, 4))
        self._btn(pr, "Browse", self._browse_profile, small=True)

        self._profile_status = tk.Label(
            c, text="No profile loaded  —  WAV files will be used each run",
            font=("Courier New", 7), bg=CARD_BG, fg=TEXT_MUT)
        self._profile_status.pack(anchor="w", pady=(0, 6))

        btns = tk.Frame(c, bg=CARD_BG); btns.pack(fill="x")
        self._save_profile_btn = self._btn(btns, "Save Profile",
                                            self._save_voice_profile, small=True)
        tk.Frame(btns, width=8, bg=CARD_BG).pack(side="left")
        self._load_profile_btn = self._btn(btns, "Load Profile",
                                            self._load_voice_profile,
                                            accent=True, small=True)
        tk.Frame(btns, width=8, bg=CARD_BG).pack(side="left")
        self._btn(btns, "Clear", self._clear_voice_profile, small=True)
        tk.Label(c,
                 text="Save extracts your voice embedding — load instantly next "
                      "time without any WAV files.",
                 font=("Courier New", 7), bg=CARD_BG, fg=TEXT_MUT,
                 wraplength=400, justify="left").pack(anchor="w", pady=(6, 0))

    def _browse_profile(self):
        p = filedialog.askopenfilename(title="Select voice profile",
            filetypes=[("Voice profile","*.pth"),("All","*.*")])
        if p: self._profile_var.set(p)

    def _find_xtts_paths(self):
        from TTS.utils.manage import ModelManager
        manager = ModelManager()
        model_path, config_path, _ = manager.download_model(
            "tts_models/multilingual/multi-dataset/xtts_v2")
        if not config_path or not os.path.isfile(config_path):
            candidates = []
            if model_path and os.path.isdir(model_path):
                for root, _, files in os.walk(model_path):
                    if "config.json" in files:
                        candidates.append(os.path.join(root, "config.json"))
            win_cache = os.path.join(os.path.expanduser("~"), "AppData", "Local",
                        "tts", "tts_models--multilingual--multi-dataset--xtts_v2")
            for base in [win_cache, os.path.expanduser("~/.local/share/tts")]:
                for root, _, files in os.walk(base):
                    if "config.json" in files:
                        candidates.append(os.path.join(root, "config.json"))
            for cp in candidates:
                if os.path.isfile(cp):
                    config_path = cp; break
        if not config_path or not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"config.json not found (model_path={model_path}). "
                "Delete cached model and re-download.")
        return os.path.dirname(config_path), config_path

    def _save_voice_profile(self):
        if not self._tts_model:
            messagebox.showerror("No model", "Load XTTS v2 first."); return
        valid_wavs, _ = self._get_ref_wavs()
        if not valid_wavs:
            messagebox.showerror("No WAVs", "No valid WAV references found."); return
        out = filedialog.asksaveasfilename(title="Save voice profile as",
            defaultextension=".pth", filetypes=[("Voice profile","*.pth")])
        if not out: return
        self._save_profile_btn.config(state="disabled")
        threading.Thread(target=self._run_save_profile,
                         args=(valid_wavs, out), daemon=True).start()

    def _run_save_profile(self, wav_files, out_path):
        try:
            self._log("Extracting voice embedding…", WARNING)
            self._set_status("extracting embedding…", WARNING)
            
            cmd = {
                "action": "save_profile",
                "refs": wav_files,
                "out_path": out_path
            }
            self._xtts_worker.send(cmd)
            resp = self._xtts_worker.get_response(timeout=180)
            
            if resp.get("status") == "done_save":
                self._log(f"Profile saved → {out_path}", ACCENT2)
                self._profile_var.set(out_path)
                self.config_data["profile_path"] = out_path
                self._profile_path  = out_path
                name = os.path.basename(out_path)
                self.after(0, lambda: self._profile_status.config(
                    text=f"Profile active: {name}", fg=ACCENT2))
                self._set_status("profile saved & loaded", ACCENT2)
            else:
                err = resp.get("message", "unknown error")
                raise Exception(err)
        except Exception as e:
            self._log(f"Save profile failed: {e}", DANGER)
            self._set_status("profile error", DANGER)
        finally:
            self.after(0, lambda: self._save_profile_btn.config(state="normal"))

    def _load_voice_profile(self):
        path = self._profile_var.get().strip()
        if not path:
            path = filedialog.askopenfilename(title="Select voice profile",
                filetypes=[("Voice profile","*.pth"),("All","*.*")])
            if not path: return
            self._profile_var.set(path)
        if not os.path.exists(path):
            messagebox.showerror("Not found", f"File not found:\n{path}"); return
        self._load_profile_btn.config(state="disabled")
        threading.Thread(target=self._run_load_profile, args=(path,), daemon=True).start()

    def _run_load_profile(self, path):
        try:
            self._log(f"Loading profile: {os.path.basename(path)}…", WARNING)
            # Worker handles the actual torch loading when generating
            self._profile_path = path
            self.config_data["profile_path"] = path
            self._log("Profile loaded.", ACCENT2)
            name = os.path.basename(path)
            self.after(0, lambda: self._profile_status.config(
                text=f"Profile active: {name}  (WAV files will NOT be used)", fg=ACCENT2))
            self._set_status("profile ready", ACCENT2)
        except Exception as e:
            self._log(f"Load profile failed: {e}", DANGER)
            self._profile_path = None
        finally:
            self.after(0, lambda: self._load_profile_btn.config(state="normal"))

    def _clear_voice_profile(self):
        self._voice_profile = None; self._profile_path = None
        self._profile_var.set(""); self.config_data.pop("profile_path", None)
        self._profile_status.config(
            text="No profile loaded  —  WAV files will be used each run", fg=TEXT_MUT)
        self._log("Voice profile cleared.", TEXT_MUT)

    def _build_qwen_profile(self, parent):
        self._qwen_profile_card_frame = tk.Frame(parent, bg=DARK_BG)
        # Visible only when Qwen backend is selected
        self._qwen_profile_card_frame.pack(fill="x", pady=(0, 10))

        c = self._card(self._qwen_profile_card_frame, "QWEN VOICE PROFILE (Beta)", ACCENT)
        
        pr = tk.Frame(c, bg=CARD_BG); pr.pack(fill="x", pady=(0, 6))
        tk.Label(pr, text="Profile (.qproc)", width=16, anchor="w",
                 font=("Courier New", 9), bg=CARD_BG, fg=TEXT_SEC).pack(side="left")
        tk.Entry(pr, textvariable=self._qwen_profile_var, bg=PANEL_BG, fg=TEXT_PRI,
                 insertbackground=TEXT_PRI, relief="flat", bd=4,
                 font=("Courier New", 9)).pack(side="left", fill="x",
                                               expand=True, padx=(0, 4))
        self._btn(pr, "Browse", self._browse_qwen_profile, small=True)

        self._qwen_profile_status = tk.Label(
            c, text="No profile active: 2-5s per-file scanning active.",
            font=("Courier New", 7), bg=CARD_BG, fg=TEXT_MUT)
        self._qwen_profile_status.pack(anchor="w", pady=(0, 6))

        btns = tk.Frame(c, bg=CARD_BG); btns.pack(fill="x")
        self._save_qwen_btn = self._btn(btns, "Generate Profile",
                                        self._save_qwen_profile, small=True)
        tk.Frame(btns, width=8, bg=CARD_BG).pack(side="left")
        self._load_qwen_btn = self._btn(btns, "Load Profile",
                                        self._load_qwen_profile,
                                        accent=True, small=True)
        tk.Frame(btns, width=8, bg=CARD_BG).pack(side="left")
        self._btn(btns, "Clear", self._clear_qwen_profile, small=True)

        tk.Label(c,
                 text="Profiles merge all reference clips into one 'Super Latent'. "
                      "Loading a profile skips audio analysis and starts generation instantly.",
                 font=("Courier New", 7), bg=CARD_BG, fg=TEXT_MUT,
                 wraplength=400, justify="left").pack(anchor="w", pady=(6, 0))

    def _browse_qwen_profile(self):
        p = filedialog.askopenfilename(title="Select Qwen profile",
            filetypes=[("Qwen profile","*.qproc"),("All","*.*")])
        if p: self._qwen_profile_var.set(p)

    def _save_qwen_profile(self):
        valid_wavs, _ = self._get_ref_wavs()
        if not valid_wavs:
            messagebox.showerror("No WAVs", "No Qwen reference WAVs found."); return
            
        out = filedialog.asksaveasfilename(title="Save Qwen profile as",
            defaultextension=".qproc", filetypes=[("Qwen profile","*.qproc")])
        if not out: return
        
        self._save_qwen_btn.config(state="disabled")
        threading.Thread(target=self._run_save_qwen_profile,
                         args=(valid_wavs, out), daemon=True).start()

    def _run_save_qwen_profile(self, wav_files, out_path):
        try:
            self._log("Qwen: Analyzing references to create super-latent profile...", WARNING)
            self._set_status("generating qwen profile...", WARNING)
            
            cmd = {
                "action": "create_profile",
                "refs": wav_files,
                "out": out_path
            }
            self._qwen_worker.send(cmd)
            resp = self._qwen_worker.get_response(timeout=300)
            
            if resp.get("status") == "done":
                self._log(f"Qwen Profile saved → {os.path.basename(out_path)}", ACCENT2)
                self._qwen_profile_pth = out_path
                self._qwen_profile_var.set(out_path)
                name = os.path.basename(out_path)
                self.after(0, lambda: self._qwen_profile_status.config(
                    text=f"Profile active: {name} (Scanning bypassed)", fg=ACCENT))
                self._set_status("qwen profile ready", ACCENT2)
            else:
                err = resp.get("message", "unknown error")
                raise Exception(err)
        except Exception as e:
            self._log(f"Save Qwen profile failed: {e}", DANGER)
        finally:
            self.after(0, lambda: self._save_qwen_btn.config(state="normal"))

    def _load_qwen_profile(self):
        path = self._qwen_profile_var.get().strip()
        if not path:
            path = filedialog.askopenfilename(title="Select Qwen profile",
                filetypes=[("Qwen profile","*.qproc"),("All","*.*")])
            if not path: return
            self._qwen_profile_var.set(path)
        
        if not os.path.exists(path):
            messagebox.showerror("Not found", f"File not found:\n{path}"); return
            
        self._qwen_profile_pth = path
        name = os.path.basename(path)
        self._qwen_profile_status.config(
            text=f"Profile active: {name} (Scanning bypassed)", fg=ACCENT)
        self._log(f"Qwen Profile loaded: {name}", ACCENT2)

    def _clear_qwen_profile(self):
        self._qwen_profile_pth = None
        self._qwen_profile_var.set("")
        self._qwen_profile_status.config(
            text="No profile active: 2-5s per-file scanning active.", fg=TEXT_MUT)
        self._set_status("qwen profile cleared", TEXT_MUT)

    # ── Skyrim SE Export Panel ────────────────────────────────────────────────
    def _build_skyrim_panel(self, parent):
        c = self._card(parent, "SKYRIM SE EXPORT (FUZ)", WARNING)
        
        row1 = tk.Frame(c, bg=CARD_BG); row1.pack(fill="x", pady=(0, 4))
        tk.Checkbutton(row1, text="Enable Skyrim FUZ Export", variable=self._skyrim_enabled,
                       bg=CARD_BG, fg=WARNING, selectcolor=DARK_BG,
                       activebackground=CARD_BG, activeforeground=WARNING,
                       font=("Courier New", 9, "bold"), bd=0, highlightthickness=0).pack(side="left")

        # Plugin & VoiceType
        row2 = tk.Frame(c, bg=CARD_BG); row2.pack(fill="x", pady=2)
        tk.Label(row2, text="Plugin", width=10, anchor="w", font=("Courier New", 8), bg=CARD_BG, fg=TEXT_SEC).pack(side="left")
        tk.Entry(row2, textvariable=self._skyrim_plugin, bg=PANEL_BG, fg=TEXT_PRI, font=("Courier New", 8), relief="flat", bd=2).pack(side="left", fill="x", expand=True, padx=2)
        
        row3 = tk.Frame(c, bg=CARD_BG); row3.pack(fill="x", pady=2)
        tk.Label(row3, text="VoiceType", width=10, anchor="w", font=("Courier New", 8), bg=CARD_BG, fg=TEXT_SEC).pack(side="left")
        tk.Entry(row3, textvariable=self._skyrim_voice_type, bg=PANEL_BG, fg=TEXT_PRI, font=("Courier New", 8), relief="flat", bd=2).pack(side="left", fill="x", expand=True, padx=2)

        # Tools paths
        tk.Frame(c, bg=BORDER, height=1).pack(fill="x", pady=6)
        tk.Label(c, text="BETHESDA TOOLS PATHS", font=("Courier New", 7, "bold"), bg=CARD_BG, fg=TEXT_MUT).pack(anchor="w")
        
        for lbl, var in [("FaceFX", self._skyrim_facefx_path), 
                         ("xWMA",  self._skyrim_xwma_path), 
                         ("Fonix", self._skyrim_fonix_path)]:
            r = tk.Frame(c, bg=CARD_BG); r.pack(fill="x", pady=1)
            tk.Label(r, text=lbl, width=8, anchor="w", font=("Courier New", 7), bg=CARD_BG, fg=TEXT_MUT).pack(side="left")
            tk.Entry(r, textvariable=var, bg=DARK_BG, fg=TEXT_MUT, font=("Courier New", 7), relief="flat").pack(side="left", fill="x", expand=True, padx=2)
            self._btn(r, "...", lambda v=var: self._browse_tool(v), small=True).pack(side="right")

    def _browse_tool(self, var):
        p = filedialog.askopenfilename(title="Select Tool File")
        if p: var.set(p)

    # ══ MIDDLE PANEL ════════════════════════════════════════════════════════

    # ── speech jobs ───────────────────────────────────────────────────────────
    def _build_jobs(self, parent):
        c = self._card(parent, "SPEECH JOBS")

        # batch import row
        imp_row = tk.Frame(c, bg=CARD_BG); imp_row.pack(fill="x", pady=(0, 6))
        self._btn(imp_row, "Import .txt", self._import_txt, small=True)
        tk.Frame(imp_row, width=4, bg=CARD_BG).pack(side="left")
        self._btn(imp_row, "Import JSON", self._import_json, small=True)
        tk.Frame(imp_row, width=4, bg=CARD_BG).pack(side="left")
        self._btn(imp_row, "Export JSON", self._export_json, small=True)
        tk.Frame(imp_row, width=8, bg=CARD_BG).pack(side="left")
        tk.Label(imp_row, text=".txt (1 line/job) or .json ([{Dialogue:..., Name:...}])",
                 font=("Courier New", 7), bg=CARD_BG, fg=TEXT_MUT).pack(side="left")

        tk.Label(c, text="Tip: For attack grunts, use ALL CAPS and multiple exclamation marks (e.g., 'RRRGH!!')",
                 font=("Courier New", 7), bg=CARD_BG, fg=TEXT_MUT, wraplength=400, justify="left").pack(anchor="w", pady=(0, 6))

        tk.Frame(c, bg=BORDER, height=1).pack(fill="x", pady=6)

        # ── Batch settings ──
        batch_row = tk.Frame(c, bg=CARD_BG); batch_row.pack(fill="x")
        
        # Naming Mode
        tk.Label(batch_row, text="Naming:", font=("Courier New", 8, "bold"),
                 bg=CARD_BG, fg=TEXT_SEC).pack(side="left", padx=(0, 4))

        for mode in ["Normal", "Sequential"]:
            tk.Radiobutton(batch_row, text=mode, variable=self._naming_mode, value=mode,
                           bg=CARD_BG, fg=TEXT_PRI, selectcolor=DARK_BG,
                           activebackground=CARD_BG, activeforeground=ACCENT,
                           font=("Courier New", 8), bd=0, highlightthickness=0).pack(side="left", padx=4)

        tk.Frame(batch_row, width=20, bg=CARD_BG).pack(side="left")

        # Custom Folder toggle
        def _toggle_batch():
            self._refresh_output_list()
        
        tk.Checkbutton(batch_row, text="", variable=self._batch_out_enabled,
                       command=_toggle_batch, bg=CARD_BG, selectcolor=DARK_BG,
                       activebackground=CARD_BG, bd=0, highlightthickness=0,
                       padx=0).pack(side="left")
        
        tk.Label(batch_row, text="Custom Batch Folder:", font=("Courier New", 8),
                 bg=CARD_BG, fg=TEXT_SEC).pack(side="left")
        
        def _browse_batch():
            initial = self._batch_out_folder.get() or self.out_folder_var.get()
            p = filedialog.askdirectory(title="Select Batch Output Folder", initialdir=initial)
            if p: 
                self._batch_out_folder.set(p)
                self._batch_out_enabled.set(True)
                self._refresh_output_list()

        # Pack button to the RIGHT of the frame so it's always visible
        self._btn(batch_row, "Browse", _browse_batch, small=True).pack(side="right", padx=(4, 0))

        batch_folder_ent = tk.Entry(batch_row, textvariable=self._batch_out_folder,
                                    bg=PANEL_BG, fg=TEXT_PRI, insertbackground=TEXT_PRI,
                                    relief="flat", bd=2, font=("Courier New", 8), width=15)
        batch_folder_ent.pack(side="left", fill="x", expand=True, padx=4)

        tk.Label(batch_row, text="Batch Size:", font=("Courier New", 8),
                 bg=CARD_BG, fg=TEXT_SEC).pack(side="left", padx=(10, 4))
        tk.Entry(batch_row, textvariable=self._batch_size_var, width=3,
                 bg=DARK_BG, fg=TEXT_PRI, font=("Courier New", 9), relief="flat").pack(side="left")

        tk.Frame(c, bg=BORDER, height=1).pack(fill="x", pady=6)

        # job list frame (flows with main middle column scroll)
        self._jobs_frame = tk.Frame(c, bg=CARD_BG)
        self._jobs_frame.pack(fill="x")

        self._job_entries = []
        self._job_output_files = {}  # idx -> output filepath
        self._last_ref_scores  = {}  # path -> score (from last scoring run)
        for _ in range(3):
            self._add_job_row()

        btns = tk.Frame(c, bg=CARD_BG)
        btns.pack(fill="x", pady=(12, 0))
        self._btn(btns, "+ Add",  self._add_job_row, small=True)
        tk.Frame(btns, width=6, bg=CARD_BG).pack(side="left")
        self._btn(btns, "Clear all", self._clear_jobs, small=True)
        tk.Frame(btns, width=6, bg=CARD_BG).pack(side="left")
        self._gen_btn = self._btn(btns, "Generate All",
                                   self._generate_all, accent=True)
        tk.Frame(btns, width=6, bg=CARD_BG).pack(side="left")
        self._stop_btn = self._btn(btns, "■ Global Stop", self._stop_gen,
                                   small=True, accent=True, color=DANGER)
        self._stop_btn.config(state="disabled")

        tk.Frame(btns, width=6, bg=CARD_BG).pack(side="left")
        self._skip_batch_btn = self._btn(btns, "⏭ Skip Batch", 
                                          lambda: self._skip_batch_event.set(), small=True)
        self._skip_batch_btn.config(state="disabled")
        
        tk.Frame(btns, width=6, bg=CARD_BG).pack(side="left")
        self._stop_batch_btn = self._btn(btns, "⏹ Stop Batch", 
                                          lambda: self._stop_batch_event.set(), small=True)
        self._stop_batch_btn.config(state="disabled")

        self._gen_progress = ttk.Progressbar(c, mode="determinate",
                                              style="TProgressbar")
        self._gen_progress.pack(fill="x", pady=(10, 4))

    def _add_job_row(self, text="", filename=""):
        idx = len(self._job_entries) + 1
        row = tk.Frame(self._jobs_frame, bg=CARD_BG); row.pack(fill="x", pady=2)
        tk.Label(row, text=f"#{idx:02d}", width=4,
                 font=("Courier New", 8), bg=CARD_BG, fg=TEXT_MUT).pack(side="left")
                 
        status = tk.Label(row, text="○", font=("Courier New", 9),
                           bg=CARD_BG, fg=TEXT_MUT, width=2)
        status.pack(side="left")

        # Skip/Stop controls - pinned to the left near status for visibility
        skip_btn = self._btn(row, "⏭", lambda: self._skip_job_event.set(), small=True)
        stop_btn = self._btn(row, "⏹", lambda: self._stop_batch_event.set(), small=True)

        var = tk.StringVar(value=text)
        expr = tk.Entry(row, textvariable=var, bg=PANEL_BG, fg=TEXT_PRI,
                      insertbackground=TEXT_PRI, relief="flat", bd=4,
                      font=("Courier New", 9))
        expr.pack(side="left", fill="x", expand=True, padx=(0, 6))
        
        # Optional custom filename
        name_var = tk.StringVar(value=filename)
        name_entry = tk.Entry(row, textvariable=name_var, bg=DARK_BG, fg=TEXT_SEC,
                              insertbackground=TEXT_PRI, relief="flat", bd=4,
                              font=("Courier New", 8), width=15)
        name_entry.pack(side="left", padx=(0, 6))

        # Individual Mood selector
        job_mood_var = tk.StringVar(value="[Auto]")
        backend = self._backend_var.get()
        active_presets = XTTS_PRESETS if backend == "xtts" else QWEN_PRESETS
        mood_vals = ["[Auto]"] + list(active_presets.keys())
        style_cb = ttk.Combobox(row, textvariable=job_mood_var, values=mood_vals,
                                 width=15, state="readonly", font=("Courier New", 8))
        style_cb.pack(side="left", padx=(0, 6))
        
        if not hasattr(self, "_job_mood_combos"):
            self._job_mood_combos = []
        self._job_mood_combos.append(style_cb)

        def _on_text_change(*args):
            if self._bert_director.get() and self._bert_pipeline:
                txt = var.get().strip()
                if len(txt) > 8: # min length to avoid noise
                    threading.Thread(target=self._run_job_bert, args=(txt, job_mood_var), daemon=True).start()
        var.trace_add("write", _on_text_change)
        if text: _on_text_change() # Trigger for initial import

        play_btn = self._btn(row, "▶", lambda i=idx: self._play_job_output(i),
                             small=True)
        play_btn.config(state="disabled")

        # entry must be defined before _remove so the closure captures it
        entry = (var, status, job_mood_var, name_var, play_btn, skip_btn, stop_btn)

        def _remove(r=row, e=entry):
            r.destroy()
            if e in self._job_entries: self._job_entries.remove(e)
        self._btn(row, "✕", _remove, small=True)
        self._job_entries.append(entry)

    def _clear_jobs(self):
        for entry in list(self._job_entries):
            # Destroy the master frame of the status label (the row itself)
            entry[1].master.destroy()
        self._job_entries.clear()
        if hasattr(self, "_job_batch_headers"):
            for h in self._job_batch_headers:
                try: h.destroy()
                except: pass
            self._job_batch_headers = []
        self._job_output_files = {}

    def _import_txt(self):
        path = filedialog.askopenfilename(
            title="Import text file",
            filetypes=[("Text files","*.txt"),("All","*.*")])
        if not path: return
        try:
            with open(path, encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            if not lines:
                messagebox.showwarning("Empty", "No lines found in file."); return
            self._clear_jobs()
            for line in lines:
                self._add_job_row(text=line)
            self._log(f"Imported {len(lines)} job(s) from {os.path.basename(path)}",
                      ACCENT2)
        except Exception as e:
            messagebox.showerror("Import failed", str(e))

    def _import_json(self):
        path = filedialog.askopenfilename(
            title="Import JSON file",
            filetypes=[("JSON files","*.json"),("All","*.*")])
        if not path: return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            
            # support both {"Dialogue": "...", "Name": "..."} and list of these
            if isinstance(data, dict):
                data = [data]
            if not isinstance(data, list):
                messagebox.showwarning("Invalid Format", "JSON must be a list of objects.")
                return
            
            jobs_added = 0
            self._clear_jobs()
            for item in data:
                if isinstance(item, dict) and "Dialogue" in item:
                    text = item["Dialogue"]
                    name = item.get("Name", "")
                    self._add_job_row(text=text, filename=name)
                    jobs_added += 1
            if jobs_added == 0:
                messagebox.showwarning("Empty", "No valid jobs found in JSON. Expected format: [{\"Dialogue\": \"...\", \"Name\": \"...\"}]")
                return
            self._log(f"Imported {jobs_added} job(s) from {os.path.basename(path)}", ACCENT2)
        except Exception as e:
            messagebox.showerror("Import failed", str(e))

    # ── Qwen Feature 1: Emotion Tag Insertion ───────────────────────────────
    def _apply_qwen_emotion_tags(self, text: str) -> str:
        """
        Use BERT emotion + QWEN_EMOTION_TAGS to modify text for the Base model.
        Uses h- prefix and punctuation cues rather than bracket tokens,
        since [breath]/[sigh] are only supported by the 8B Instruct model
        and are spoken literally by the 0.6B Base model.

        Guard rules applied before BERT runs:
        1) If text already starts with a vocal filler (hmmm, mmm, uhh, ahh, ohh,
           sigh, etc.) we skip emotion tagging entirely — adding h- on top would
           create a redundant double-expression artifact.
        2) We strip leading fillers before sending to BERT so the model sees the
           meaningful sentence content, not the ambiguous opener that confuses it.

        Note: BERT (distilbert-base-uncased-emotion) understands keywords/word
        co-occurrence, NOT narrative context or non-English text.  Treat its
        output as a useful hint, not a deep semantic understanding.
        """
        if not self._bert_pipeline:
            return text

        # Vocal filler guard — patterns that represent natural vocal expressions.
        # If the text opens with one of these, it already has its own prosodic
        # cue; adding h- on top creates a double-expression that plays back oddly.
        import re
        FILLER_RE = re.compile(
            r'^(h+m+[hm]*|m+[hm]+|u+h+|a+h+|o+h+|e+r+|s+igh+|hmph|hah|heh|huh|ugh)'
            r'[\s,\.!?\u2026]*',
            re.IGNORECASE
        )
        filler_match = FILLER_RE.match(text.strip())
        if filler_match:
            self._log(
                f"  [Qwen tags] skipped — filler opener '{filler_match.group().strip()}'",
                TEXT_MUT
            )
            return text

        # Strip filler openers so BERT analyses the actual semantic content
        bert_input = FILLER_RE.sub("", text.strip()).strip() or text
        try:
            res = self._bert_pipeline(bert_input[:512])[0]
            emotion = res["label"].lower()
        except Exception:
            return text

        prefix, suffix, desc = QWEN_EMOTION_TAGS.get(emotion, ("", "", "no modification"))
        if not prefix and not suffix:
            return text

        # Strip trailing punctuation before adding our suffix
        stripped = text.rstrip(" .,!?\u2026")
        tagged = f"{prefix}{stripped}{suffix}"

        self._log(f"  [Qwen tags] {emotion} -> {desc}: {tagged[:60]}", TEXT_MUT)
        return tagged

    # ── Qwen Feature 2: Ref Clip Scoring ────────────────────────────────────
    def _score_qwen_refs(self):
        """Score Qwen reference clips by SNR, duration, and spectral clarity."""
        folder = self.qwen_ref_folder_var.get()
        wavs = glob.glob(os.path.join(folder, "*.wav"))
        if not wavs:
            messagebox.showwarning("No refs", f"No WAV files in '{folder}'."); return
        threading.Thread(target=self._run_score_refs, args=(wavs,), daemon=True).start()

    def _run_score_refs(self, wavs):
        self._set_status("scoring refs…", WARNING)
        self._log(f"Scoring {len(wavs)} Qwen reference clips…", WARNING)
        try:
            import numpy as np
            AudioSegment = _import_pydub()
            scores = []
            for wav in wavs:
                try:
                    seg = AudioSegment.from_file(wav).set_channels(1)
                    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
                    if len(samples) == 0: continue
                    samples /= (np.abs(samples).max() + 1e-9)
                    sr = seg.frame_rate
                    dur = len(samples) / sr

                    # 1. RMS energy (want > 0.05)
                    rms = float(np.sqrt(np.mean(samples ** 2)))

                    # 2. SNR estimate: signal power vs noise floor (last 0.1s)
                    noise_len = min(int(sr * 0.1), len(samples) // 4)
                    noise_rms = float(np.sqrt(np.mean(samples[-noise_len:] ** 2))) + 1e-9
                    snr = min(rms / noise_rms, 20.0)

                    # 3. Spectral clarity: ratio of energy above 1kHz
                    from numpy.fft import rfft, rfftfreq
                    chunk = min(4096, len(samples))
                    F = np.abs(rfft(samples[:chunk] * np.hanning(chunk)))
                    freqs = rfftfreq(chunk, 1 / sr)
                    hi_energy = float(np.sum(F[freqs >= 1000] ** 2))
                    lo_energy = float(np.sum(F[freqs <  1000] ** 2)) + 1e-9
                    clarity = min(hi_energy / lo_energy, 2.0) / 2.0

                    # 4. Duration score (3-15s is ideal)
                    dur_score = 1.0 if 3 <= dur <= 15 else max(0, 1 - abs(dur - 9) / 9)

                    # Composite 0-100
                    composite = int((
                        min(rms * 5, 1.0) * 25 +
                        min(snr / 20, 1.0) * 35 +
                        clarity * 25 +
                        dur_score * 15
                    ))
                    scores.append((composite, os.path.basename(wav), dur, rms, snr, clarity))
                except Exception as e:
                    self._log(f"  skip {os.path.basename(wav)}: {e}", TEXT_MUT)

            scores.sort(reverse=True)
            self._log("── Ref Clip Scores (higher = better) ──", ACCENT)
            good = sum(1 for s, *_ in scores if s >= 60)
            ok   = sum(1 for s, *_ in scores if 40 <= s < 60)
            poor = sum(1 for s, *_ in scores if s < 40)
            for score, name, dur, rms, snr, clarity in scores:
                icon = "✅" if score >= 60 else ("⚠️ " if score >= 40 else "❌")
                self._log(
                    f"  {icon} {score:3d}  {name:<35s} "
                    f"{dur:4.1f}s  snr={snr:.1f}  clarity={clarity:.2f}",
                    ACCENT2 if score >= 60 else (WARNING if score >= 40 else DANGER))
            summary = f"{good} good · {ok} ok · {poor} poor — {len(scores)} total"
            self._log(f"── {summary} ──", ACCENT)
            # Cache scored paths so Prune button knows which to remove
            self._last_ref_scores = {
                os.path.join(self.qwen_ref_folder_var.get(), name): score
                for score, name, *_ in scores
            }
            self.after(0, lambda: self._qwen_ref_score_lbl.config(
                text=summary, fg=ACCENT2 if poor == 0 else WARNING))
            # Enable prune whenever there are any scored files — the dialog
            # handles the per-threshold "nothing to remove" message.
            self.after(0, lambda n=len(scores): self._prune_btn.config(
                state="normal" if n > 0 else "disabled",
                fg=DANGER if n > 0 else TEXT_MUT))
            self._set_status("scoring done", ACCENT2)
        except ImportError:
            self._log("numpy required for ref scoring.", DANGER)
        except Exception as e:
            self._log(f"Scoring failed: {e}", DANGER)

    def _prune_poor_refs(self):
        """Delete Qwen reference clips below a user-chosen score threshold."""
        if not self._last_ref_scores:
            messagebox.showinfo("Score first", "Run '📊 Score Qwen Refs' first."); return

        # Ask threshold via a small dialog
        dlg = tk.Toplevel(self)
        dlg.title("Prune Refs — Choose Threshold")
        dlg.configure(bg=CARD_BG)
        dlg.resizable(False, False)
        dlg.grab_set()
        tk.Label(dlg, text="Remove clips scored below:",
                 font=("Courier New", 9), bg=CARD_BG, fg=TEXT_PRI).pack(padx=16, pady=(14, 4))
        thresh_var = tk.IntVar(value=40)
        for val, lbl in [(40, "40  —  Poor only  (❌ red clips)"),
                         (60, "60  —  Poor + OK  (❌ + ⚠️  clips)")]:
            tk.Radiobutton(dlg, text=lbl, variable=thresh_var, value=val,
                           bg=CARD_BG, fg=TEXT_SEC, selectcolor=DARK_BG,
                           activebackground=CARD_BG, activeforeground=ACCENT3,
                           font=("Courier New", 8), bd=0).pack(anchor="w", padx=20, pady=2)
        chosen = [None]
        def _ok():
            chosen[0] = thresh_var.get(); dlg.destroy()
        def _cancel():
            dlg.destroy()
        btn_row = tk.Frame(dlg, bg=CARD_BG); btn_row.pack(pady=12)
        tk.Button(btn_row, text="Prune", command=_ok,
                  bg=DANGER, fg=DARK_BG, font=("Courier New", 8, "bold"),
                  relief="flat", padx=12, pady=4).pack(side="left", padx=6)
        tk.Button(btn_row, text="Cancel", command=_cancel,
                  bg=PANEL_BG, fg=TEXT_SEC, font=("Courier New", 8),
                  relief="flat", padx=12, pady=4).pack(side="left", padx=6)
        self.wait_window(dlg)
        if chosen[0] is None: return

        threshold = chosen[0]
        poor = {p: s for p, s in self._last_ref_scores.items() if s < threshold}
        if not poor:
            messagebox.showinfo("All clean",
                f"No clips below score {threshold} to remove."); return
        label = "Poor only (<40)" if threshold == 40 else f"Poor + OK (<{threshold})"

        # Custom scrollable confirm dialog (avoids screen overflow with many files)
        confirmed = [False]
        cdlg = tk.Toplevel(self)
        cdlg.title("Confirm Prune")
        cdlg.configure(bg=CARD_BG)
        cdlg.resizable(False, True)
        cdlg.grab_set()
        # Cap height to 60% of screen
        screen_h = self.winfo_screenheight()
        max_h = int(screen_h * 0.60)
        tk.Label(cdlg,
                 text=f"Permanently delete {len(poor)} clip(s) [{label}]?",
                 font=("Courier New", 9, "bold"), bg=CARD_BG, fg=TEXT_PRI
                 ).pack(padx=16, pady=(12, 4))
        # Scrollable file list
        list_frame = tk.Frame(cdlg, bg=CARD_BG)
        list_frame.pack(fill="both", expand=True, padx=16, pady=(0, 6))
        sb = tk.Scrollbar(list_frame)
        sb.pack(side="right", fill="y")
        lb = tk.Listbox(list_frame, bg=PANEL_BG, fg=DANGER,
                        font=("Courier New", 8), relief="flat", bd=0,
                        height=min(len(poor), 12),
                        yscrollcommand=sb.set)
        lb.pack(side="left", fill="both", expand=True)
        sb.config(command=lb.yview)
        for p, s in sorted(poor.items(), key=lambda x: x[1]):
            lb.insert("end", f"[{s:3d}]  {os.path.basename(p)}")
        # Clamp window height
        cdlg.update_idletasks()
        if cdlg.winfo_reqheight() > max_h:
            cdlg.geometry(f"460x{max_h}")
        # Confirm / cancel buttons — always visible at bottom
        cbtn_row = tk.Frame(cdlg, bg=CARD_BG)
        cbtn_row.pack(pady=10, fill="x", padx=16)
        def _confirm(): confirmed[0] = True; cdlg.destroy()
        def _ccancel(): cdlg.destroy()
        tk.Button(cbtn_row, text=f"Delete {len(poor)} files", command=_confirm,
                  bg=DANGER, fg=DARK_BG, font=("Courier New", 8, "bold"),
                  relief="flat", padx=12, pady=4).pack(side="left")
        tk.Button(cbtn_row, text="Cancel", command=_ccancel,
                  bg=PANEL_BG, fg=TEXT_SEC, font=("Courier New", 8),
                  relief="flat", padx=12, pady=4).pack(side="left", padx=(8, 0))
        self.wait_window(cdlg)
        if not confirmed[0]: return
        deleted = failed = 0
        for path in poor:
            try:
                if os.path.isfile(path):
                    os.remove(path)
                    self._log(f"  🗑 Deleted: {os.path.basename(path)}", WARNING)
                    deleted += 1
            except Exception as e:
                self._log(f"  ✗ Could not delete {os.path.basename(path)}: {e}", DANGER)
                failed += 1
        # Remove deleted files from cache
        for p in list(poor.keys()):
            self._last_ref_scores.pop(p, None)
        remaining_poor = sum(1 for s in self._last_ref_scores.values() if s < threshold)
        self._log(f"Pruned {deleted} clip(s){f', {failed} failed' if failed else ''}.", ACCENT2)
        self._set_status(f"pruned {deleted} clips", ACCENT2)
        # Keep prune enabled as long as any scored files remain
        remaining_total = len(self._last_ref_scores)
        if remaining_total == 0:
            self.after(0, lambda: self._prune_btn.config(state="disabled", fg=TEXT_MUT))
        # Update summary label
        good = sum(1 for s in self._last_ref_scores.values() if s >= 60)
        ok   = sum(1 for s in self._last_ref_scores.values() if 40 <= s < 60)
        total = len(self._last_ref_scores)
        self.after(0, lambda: self._qwen_ref_score_lbl.config(
            text=f"{good} good · {ok} ok · {remaining_poor} poor — {total} total",
            fg=ACCENT2 if remaining_poor == 0 else WARNING))

    # ── Qwen Feature 3: Mumble Detection ────────────────────────────────────
    def _is_mumbled(self, wav_path: str) -> bool:
        """
        Return True if the output audio is garbled, silent, or too short.
        Criteria:
          - RMS < 0.01  (near-silent)
          - Zero-crossing rate < 400/s  (no voiced speech)
          - Duration < 0.5s
        """
        try:
            import numpy as np
            import wave as _wave
            with _wave.open(wav_path, "rb") as wf:
                sr = wf.getframerate()
                nf = wf.getnframes()
                raw = wf.readframes(nf)
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            dur = len(samples) / sr
            if dur < 0.5:
                return True
            rms = float(np.sqrt(np.mean(samples ** 2)))
            if rms < 0.01:
                return True
            zc = int(np.sum(np.diff(np.sign(samples)) != 0))
            zcr = zc / dur
            if zcr < 400:
                return True
            return False
        except Exception:
            return False

    def _play_job_output(self, idx):
        """Play the output file for a specific job index."""
        path = self._job_output_files.get(idx)
        if not path or not os.path.isfile(path):
            self._log(f"No output file for job {idx} yet.", TEXT_MUT); return
        self._stop_playback()
        self._current_file = path
        self._load_waveform(path)
        self._play_selected()

    def _export_json(self):
        """Export current job list to a JSON file compatible with Import JSON."""
        jobs = [(var.get().strip(), nv.get().strip())
                for var, _, __, nv, _pb in self._job_entries
                if var.get().strip()]
        if not jobs:
            messagebox.showwarning("No jobs", "No job text to export."); return
        path = filedialog.asksaveasfilename(
            title="Export jobs as JSON",
            defaultextension=".json",
            filetypes=[("JSON files","*.json"),("All","*.*")])
        if not path: return
        try:
            data = [{"Dialogue": text, "Name": name} for text, name in jobs]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._log(f"Exported {len(data)} job(s) → {os.path.basename(path)}", ACCENT2)
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    # ── output player + waveform ───────────────────────────────────────────────
    def _build_output_player(self, parent):
        c = self._card(parent, "OUTPUT FILES  —  PLAYBACK")

        # file list
        list_frame = tk.Frame(c, bg=CARD_BG); list_frame.pack(fill="x", pady=(0, 6))
        self._file_listbox = tk.Listbox(
            list_frame, bg=PANEL_BG, fg=TEXT_PRI,
            selectbackground=ACCENT, selectforeground=DARK_BG,
            font=("Courier New", 8), relief="flat", bd=0,
            height=6, activestyle="none")
        self._file_listbox.pack(side="left", fill="both", expand=True)
        file_scroll = ttk.Scrollbar(list_frame, orient="vertical",
                                     command=self._file_listbox.yview)
        file_scroll.pack(side="right", fill="y")
        self._file_listbox.configure(yscrollcommand=file_scroll.set)
        self._file_listbox.bind("<<ListboxSelect>>", self._on_file_select)
        self._file_listbox.bind("<Double-Button-1>", lambda e: self._play_selected())

        # refresh + open folder buttons
        rf_row = tk.Frame(c, bg=CARD_BG); rf_row.pack(fill="x", pady=(0, 6))
        self._btn(rf_row, "Refresh list", self._refresh_output_list, small=True)
        tk.Frame(rf_row, width=8, bg=CARD_BG).pack(side="left")
        self._btn(rf_row, "Open folder", self._open_output_folder, small=True)
        tk.Frame(rf_row, width=8, bg=CARD_BG).pack(side="left")
        self._btn(rf_row, "Copy Path", self._copy_out_path, small=True)
        tk.Frame(rf_row, width=8, bg=CARD_BG).pack(side="left")
        self._btn(rf_row, "✕ Delete All", self._delete_all_outputs, small=True)


        # waveform canvas
        self._wave_canvas = tk.Canvas(c, bg=PANEL_BG, height=70,
                                       highlightthickness=0)
        self._wave_canvas.pack(fill="x", pady=(0, 6))
        self._wave_canvas.bind("<Configure>", lambda e: self._redraw_waveform())

        # file info label
        self._file_info_lbl = tk.Label(c, text="No file selected",
                                        font=("Courier New", 7),
                                        bg=CARD_BG, fg=TEXT_MUT)
        self._file_info_lbl.pack(anchor="w", pady=(0, 6))

        # playback controls
        ctrl = tk.Frame(c, bg=CARD_BG); ctrl.pack(fill="x")
        self._play_btn = self._btn(ctrl, "▶  Play",  self._play_selected,
                                    accent=True, small=True)
        tk.Frame(ctrl, width=6, bg=CARD_BG).pack(side="left")
        self._stop_btn = self._btn(ctrl, "■  Stop",  self._stop_playback,
                                    small=True)
        tk.Frame(ctrl, width=6, bg=CARD_BG).pack(side="left")
        self._btn(ctrl, "Delete file", self._delete_selected, small=True,
                  color="#2a1a1a", fg_col=DANGER)

        if not self._pygame_ok:
            tk.Label(c, text="⚠  pip install pygame  to enable playback",
                     font=("Courier New", 7), bg=CARD_BG, fg=WARNING,
                     ).pack(anchor="w", pady=(4, 0))

        # ── Audio Analyzer launcher ──
        tk.Frame(c, bg=BORDER, height=1).pack(fill="x", pady=(10, 6))
        analyzer_row = tk.Frame(c, bg=CARD_BG); analyzer_row.pack(fill="x")
        self._btn(analyzer_row, "🔬 Audio Analyzer",
                  self._open_audio_analyzer, accent=True)
        tk.Label(analyzer_row,
                 text="  Detect clipping, noise, muffling, DC offset & more",
                 font=("Courier New", 7), bg=CARD_BG, fg=TEXT_MUT).pack(side="left")

        # internal waveform data
        self._wave_data = None
        self._refresh_output_list()

    def _refresh_output_list(self):
        folder = self.out_folder_var.get()
        if self._batch_out_enabled.get():
            custom = self._batch_out_folder.get().strip()
            if custom and os.path.isdir(custom):
                folder = custom

        self._file_listbox.delete(0, "end")
        if os.path.isdir(folder):
            wavs = sorted(glob.glob(os.path.join(folder, "*.wav")), reverse=True)
            for w in wavs:
                self._file_listbox.insert("end", os.path.basename(w))
        self._current_file = None
        self._wave_data    = None
        self._wave_canvas.delete("all")
        self._file_info_lbl.config(text=f"Output folder: {folder}")

    def _on_file_select(self, event=None):
        sel = self._file_listbox.curselection()
        if not sel: return
        fname = self._file_listbox.get(sel[0])
        
        folder = self.out_folder_var.get()
        if self._batch_out_enabled.get():
            custom = self._batch_out_folder.get().strip()
            if custom and os.path.isdir(custom):
                folder = custom
                
        self._current_file = os.path.join(folder, fname)
        self._load_waveform(self._current_file)

    def _load_waveform(self, path):
        """Read WAV samples and store downsampled peaks for drawing."""
        try:
            import numpy as np
            with wave.open(path, "rb") as wf:
                n_frames = wf.getnframes()
                n_ch     = wf.getnchannels()
                sw       = wf.getsampwidth()
                sr       = wf.getframerate()
                raw      = wf.readframes(n_frames)

            dtype = np.int16 if sw == 2 else np.int8
            samples = np.frombuffer(raw, dtype=dtype)
            if n_ch > 1:
                samples = samples[::n_ch]
            norm = samples.astype(np.float32) / (32768 if sw == 2 else 128)

            # duration info
            dur = len(norm) / sr
            size_kb = os.path.getsize(path) // 1024
            self._file_info_lbl.config(
                text=f"{os.path.basename(path)}   {dur:.1f}s · {sr} Hz · {size_kb} KB",
                fg=TEXT_SEC)

            # downsample to ~300 peaks for display
            chunk = max(1, len(norm) // 300)
            peaks = [float(np.max(np.abs(norm[i:i+chunk])))
                     for i in range(0, len(norm), chunk)]
            self._wave_data = peaks
            self._redraw_waveform()
        except Exception as e:
            self._wave_data = None
            self._wave_canvas.delete("all")
            self._file_info_lbl.config(text=f"Could not read file: {e}", fg=DANGER)

    def _redraw_waveform(self):
        self._wave_canvas.delete("all")
        if not self._wave_data: return
        w = self._wave_canvas.winfo_width()
        h = self._wave_canvas.winfo_height()
        if w < 2 or h < 2: return

        mid  = h / 2
        n    = len(self._wave_data)
        step = w / max(n, 1)

        for i, peak in enumerate(self._wave_data):
            x     = i * step
            amp   = peak * (mid - 2)
            # colour: gradient from ACCENT (left) towards ACCENT2 (right)
            ratio = i / max(n-1, 1)
            r = int(0x7c + ratio*(0x5d - 0x7c))
            g = int(0x6a + ratio*(0xca - 0x6a))
            b = int(0xf7 + ratio*(0xa5 - 0xf7))
            col = f"#{r:02x}{g:02x}{b:02x}"
            self._wave_canvas.create_line(
                x, mid - amp, x, mid + amp, fill=col, width=max(1, step*0.7))

    # playback
    def _play_selected(self):
        if not self._current_file:
            sel = self._file_listbox.curselection()
            if sel:
                fname = self._file_listbox.get(sel[0])
                self._current_file = os.path.join(
                    self.out_folder_var.get(), fname)
            else:
                messagebox.showinfo("No file", "Select a file first."); return
        if not self._pygame_ok:
            messagebox.showwarning("pygame missing",
                "Install pygame to enable playback:\npip install pygame")
            return
        self._stop_playback()
        self._playing = True
        self._play_btn.config(text="▶  Playing…")
        self._play_thread = threading.Thread(
            target=self._run_playback,
            args=(self._current_file,), daemon=True)
        self._play_thread.start()

    def _run_playback(self, path):
        try:
            import pygame
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() and self._playing:
                import time; time.sleep(0.05)
        except Exception as e:
            self._log(f"Playback error: {e}", DANGER)
        finally:
            self._playing = False
            self.after(0, lambda: self._play_btn.config(text="▶  Play"))

    def _stop_playback(self):
        self._playing = False
        if self._pygame_ok:
            try:
                import pygame
                pygame.mixer.music.stop()
            except Exception:
                pass
        self._play_btn.config(text="▶  Play")

    def _delete_selected(self):
        if not self._current_file: return
        if not messagebox.askyesno("Delete",
                f"Delete {os.path.basename(self._current_file)}?"):
            return
        try:
            self._stop_playback()
            os.remove(self._current_file)
            self._current_file = None
            self._wave_data    = None
            self._wave_canvas.delete("all")
            self._file_info_lbl.config(text="File deleted", fg=TEXT_MUT)
            self._refresh_output_list()
        except Exception as e:
            messagebox.showerror("Delete failed", str(e))

    def _open_output_folder(self):
        folder = self.out_folder_var.get()
        if self._batch_out_enabled.get():
            custom = self._batch_out_folder.get().strip()
            if custom and os.path.isdir(custom):
                folder = custom

        if os.path.isdir(folder):
            import subprocess, sys
            if sys.platform == "win32":
                subprocess.Popen(["explorer", os.path.normpath(folder)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])

    def _copy_out_path(self):
        folder = self.out_folder_var.get()
        if self._batch_out_enabled.get():
            custom = self._batch_out_folder.get().strip()
            if custom and os.path.isdir(custom):
                folder = custom

        self.clipboard_clear()
        self.clipboard_append(os.path.abspath(folder))
        self._log("Folder path copied to clipboard.", ACCENT2)

    def _delete_all_outputs(self):
        folder = self.out_folder_var.get()
        if self._batch_out_enabled.get():
            custom = self._batch_out_folder.get().strip()
            if custom and os.path.isdir(custom):
                folder = custom

        files = glob.glob(os.path.join(folder, "*.wav"))
        if not files: return
        if not messagebox.askyesno("Delete All",
                f"Delete all {len(files)} WAV files in {os.path.basename(folder)}?"):
            return
        self._stop_playback()
        for f in files:
            try: os.remove(f)
            except: pass
        self._refresh_output_list()
        self._log("Output folder cleared.", TEXT_MUT)


    # ══ RIGHT PANEL ═════════════════════════════════════════════════════════

    def _build_log(self, parent):
        tk.Label(parent, text="LOG", font=("Courier New", 9, "bold"),
                 bg=DARK_BG, fg=TEXT_SEC).pack(anchor="w", pady=(0, 4))
        self._log_text = scrolledtext.ScrolledText(
            parent, bg=PANEL_BG, fg=TEXT_PRI, font=("Courier New", 8),
            relief="flat", bd=0, insertbackground=TEXT_PRI,
            state="disabled", wrap="word")
        self._log_text.pack(fill="both", expand=True)
        for tag, col in [("ok",ACCENT2),("warn",WARNING),("err",DANGER),
                         ("acc",ACCENT),("mut",TEXT_MUT),("pri",TEXT_PRI)]:
            self._log_text.tag_config(tag, foreground=col)
        clr = tk.Frame(parent, bg=DARK_BG); clr.pack(fill="x", pady=(4, 0))
        self._btn(clr, "Clear log", self._clear_log, small=True)

    def _log(self, msg, color=None):
        tag = {ACCENT2:"ok",WARNING:"warn",DANGER:"err",
               ACCENT:"acc",TEXT_MUT:"mut"}.get(color,"pri")
        ts  = datetime.now().strftime("%H:%M:%S")
        def _w():
            self._log_text.config(state="normal")
            self._log_text.insert("end", f"[{ts}] {msg}\n", tag)
            self._log_text.see("end")
            self._log_text.config(state="disabled")
        self.after(0, _w)

    def _clear_log(self):
        self._log_text.config(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.config(state="disabled")

    # ══ MODEL LOADING (via subprocess workers) ══════════════════════════════

    def _worker_log(self, text, level="info"):
        """Receive log message from a worker; route to the GUI log panel."""
        color = {"ok": ACCENT2, "warn": WARNING, "error": DANGER}.get(level, TEXT_MUT)
        self._log(f"  {text}", color)

    def _load_model(self):
        backend = self._backend_var.get()
        if backend == "xtts":
            if self._xtts_worker.is_alive():
                messagebox.showinfo("Model", "XTTS v2 worker already running."); return
            self._log("Starting XTTS v2 worker (xtts-env-py310)…", WARNING)
        else:
            if self._qwen_worker.is_alive():
                messagebox.showinfo("Model", "Qwen3-TTS worker already running."); return
            if not os.path.isfile(_QWEN_PYTHON):
                self._log("qwen-env-py310 not found. Run setup_qwen_env.bat first.", DANGER)
                messagebox.showerror("No Qwen env",
                    "Run setup_qwen_env.bat to create the Qwen Python environment.")
                return
            self._log("Starting Qwen3-TTS worker (qwen-env-py310)…", WARNING)

        self._load_model_btn.config(state="disabled")
        self._set_status("loading model…", WARNING)
        worker = self._xtts_worker if backend == "xtts" else self._qwen_worker

        def on_ready():
            label = "XTTS v2" if backend == "xtts" else "Qwen3-TTS"
            self._log(f"{label} worker ready!", ACCENT2)
            self._set_status("ready", ACCENT2)
            if backend == "xtts":
                self._tts_model = True   # signal to _generate_all check
                self.after(0, lambda: self._model_lbl.config(
                    text="XTTS v2  (worker)", fg=ACCENT2))
            else:
                self._qwen_model = True
                self.after(0, lambda: self._model_lbl.config(
                    text="Qwen3-TTS  (worker)", fg=ACCENT3))
            self.after(0, lambda: self._load_model_btn.config(state="normal"))

        def on_error(msg):
            self._log(f"Worker start failed: {msg}", DANGER)
            self._set_status("model error", DANGER)
            self.after(0, lambda: self._load_model_btn.config(state="normal"))

        worker.start(on_ready, on_error)

    # ══ GENERATION ══════════════════════════════════════════════════════════

    def _get_ref_wavs(self):
        # Use the correct reference folder for the active backend
        backend = self._backend_var.get()
        if backend == "qwen":
            folder = self.qwen_ref_folder_var.get()
        else:
            folder = self.xtts_ref_folder_var.get()
        self.ref_folder_var = self.qwen_ref_folder_var if backend == "qwen" else self.xtts_ref_folder_var
        wavs   = glob.glob(os.path.join(folder, "*.wav"))
        AudioSegment = _import_pydub()
        valid, skipped = [], []
        for w in wavs:
            try:
                audio = AudioSegment.from_file(w)
                dur = len(audio) / 1000
                if dur < MIN_DUR_SEC:
                    skipped.append(w)
                    continue
                
                # Check for incompatible sample rate or channels (causes XTTS corruption)
                if audio.frame_rate != TARGET_SR or audio.channels != TARGET_CHANNELS:
                    self._log(f"  Normalizing format: {os.path.basename(w)} → Mono 24kHz", WARNING)
                    audio = audio.set_channels(TARGET_CHANNELS).set_frame_rate(TARGET_SR).set_sample_width(SAMPLE_WIDTH)
                    audio.export(w, format="wav")
                    
                valid.append(w)
            except Exception as e:
                self._log(f"Error reading {w}: {e}", DANGER)
                skipped.append(w)
        return valid, skipped

    def _enhance_xtts_refs(self):
        """Option B: Trim silence, normalize loudness, and apply presence EQ."""
        valid_wavs, _ = self._get_ref_wavs()
        if not valid_wavs:
            messagebox.showwarning("No WAVs", "No reference WAVs found to enhance."); return
        
        if not messagebox.askyesno("Confirm",
                f"This will permanently clean {len(valid_wavs)} clips:\n"
                "  • Trim leading/trailing silence\n"
                "  • Normalize to -3 dBFS\n"
                "  • +4 dB presence boost at 2 kHz (reduces muffling)\n\nProceed?"):
            return

        def _run():
            self._set_status("enhancing refs…", WARNING)
            self._log(f"Enhancing {len(valid_wavs)} reference clips…", WARNING)
            AudioSegment = _import_pydub()
            try:
                import numpy as np
                has_numpy = True
            except ImportError:
                has_numpy = False

            ok = fail = 0
            for wav in valid_wavs:
                try:
                    audio = AudioSegment.from_file(wav)
                    # 2. Normalize to -3 dBFS
                    audio = audio.normalize(headroom=3.0)
                    # 3. Presence EQ: +4 dB shelf boost at ~2 kHz using numpy
                    if has_numpy:
                        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                        sr = audio.frame_rate
                        # Simple single-pole high-shelf filter at 2kHz, +4dB
                        gain_linear = 10 ** (4.0 / 20.0)
                        fc = 2000.0
                        w0 = 2 * np.pi * fc / sr
                        # Bilinear transform shelf: y[n] = x[n] + alpha*(x[n]-x[n-1]) 
                        alpha = (gain_linear - 1.0) * (1.0 - np.cos(w0)) / (1.0 + np.cos(w0) + 1e-9)
                        out = np.zeros_like(samples)
                        out[0] = samples[0]
                        for i in range(1, len(samples)):
                            out[i] = samples[i] + alpha * (samples[i] - samples[i-1])
                        # Clip and convert back
                        out = np.clip(out, -32768, 32767).astype(np.int16)
                        from pydub import AudioSegment as AS
                        audio = AS(out.tobytes(), frame_rate=sr,
                                   sample_width=2, channels=audio.channels)
                    audio.export(wav, format="wav")
                    ok += 1
                    self._log(f"  ✓ {os.path.basename(wav)}", ACCENT2)
                except Exception as e:
                    fail += 1
                    self._log(f"  ✗ {os.path.basename(wav)}: {e}", DANGER)

            self._log(f"Done — {ok} enhanced, {fail} failed.", ACCENT2)
            self._set_status("refs enhanced", ACCENT2)

        threading.Thread(target=_run, daemon=True).start()

    def _stop_gen(self):
        """Signal the generation loop to stop immediately."""
        self._stop_generation.set()
        self._log("Global Stop — killing worker processes immediately…", DANGER)
        self._set_status("killing processes…", DANGER)
        self.after(0, lambda: self._stop_btn.config(state="disabled"))
        
        # Hard kill the active backends for immediate stop
        for w in [self._xtts_worker, self._qwen_worker, self._rvc_worker]:
            try:
                if w.is_alive():
                    w._proc.kill()
            except:
                pass

    def _generate_all(self):
        backend = self._backend_var.get()
        worker  = self._xtts_worker if backend == "xtts" else self._qwen_worker
        if not worker.is_alive():
            messagebox.showerror("No model", f"Load {'XTTS v2' if backend=='xtts' else 'Qwen3-TTS'} first."); return

        jobs = []
        for i, (var, st, sv, nv, pb, sk, sp) in enumerate(self._job_entries, 1):
            if var.get().strip():
                jobs.append((i, var.get().strip(), st, sv.get(), nv.get().strip()))

        if not jobs:
            messagebox.showwarning("No text", "Enter at least one text job."); return

        bs = self._batch_size_var.get() or 10
        batches = [jobs[i:i + bs] for i in range(0, len(jobs), bs)]

        out_dir = self.out_folder_var.get()
        if self._batch_out_enabled.get():
            custom = self._batch_out_folder.get().strip()
            if custom and os.path.isdir(custom):
                out_dir = custom
            elif custom: # If it's a valid path but not a dir, try creating it
                try: 
                    os.makedirs(custom, exist_ok=True)
                    out_dir = custom
                except: pass
        
        os.makedirs(out_dir, exist_ok=True)
        self._stop_generation.clear()
        self._skip_job_event.clear()
        self._skip_batch_event.clear()
        self._stop_batch_event.clear()
        
        self._gen_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._skip_batch_btn.config(state="normal")
        self._stop_batch_btn.config(state="normal")

        threading.Thread(target=self._run_generation,
                         args=(batches, out_dir), daemon=True).start()

    def _run_generation(self, batches, out_dir):
        import re
        backend = self._backend_var.get()
        worker  = self._xtts_worker if backend == "xtts" else self._qwen_worker
        
        total_jobs = sum(len(b) for b in batches)
        processed_count = 0

        profile_path = None
        valid_wavs, skipped = [], []
        
        if backend == "xtts":
            profile_path = self._profile_path
            if profile_path:
                self._log("XTTS v2 — Using loaded voice profile.", ACCENT)
            else:
                valid_wavs, skipped = self._get_ref_wavs()
                if not valid_wavs:
                    self._log(f"No valid XTTS references found.", DANGER)
                    self.after(0, lambda: self._gen_btn.config(state="normal")); return
                self._log(f"XTTS v2 — {len(valid_wavs)} ref(s), {len(skipped)} skipped.", ACCENT)
        else:
            # QWEN
            profile_path = self._qwen_profile_pth
            if profile_path:
                self._log("QWEN — Using multi-reference profile (fast mode).", ACCENT)
            else:
                valid_wavs, skipped = self._get_ref_wavs()
                if not valid_wavs:
                    self._log(f"No valid Qwen references found.", DANGER)
                    self.after(0, lambda: self._gen_btn.config(state="normal")); return
                self._log(f"QWEN — {len(valid_wavs)} ref(s), {len(skipped)} skipped.", ACCENT)

        # Pull settings from the right backend vars
        if backend == "qwen":
            speed = 1.0
            temp  = self.qwen_temp_var.get()
            rep   = self.qwen_rep_pen_var.get()
            top_k = int(self.qwen_top_k_var.get())
            top_p = self.qwen_top_p_var.get()
            active_presets = QWEN_PRESETS
        else:
            speed = self.xtts_speed_var.get()
            temp  = self.xtts_temp_var.get()
            rep   = self.xtts_rep_pen_var.get()
            top_k = int(self.xtts_top_k_var.get())
            top_p = self.xtts_top_p_var.get()
            active_presets = XTTS_PRESETS

        lang  = self._lang_map.get(self.lang_var.get(), self.lang_var.get())
        global_preset = self._active_preset.get() or "Natural"
        use_icl = self._qwen_voice_design.get()

        self._gen_progress["maximum"] = total_jobs
        self._gen_progress["value"]   = 0

        rvc_on = (backend == "xtts" and self._rvc_enabled.get()
                  and self._rvc_worker.is_alive()
                  and bool(self._rvc_model_var.get()))
        xtts_pro = self._xtts_audio_pro.get()

        # If Auto-RVC is on and scope=global, resolve once before the loop
        if (rvc_on and self._rvc_auto_var.get()
                and self._rvc_auto_scope.get() == "global"):
            self._apply_global_auto_rvc()

        total_batches = len(batches)
        
        # Outer Batch Loop
        for b_idx, batch in enumerate(batches, 1):
            if self._stop_generation.is_set(): break
            
            self._log(f"--- Processing Batch {b_idx}/{total_batches} ({len(batch)} jobs) ---", WARNING)
            self._skip_batch_event.clear()
            self._stop_batch_event.clear()

            # Inner Job Loop
            for idx, text, status_lbl, job_mood, custom_filename in batch:
                processed_count += 1
                self._skip_job_event.clear()
                
                # 1. Batch-level Interrupts
                if self._stop_generation.is_set():
                    self.after(0, lambda s=status_lbl: s.config(text="○", fg=TEXT_MUT))
                    break
                if self._skip_batch_event.is_set():
                    self.after(0, lambda s=status_lbl: s.config(text="⏭", fg=TEXT_MUT))
                    continue
                if self._stop_batch_event.is_set():
                    self.after(0, lambda s=status_lbl: s.config(text="○", fg=TEXT_MUT))
                    break
                    
                # 2. Job-level skip
                if self._skip_job_event.is_set():
                    self._log(f"  Skipping Job {processed_count}...", WARNING)
                    self.after(0, lambda s=status_lbl: s.config(text="⏭", fg=TEXT_MUT))
                    continue

                j_pname = job_mood if job_mood != "[Auto]" else global_preset
                j_temp, j_speed, j_rep, j_top_k, j_top_p = temp, speed, rep, top_k, top_p
                if j_pname in active_presets:
                    p = active_presets[j_pname]
                    j_temp, j_speed, j_rep, j_top_k, j_top_p = p[0], p[1], p[2], p[3], p[4]

                self.after(0, lambda s=status_lbl: s.config(text="◌", fg=WARNING))
                self._set_status(f"job {processed_count}/{total_jobs}…", WARNING)

                ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
                pfx      = f"{backend}_{j_pname+'_' if j_pname else ''}"
                
                if custom_filename:
                    base_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', custom_filename)
                    if not base_name.lower().endswith(".wav"):
                        base_name += ".wav"
                    out = os.path.join(out_dir, base_name)
                elif self._naming_mode.get() == "Sequential":
                    out = os.path.join(out_dir, f"{idx:02d}.wav")
                else:
                    clean    = re.sub(r'[^a-zA-Z0-9]', '_', text[:20].strip())
                    out      = os.path.join(out_dir, f"{pfx}{idx:02d}_{clean}_{ts[-6:]}.wav")
                    
                send_text = text
                if backend == "qwen" and self._qwen_emotion_tags.get():
                    if use_icl:
                        self._log(f"  [Qwen tags] skipped — ICL mode active", TEXT_MUT)
                    else:
                        send_text = self._apply_qwen_emotion_tags(text)

                preview  = f'"{text[:50]}…"' if len(text) > 50 else f'"{text}"'
                self._log(f"[{backend.upper()}] B{b_idx} | Job {processed_count}: {preview}", ACCENT)

                _auto_rvc_on    = rvc_on and self._rvc_auto_var.get()
                _auto_rvc_scope = self._rvc_auto_scope.get()

                # Feature 4: streaming chunk handler for Qwen
                if backend == "qwen" and self._qwen_stream.get():
                    import queue as _q
                    _chunk_queue = _q.Queue()
                    _stream_active = [True]
                    def _on_chunk(path):
                        _chunk_queue.put(path)
                        self.after(0, lambda: status_lbl.config(text="▶", fg=ACCENT))
                    def _drain_chunks():
                        if not _stream_active[0]: return
                        if self._pygame_ok:
                            try:
                                import pygame
                                if not pygame.mixer.music.get_busy():
                                    path = _chunk_queue.get_nowait()
                                    pygame.mixer.music.load(path)
                                    pygame.mixer.music.play()
                            except: pass
                        if _stream_active[0]: self.after(150, _drain_chunks)
                    self._qwen_worker._on_chunk = _on_chunk
                    self.after(200, _drain_chunks)
                else:
                    _stream_active = [False]
                    self._qwen_worker._on_chunk = None

                cmd = {
                    "action":      "generate",
                    "text":        send_text,
                    "refs":        valid_wavs,
                    "profile_path": profile_path,
                    "lang":        lang,
                    "out":         out,
                    "speed":       j_speed,
                    "temperature": j_temp,
                    "rep_pen":     j_rep,
                    "top_k":       j_top_k,
                    "top_p":       j_top_p,
                    "preset":      j_pname,
                    "use_icl":     use_icl,
                    "post_process": xtts_pro if (backend == "xtts" and not rvc_on) else False,
                    "stream":       backend == "qwen" and self._qwen_stream.get(),
                }
                
                # Feature 3: retry on mumble (Qwen only)
                _max_tries = 3 if (backend == "qwen" and self._qwen_retry_mumble.get()) else 1
                resp = None
                for _attempt in range(_max_tries):
                    if _attempt > 0:
                        self._log(f"  → Mumble retry {_attempt}/{_max_tries-1}", WARNING)
                        cmd["temperature"] = max(0.20, cmd["temperature"] - 0.15 * _attempt)
                        cmd["out"] = out.replace(".wav", f"_try{_attempt}.wav")
                    worker.send(cmd)
                    
                    # Fast polling loop for immediate kill responsiveness
                    while True:
                        resp = worker.get_response(timeout=0.5)
                        if resp.get("message") != "Response timeout.":
                            break
                        if self._stop_generation.is_set() or not worker.is_alive():
                            resp = {"status": "error", "message": "Job killed by global stop."}
                            break
                    
                    if resp.get("status") != "done": break
                    if not (backend == "qwen" and self._qwen_retry_mumble.get()): break
                    if not self._is_mumbled(resp.get("file", "")): break
                
                if resp is None: resp = {"status": "error", "message": "no response"}
                _stream_active[0] = False
                
                if self._stop_generation.is_set():
                    self.after(0, lambda s=status_lbl: s.config(text="○", fg=TEXT_MUT))
                    break

                if resp.get("status") == "done":
                    final_f = resp['file']
                    
                    # RVC
                    if rvc_on:
                        m_name = self._rvc_model_var.get()
                        if m_name:
                            self._set_status(f"RVC {processed_count}…", ACCENT3)
                            m_path = os.path.join("rvc_models", m_name)
                            m_dir = os.path.dirname(m_path)
                            model_stem = os.path.splitext(os.path.basename(m_path))[0]
                            index_candidates = glob.glob(os.path.join(m_dir, "*.index"))
                            i_path = next((p for p in index_candidates if model_stem in os.path.basename(p)), 
                                          index_candidates[0] if index_candidates else None)
                            
                            if _auto_rvc_on and _auto_rvc_scope == "per-job":
                                _p, _ir, _fm, _pn, _em, _it = self._auto_rvc_preset_for(text)
                            else:
                                _p, _ir, _fm = self._rvc_pitch_var.get(), self._rvc_index_var.get(), self._rvc_method_var.get()
                            
                            rvc_cmd = {
                                "action": "infer", "input": final_f, "out": final_f.replace(".wav", "_rvc.wav"),
                                "model": m_path, "index": i_path if i_path and os.path.isfile(i_path) else None,
                                "pitch": _p, "index_rate": _ir, "f0_method": _fm,
                            }
                            self._rvc_worker.send(rvc_cmd)
                            
                            r_resp = None
                            while True:
                                r_resp = self._rvc_worker.get_response(timeout=0.5)
                                if r_resp.get("message") != "Response timeout.":
                                    break
                                if self._stop_generation.is_set() or not self._rvc_worker.is_alive():
                                    r_resp = {"status": "error", "message": "RVC process killed."}
                                    break
                                    
                            if r_resp.get("status") == "done":
                                import time; time.sleep(0.1)
                                os.replace(r_resp['file'], final_f)
                                self._log(f"  → RVC applied ({_p:+d})", ACCENT3)
                    
                    # Skyrim
                    if self._skyrim_enabled.get():
                        self._set_status(f"Skyrim FUZ {processed_count}…", WARNING)
                        try:
                            conv = SkyrimConverter(facefx_path=self._skyrim_facefx_path.get(), 
                                                  xwma_path=self._skyrim_xwma_path.get(), fonix_path=self._skyrim_fonix_path.get())
                            plugin, vtype = self._skyrim_plugin.get().strip(), self._skyrim_voice_type.get().strip()
                            formid = os.path.splitext(os.path.basename(custom_filename))[0] if custom_filename else f"{idx:02d}_{re.sub(r'[^a-zA-Z0-9]', '_', text[:30].strip())}"
                            skybin = os.path.join("output", "Skyrim_Export", "sound", "voice", plugin, vtype)
                            os.makedirs(skybin, exist_ok=True)
                            conv.create_skyrim_fuz(final_f, text, os.path.join(skybin, f"{formid}.fuz"))
                            self._log(f"  ✓ Skyrim FUZ: {formid}.fuz", ACCENT2)
                        except Exception as e: self._log(f"  ! Skyrim fail: {e}", DANGER)

                    self._log(f"  → {os.path.basename(final_f)} ({resp.get('duration',0):.1f}s)", ACCENT2)
                    self._job_output_files[idx] = final_f
                    # Enable play button
                    _pb = next((pb for v,s,_,__,pb,sk,sp in self._job_entries if s is status_lbl), None)
                    if _pb: self.after(0, lambda b=_pb: b.config(state="normal"))
                    self.after(0, lambda s=status_lbl: s.config(text="●", fg=ACCENT2))
                else:
                    self._log(f"  Job {processed_count} failed: {resp.get('message')}", DANGER)
                    self.after(0, lambda s=status_lbl: s.config(text="✕", fg=DANGER))

                self._gen_progress["value"] = processed_count

        self.after(0, lambda: self._gen_btn.config(state="normal"))
        self.after(0, lambda: self._stop_btn.config(state="disabled"))
        self.after(0, lambda: self._skip_batch_btn.config(state="disabled"))
        self.after(0, lambda: self._stop_batch_btn.config(state="disabled"))
        
        if self._stop_generation.is_set():
            self._set_status(f"stopped ({processed_count}/{total_jobs})", WARNING)
        else:
            self._set_status("all complete", ACCENT2)
            self._log(f"Done — {processed_count} jobs total.", ACCENT2)
        self.after(500, self._refresh_output_list)

    # ══ HELPERS ═════════════════════════════════════════════════════════════

    def _set_status(self, text, color=TEXT_MUT):
        def _u():
            self._status_lbl.config(text=text, fg=color)
            self._status_dot.config(fg=color)
        self.after(0, _u)

    # ══ AUDIO ANALYZER PRO ═════════════════════════════════════════════════

    def _open_audio_analyzer(self):
        """
        Unified Audio Analyzer Pro window.
        Features: WAV quality scoring, sortable columns (Score default),
        audio playback, filter dropdown, integrated delete / prune workflow.
        """
        win = tk.Toplevel(self)
        win.title("VoiceTTSr — Audio Analyzer")
        win.geometry("1140x740")
        win.minsize(900, 580)
        win.configure(bg=DARK_BG)

        # ── results cache: full_path -> result_dict ───────────────────────
        _cache = {}   # populated by _run_audio_analysis

        # ── header ───────────────────────────────────────────────────────
        hdr = tk.Frame(win, bg=DARK_BG)
        hdr.pack(fill="x", padx=20, pady=(14, 0))
        tk.Label(hdr, text="Audio Analyzer",
                 font=("Courier New", 14, "bold"), bg=DARK_BG, fg=ACCENT).pack(side="left")
        tk.Label(hdr,
                 text="  Clipping  Noise  Muffling  DC Offset  Silence  Codec Artefacts",
                 font=("Courier New", 8), bg=DARK_BG, fg=TEXT_MUT).pack(side="left", pady=4)
        tk.Frame(win, bg=BORDER, height=1).pack(fill="x", padx=20, pady=8)

        # ── source selector ──────────────────────────────────────────────
        src_card = tk.Frame(win, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1)
        src_card.pack(fill="x", padx=20, pady=(0, 6))
        src_inner = tk.Frame(src_card, bg=CARD_BG)
        src_inner.pack(fill="x", padx=12, pady=8)

        tk.Label(src_inner, text="Folder:", font=("Courier New", 9, "bold"),
                 bg=CARD_BG, fg=TEXT_SEC).pack(side="left")
        for label, getter in [
            ("Output",    lambda: self.out_folder_var.get()),
            ("XTTS Refs", lambda: self.xtts_ref_folder_var.get()),
            ("Qwen Refs", lambda: self.qwen_ref_folder_var.get()),
        ]:
            tk.Button(src_inner, text=label,
                      command=lambda g=getter: _folder_var.set(g()),
                      bg=PANEL_BG, fg=TEXT_SEC, font=("Courier New", 8),
                      relief="flat", padx=8, pady=3, cursor="hand2"
                      ).pack(side="left", padx=(8, 0))

        tk.Frame(src_inner, width=10, bg=CARD_BG).pack(side="left")
        _folder_var = tk.StringVar(value=self.out_folder_var.get())
        path_entry = tk.Entry(src_inner, textvariable=_folder_var,
                              bg=PANEL_BG, fg=TEXT_PRI, insertbackground=TEXT_PRI,
                              relief="flat", bd=4, font=("Courier New", 8))
        path_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))

        def _browse():
            p = filedialog.askdirectory(title="Select folder to analyse")
            if p: _folder_var.set(p)
        tk.Button(src_inner, text="Browse", command=_browse,
                  bg=PANEL_BG, fg=TEXT_SEC, font=("Courier New", 8),
                  relief="flat", padx=8, pady=3, cursor="hand2").pack(side="left")

        # ── threshold controls ───────────────────────────────────────────
        thr_frame = tk.Frame(win, bg=DARK_BG)
        thr_frame.pack(fill="x", padx=20, pady=(0, 4))
        _min_dur   = tk.DoubleVar(value=1.0)
        _max_dur   = tk.DoubleVar(value=30.0)
        _clip_thr  = tk.DoubleVar(value=0.99)
        _noise_thr = tk.DoubleVar(value=0.015)

        for col, (lbl, var) in enumerate([
            ("Min dur (s)", _min_dur), ("Max dur (s)", _max_dur),
            ("Clip level",  _clip_thr), ("Noise floor", _noise_thr),
        ]):
            tk.Label(thr_frame, text=lbl, font=("Courier New", 7),
                     bg=DARK_BG, fg=TEXT_MUT).grid(row=0, column=col*2, sticky="e", padx=(10, 2))
            tk.Entry(thr_frame, textvariable=var, width=6, bg=PANEL_BG, fg=TEXT_PRI,
                     insertbackground=TEXT_PRI, relief="flat", bd=3,
                     font=("Courier New", 8)).grid(row=0, column=col*2+1, sticky="w", padx=(0, 4))

        # ── results tree ─────────────────────────────────────────────────
        COLS = [
            ("file",     "File",       260, "w"),
            ("status",   "Status",      56, "center"),
            ("score",    "Score",       50, "center"),
            ("dur",      "Duration",    60, "center"),
            ("rms",      "RMS",         52, "center"),
            ("peak",     "Peak",        60, "center"),
            ("clipping", "Clipping",    62, "center"),
            ("snr",      "SNR",         52, "center"),
            ("muffled",  "Muffled",     60, "center"),
            ("dc",       "DC Offset",   68, "center"),
            ("issues",   "Issues",     220, "w"),
        ]
        col_ids = [c[0] for c in COLS]

        style = ttk.Style()
        style.configure("Analyzer.Treeview",
                        background=PANEL_BG, foreground=TEXT_PRI,
                        fieldbackground=PANEL_BG, rowheight=22,
                        font=("Courier New", 8))
        style.configure("Analyzer.Treeview.Heading",
                        background=CARD_BG, foreground=TEXT_SEC,
                        font=("Courier New", 8, "bold"), relief="flat")
        style.map("Analyzer.Treeview",
                  background=[("selected", ACCENT)],
                  foreground=[("selected", DARK_BG)])

        res_outer = tk.Frame(win, bg=DARK_BG)
        res_outer.pack(fill="both", expand=True, padx=20)
        res_outer.rowconfigure(0, weight=1)
        res_outer.columnconfigure(0, weight=1)

        tree = ttk.Treeview(res_outer, columns=col_ids, show="headings",
                            style="Analyzer.Treeview", selectmode="extended")
        vsb = ttk.Scrollbar(res_outer, orient="vertical",   command=tree.yview)
        hsb = ttk.Scrollbar(res_outer, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        tree.tag_configure("ok",   foreground=ACCENT2)
        tree.tag_configure("warn", foreground=WARNING)
        tree.tag_configure("err",  foreground=DANGER)

        # ── sort state ───────────────────────────────────────────────────
        _sort_state = {}  # col -> bool (ascending/descending)

        def _sort_tree(col, initial=False):
            items = [(tree.set(k, col), k) for k in tree.get_children("")]
            rev = _sort_state.get(col, True) if not initial else True  # score: desc default
            try:
                items.sort(key=lambda x: float(x[0].replace("%","").replace("dB","").replace("s","")), reverse=rev)
            except ValueError:
                items.sort(key=lambda x: x[0], reverse=rev)
            for i, (_, k) in enumerate(items):
                tree.move(k, "", i)
            _sort_state[col] = not rev
            # mark active sort column in heading
            for cid, hdr_txt, *_ in COLS:
                arrow = (" \u25bc" if rev else " \u25b2") if cid == col else ""
                tree.heading(cid, text=hdr_txt + arrow)

        for cid, hdr_txt, w, anch in COLS:
            tree.heading(cid, text=hdr_txt, command=lambda c=cid: _sort_tree(c))
            tree.column(cid, width=w, anchor=anch, minwidth=40, stretch=(cid == "issues"))

        # ── playback helpers ─────────────────────────────────────────────
        _play_lbl = tk.StringVar(value="No selection")
        _current_play_path = [None]

        def _get_selected_path():
            sel = tree.selection()
            if not sel:
                return None
            fname = tree.set(sel[0], "file")
            folder = _folder_var.get()
            path = os.path.join(folder, fname)
            return path if os.path.isfile(path) else None

        def _play_selected(*_):
            path = _get_selected_path()
            if not path:
                return
            _current_play_path[0] = path
            _play_lbl.set(f"Playing: {os.path.basename(path)}")
            if not self._pygame_ok:
                _play_lbl.set("pip install pygame to enable playback")
                return
            def _do_play():
                try:
                    import pygame
                    self._stop_playback()
                    pygame.mixer.music.load(path)
                    pygame.mixer.music.play()
                    self._playing = True
                    self._play_btn.config(text="▶  Playing…")
                    while pygame.mixer.music.get_busy():
                        import time; time.sleep(0.05)
                except Exception as e:
                    _play_lbl.set(f"Playback error: {e}")
                finally:
                    self._playing = False
                    try: self._play_btn.config(text="▶  Play")
                    except Exception: pass
            import threading
            threading.Thread(target=_do_play, daemon=True).start()

        tree.bind("<Double-Button-1>", _play_selected)

        # ── filter ───────────────────────────────────────────────────────
        # All items are always in the tree; filter just adjusts row visibility
        # by detaching/reattaching items from the tree widget.
        _all_items = []   # list of (iid,) in insertion order
        _filter_var = tk.StringVar(value="All Files")
        FILTER_OPTIONS = ["All Files", "Warnings & Issues", "Issues Only"]

        def _apply_filter(*_):
            fv = _filter_var.get()
            for iid in tree.get_children(""):
                tree.detach(iid)
            for iid in _all_items:
                tags = tree.item(iid, "tags")
                show = True
                if fv == "Issues Only":
                    show = "err" in tags
                elif fv == "Warnings & Issues":
                    show = "err" in tags or "warn" in tags
                if show:
                    tree.reattach(iid, "", "end")
            # re-sort after filter
            _sort_tree("score", initial=True)

        # ── bottom bar ───────────────────────────────────────────────────
        bot = tk.Frame(win, bg=DARK_BG)
        bot.pack(fill="x", padx=20, pady=(6, 8))

        _status_lbl = tk.Label(bot, text="Ready — select a folder and click Analyse",
                               font=("Courier New", 8), bg=DARK_BG, fg=TEXT_MUT)
        _status_lbl.pack(side="left")
        _progress = ttk.Progressbar(bot, mode="determinate", style="TProgressbar", length=140)
        _progress.pack(side="left", padx=(10, 0))

        # ── action buttons (right-aligned, added right-to-left) ───────────
        tk.Button(bot, text="Clear", font=("Courier New", 8),
                  bg=PANEL_BG, fg=TEXT_MUT, relief="flat", padx=10, pady=4,
                  cursor="hand2",
                  command=lambda: [_cache.clear(), _all_items.clear(),
                                   tree.delete(*tree.get_children())]
                  ).pack(side="right", padx=(4, 0))

        def _export_report():
            path = filedialog.asksaveasfilename(
                title="Save analysis report", defaultextension=".csv",
                filetypes=[("CSV","*.csv"),("All","*.*")])
            if not path: return
            try:
                import csv
                with open(path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(col_ids)
                    for iid in _all_items:
                        w.writerow([tree.set(iid, c) for c in col_ids])
                self._log(f"Analysis report saved: {os.path.basename(path)}", ACCENT2)
            except Exception as e:
                messagebox.showerror("Export failed", str(e))

        tk.Button(bot, text="Export CSV", font=("Courier New", 8),
                  bg=PANEL_BG, fg=TEXT_SEC, relief="flat", padx=10, pady=4,
                  cursor="hand2", command=_export_report
                  ).pack(side="right", padx=(4, 0))

        def _delete_selected():
            sel = tree.selection()
            if not sel:
                messagebox.showinfo("Nothing selected", "Select rows to delete."); return
            paths = []
            for iid in sel:
                fname = tree.set(iid, "file")
                p = os.path.join(_folder_var.get(), fname)
                if os.path.isfile(p):
                    paths.append((iid, p))
            if not paths: return
            if not messagebox.askyesno("Delete",
                    f"Permanently delete {len(paths)} file(s)?"):
                return
            deleted = failed = 0
            for iid, path in paths:
                try:
                    os.remove(path)
                    _cache.pop(path, None)
                    if iid in _all_items: _all_items.remove(iid)
                    tree.delete(iid)
                    deleted += 1
                except Exception as e:
                    self._log(f"  Delete failed: {os.path.basename(path)}: {e}", DANGER)
                    failed += 1
            _status_lbl.config(
                text=f"Deleted {deleted} file(s){f', {failed} failed' if failed else ''}.",
                fg=ACCENT2 if not failed else WARNING)
            self._refresh_output_list()

        tk.Button(bot, text="Delete Selected", font=("Courier New", 8),
                  bg="#2a1218", fg=DANGER, relief="flat", padx=10, pady=4,
                  cursor="hand2", command=_delete_selected
                  ).pack(side="right", padx=(4, 0))

        tk.Button(bot, text="Play Selected", font=("Courier New", 8),
                  bg=PANEL_BG, fg=ACCENT2, relief="flat", padx=10, pady=4,
                  cursor="hand2", command=_play_selected
                  ).pack(side="right", padx=(4, 0))

        # ── filter dropdown (right of Analyse button) ─────────────────────
        filter_cb = ttk.Combobox(bot, textvariable=_filter_var, values=FILTER_OPTIONS,
                                 state="readonly", font=("Courier New", 8), width=18)
        filter_cb.pack(side="right", padx=(4, 0))
        tk.Label(bot, text="Show:", font=("Courier New", 8),
                 bg=DARK_BG, fg=TEXT_MUT).pack(side="right")
        _filter_var.trace_add("write", _apply_filter)

        tk.Button(bot, text="Analyse", font=("Courier New", 9, "bold"),
                  bg=ACCENT, fg=DARK_BG, relief="flat", padx=14, pady=4,
                  cursor="hand2",
                  command=lambda: threading.Thread(
                      target=self._run_audio_analysis,
                      args=(_folder_var.get(), tree, _status_lbl, _progress,
                            _min_dur.get(), _max_dur.get(),
                            _clip_thr.get(), _noise_thr.get(),
                            _cache, _all_items, _apply_filter),
                      daemon=True).start()
                  ).pack(side="right", padx=(8, 0))

        # summary bar
        _summary_var = tk.StringVar(value="")
        tk.Label(win, textvariable=_summary_var, font=("Courier New", 8),
                 bg=DARK_BG, fg=TEXT_SEC, anchor="w").pack(fill="x", padx=20, pady=(0, 4))

        # store summary ref so _run_audio_analysis can update it
        win._summary_var = _summary_var

    def _run_audio_analysis(self, folder, tree, status_lbl, progress,
                            min_dur, max_dur, clip_thr, noise_thr,
                            cache, all_items, apply_filter_cb=None):
        """
        Analyse every WAV in folder, score it 0-100, and populate the treeview.

        Quality score deductions:
          Duration too short (<min_dur)  -40 / -15
          Near-silent (RMS<0.01)         -50
          Very low level                 -20
          Clipping (peak>=clip_thr)      -30
          Flat-top clipping              -10
          DC offset                      -15 / -5
          SNR < 10 dB                    -25 / -10
          Muffled (<2kHz dominates)      -30 / -15
          Codec artefacts (HF spikes)    -10
          Noisy tail                     -10

        Final status: 75+ = OK, 45+ = Warning, <45 = Error.
        """
        import wave as _wave
        try:
            import numpy as np
            from numpy.fft import rfft, rfftfreq
        except ImportError:
            self.after(0, lambda: status_lbl.config(
                text="numpy required — pip install numpy", fg=DANGER))
            return

        wavs = sorted(glob.glob(os.path.join(folder, "*.wav")))
        if not wavs:
            self.after(0, lambda: status_lbl.config(
                text=f"No WAV files found in '{folder}'", fg=WARNING))
            return

        # Clear previous
        self.after(0, lambda: [tree.delete(i) for i in tree.get_children()])
        self.after(0, lambda: progress.configure(maximum=len(wavs), value=0))
        self.after(0, lambda: status_lbl.config(
            text=f"Analysing {len(wavs)} file(s)…", fg=WARNING))
        cache.clear()
        all_items.clear()

        counts = {"ok": 0, "warn": 0, "err": 0}

        for i, wav_path in enumerate(wavs):
            fname = os.path.basename(wav_path)
            score = 100
            issues = []

            try:
                with _wave.open(wav_path, "rb") as wf:
                    sr      = wf.getframerate()
                    nch     = wf.getnchannels()
                    sw      = wf.getsampwidth()
                    nframes = wf.getnframes()
                    raw     = wf.readframes(nframes)

                dtype   = np.int16 if sw == 2 else np.int32
                samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)
                if nch > 1:
                    samples = samples.reshape(-1, nch).mean(axis=1)
                scale   = float(2 ** (sw * 8 - 1))
                samples /= scale
                dur     = len(samples) / sr
                peak    = float(np.abs(samples).max())
                rms     = float(np.sqrt(np.mean(samples ** 2)))
                peak_dbfs = 20 * np.log10(peak + 1e-9)
                clip_pct  = float(np.mean(np.abs(samples) >= clip_thr) * 100)

                # 1. Duration
                if dur < min_dur:
                    issues.append(f"too short ({dur:.2f}s)")
                    score -= 40
                elif dur > max_dur:
                    issues.append(f"too long ({dur:.0f}s)")
                    score -= 15

                # 2. RMS / silence
                if rms < 0.01:
                    issues.append(f"near-silent (RMS={rms:.4f})")
                    score -= 50
                elif rms < 0.02:
                    issues.append("very low level")
                    score -= 20

                # 3. Peak clipping
                if clip_pct > 1.0:
                    issues.append(f"heavy clipping ({clip_pct:.1f}%)")
                    score -= 30
                elif clip_pct > 0.1:
                    issues.append(f"some clipping ({clip_pct:.2f}%)")
                    score -= 10

                # 4. Flat-top clipping
                max_val   = np.abs(samples).max()
                flat_runs = int(np.sum(np.abs(samples) >= max_val * 0.999))
                if flat_runs > 10 and clip_pct < 0.1:
                    issues.append(f"flat-top clips ({flat_runs} smp)")
                    score -= 10

                # 5. DC offset
                dc = float(np.mean(samples))
                if abs(dc) > 0.05:
                    issues.append(f"DC offset ({dc:+.3f})")
                    score -= 15
                elif abs(dc) > 0.02:
                    issues.append(f"mild DC ({dc:+.3f})")
                    score -= 5

                # 6. SNR (signal vs head of file noise floor)
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

                # 7. Muffling — spectral energy above 2 kHz
                chunk    = min(8192, len(samples))
                F        = np.abs(rfft(samples[:chunk] * np.hanning(chunk)))
                freqs    = rfftfreq(chunk, 1 / sr)
                hi       = float(np.sum(F[freqs >= 2000] ** 2))
                tot_spec = float(np.sum(F ** 2)) + 1e-9
                hi_ratio = hi / tot_spec
                muffled_flag = hi_ratio < 0.08
                if hi_ratio < 0.04:
                    issues.append(f"severely muffled ({hi_ratio*100:.0f}%>2kHz)")
                    score -= 30
                elif hi_ratio < 0.08:
                    issues.append(f"muffled ({hi_ratio*100:.0f}%>2kHz)")
                    score -= 15

                # 8. Codec artefacts (HF spikes)
                hf       = float(np.sum(F[freqs >= 10000] ** 2))
                hf_ratio = hf / (tot_spec + 1e-9)
                if hf_ratio > 0.15 and hi_ratio > 0.05:
                    issues.append(f"codec artefacts? (HF {hf_ratio:.0%})")
                    score -= 10

                # 9. Background noise (tail RMS)
                tail_len = min(int(sr * 0.3), len(samples) // 4)
                tail_rms = float(np.sqrt(np.mean(samples[-tail_len:] ** 2)))
                if tail_rms >= noise_thr:
                    issues.append(f"noisy tail (RMS={tail_rms:.3f})")
                    score -= 10

                score = max(0, score)
                if score >= 75:
                    status_icon = "OK"
                    tag = "ok"
                elif score >= 45:
                    status_icon = "WARN"
                    tag = "warn"
                else:
                    status_icon = "ERR"
                    tag = "err"

                issue_str = " · ".join(issues) if issues else "Clean"
                row = (
                    fname,
                    {"ok": "OK", "warn": "WARN", "err": "ERR"}[tag],
                    str(score),
                    f"{dur:.1f}s",
                    f"{rms*100:.1f}%",
                    f"{peak_dbfs:.1f}dB",
                    f"{clip_pct:.2f}%",
                    f"{snr:.0f}dB" if noise_len > 64 else "—",
                    "Yes" if muffled_flag else "No",
                    f"{dc:+.3f}" if abs(dc) > 0.001 else "OK",
                    issue_str,
                )

            except Exception as e:
                tag = "err"
                score = 0
                issues = [str(e)]
                row = (fname, "ERR", "0", "?", "?", "?", "?", "?", "?", "?",
                       f"Read error: {e}")

            cache[wav_path] = {"path": wav_path, "score": score, "tag": tag}
            counts[tag] += 1

            def _insert(r=row, t=tag):
                iid = tree.insert("", "end", values=r, tags=(t,))
                all_items.append(iid)

            self.after(0, _insert)
            self.after(0, lambda v=i+1: progress.configure(value=v))

        # After all inserts, sort by Score descending by default
        total_files = len(wavs)

        def _finalize():
            # Sort by score descending (initial default)
            items = [(tree.set(k, "score"), k) for k in tree.get_children("")]
            try:
                items.sort(key=lambda x: float(x[0]), reverse=True)
            except ValueError:
                pass
            for idx2, (_, k) in enumerate(items):
                tree.move(k, "", idx2)
            # Mark heading
            for cid, hdr_txt, *_ in [
                ("file","File"),("status","Status"),("score","Score"),
                ("dur","Duration"),("rms","RMS"),("peak","Peak"),
                ("clipping","Clipping"),("snr","SNR"),("muffled","Muffled"),
                ("dc","DC Offset"),("issues","Issues"),
            ]:
                tree.heading(cid, text=hdr_txt + (" \u25bc" if cid == "score" else ""))

            summary = (f"  {total_files} files  —  "
                       f"OK {counts['ok']}  "
                       f"WARN {counts['warn']}  "
                       f"ERR {counts['err']}  "
                       f"  (click column headers to sort)")
            status_lbl.config(
                text=f"Done — {total_files} file(s) analysed.",
                fg=ACCENT2 if counts["err"] == 0 and counts["warn"] == 0
                   else (WARNING if counts["err"] == 0 else DANGER))
            try:
                win = status_lbl.winfo_toplevel()
                if hasattr(win, "_summary_var"):
                    win._summary_var.set(summary)
            except Exception:
                pass

        self.after(0, _finalize)



    def _on_close(self):
        self._stop_playback()
        self._save_config()
        # cleanly shut down any running workers
        self._xtts_worker.stop()
        self._qwen_worker.stop()
        self._rvc_worker.stop()
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = VoiceClonerApp()
    app.mainloop()