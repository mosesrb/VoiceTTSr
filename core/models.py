"""
VoiceTTSr Core Data Models
Immutable dataclasses representing generation jobs, engine parameters, and batch configurations.
Enforces thread-safe state snapshots between GUI and worker execution threads.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass(frozen=True)
class GenerationJob:
    """A single TTS synthesis job."""
    index: int
    text: str
    custom_filename: str = ""
    job_mood: str = ""
    status_label_id: Optional[Any] = None


@dataclass(frozen=True)
class RvcParameters:
    """Parameters for RVC v2 voice conversion."""
    enabled: bool = False
    model_path: str = ""
    index_path: str = ""
    pitch: int = 0
    f0_method: str = "rmvpe"
    index_rate: float = 0.75
    filter_radius: int = 3
    resample_sr: int = 0
    rms_mix_rate: float = 0.25
    protect: float = 0.33
    auto_rvc: bool = False
    auto_scope: str = "global"


@dataclass(frozen=True)
class EngineParameters:
    """Parameters passed to the underlying TTS worker backend."""
    backend: str                        # 'xtts', 'qwen', 'chatterbox'
    language: str = "en"
    speed: float = 1.0
    temperature: float = 0.75
    repetition_penalty: float = 2.0
    top_k: int = 50
    top_p: float = 0.85
    exaggeration: float = 0.5          # Chatterbox specific
    cfg_weight: float = 0.5            # Chatterbox specific
    max_steps: int = 40                # Chatterbox diffusion steps
    global_preset: str = "Natural"
    use_icl: bool = False              # Qwen Voice Design
    profile_path: Optional[str] = None
    ref_wavs: List[str] = field(default_factory=list)
    xtts_audio_pro: bool = False


@dataclass(frozen=True)
class GenerationContext:
    """Full snapshot of a generation run captured on the main thread."""
    output_dir: str
    batches: List[List[GenerationJob]]
    engine_params: EngineParameters
    rvc_params: RvcParameters
    skyrim_mode: bool = False
    skyrim_paths: Dict[str, str] = field(default_factory=dict)
