import pytest
from core.models import GenerationJob, EngineParameters, RvcParameters, GenerationContext
from core import (
    GenerationJob as JobPkg,
    EngineParameters as EnginePkg,
    RvcParameters as RvcPkg,
    GenerationContext as CtxPkg,
)


class TestCoreModels:
    def test_package_exports(self):
        assert GenerationJob is JobPkg
        assert EngineParameters is EnginePkg
        assert RvcParameters is RvcPkg
        assert GenerationContext is CtxPkg

    def test_generation_job_immutability(self):
        dummy_btn = object()
        dummy_lbl = object()
        job = GenerationJob(
            index=1,
            text="Hello world",
            custom_filename="greet.wav",
            job_mood="Warm",
            status_label_id=dummy_lbl,
            play_button_id=dummy_btn,
        )
        assert job.index == 1
        assert job.text == "Hello world"
        assert job.custom_filename == "greet.wav"
        assert job.job_mood == "Warm"
        assert job.status_label_id is dummy_lbl
        assert job.play_button_id is dummy_btn

        # Verify frozen immutability
        with pytest.raises(Exception):
            job.text = "Changed text"

    def test_engine_parameters_defaults_and_immutability(self):
        params = EngineParameters(backend="xtts")
        assert params.backend == "xtts"
        assert params.language == "en"
        assert params.speed == 1.0
        assert params.temperature == 0.75
        assert params.repetition_penalty == 2.0
        assert params.top_k == 50
        assert params.top_p == 0.85
        assert params.exaggeration == 0.5
        assert params.cfg_weight == 0.5
        assert params.max_steps == 40
        assert params.global_preset == "Natural"
        assert params.use_icl is False
        assert params.profile_path is None
        assert params.ref_wavs == []
        assert params.xtts_audio_pro is False
        assert params.naming_mode == "Standard"
        assert params.stream is False
        assert params.retry_mumble is False
        assert params.emotion_tags is False

        with pytest.raises(Exception):
            params.speed = 1.5

    def test_rvc_parameters_immutability(self):
        rvc = RvcParameters(
            enabled=True,
            model_path="models/voice.pth",
            pitch=2,
            auto_rvc=True,
            auto_scope="per-job",
        )
        assert rvc.enabled is True
        assert rvc.pitch == 2
        assert rvc.auto_scope == "per-job"

        with pytest.raises(Exception):
            rvc.pitch = 0

    def test_generation_context_structure(self):
        job = GenerationJob(index=0, text="Test sentence")
        params = EngineParameters(
            backend="qwen",
            speed=1.0,
            naming_mode="Sequential",
            stream=True,
            retry_mumble=True,
            emotion_tags=True,
        )
        rvc = RvcParameters(enabled=False)

        ctx = GenerationContext(
            output_dir="Output",
            batches=[[job]],
            engine_params=params,
            rvc_params=rvc,
            skyrim_mode=True,
            skyrim_paths={"plugin": "Mod.esp", "voice_type": "MaleNord"},
        )

        assert ctx.output_dir == "Output"
        assert len(ctx.batches) == 1
        assert len(ctx.batches[0]) == 1
        assert ctx.batches[0][0].text == "Test sentence"
        assert ctx.engine_params.backend == "qwen"
        assert ctx.engine_params.naming_mode == "Sequential"
        assert ctx.engine_params.stream is True
        assert ctx.skyrim_mode is True
        assert ctx.skyrim_paths["plugin"] == "Mod.esp"

        with pytest.raises(Exception):
            ctx.output_dir = "AnotherDir"
