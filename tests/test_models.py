import pytest
from core.models import GenerationJob, EngineParameters, RvcParameters, GenerationContext


class TestCoreModels:
    def test_generation_job_immutability(self):
        job = GenerationJob(index=1, text="Hello world", custom_filename="greet.wav", job_mood="Warm")
        assert job.index == 1
        assert job.text == "Hello world"
        assert job.custom_filename == "greet.wav"
        assert job.job_mood == "Warm"

        # Verify frozen immutability
        with pytest.raises(Exception):
            job.text = "Changed text"

    def test_engine_parameters_defaults(self):
        params = EngineParameters(backend="xtts")
        assert params.backend == "xtts"
        assert params.language == "en"
        assert params.speed == 1.0
        assert params.temperature == 0.75

    def test_generation_context_structure(self):
        job = GenerationJob(index=0, text="Test sentence")
        params = EngineParameters(backend="qwen", speed=1.0)
        rvc = RvcParameters(enabled=False)

        ctx = GenerationContext(
            output_dir="Output",
            batches=[[job]],
            engine_params=params,
            rvc_params=rvc,
            skyrim_mode=False
        )

        assert ctx.output_dir == "Output"
        assert len(ctx.batches) == 1
        assert len(ctx.batches[0]) == 1
        assert ctx.batches[0][0].text == "Test sentence"
        assert ctx.engine_params.backend == "qwen"
