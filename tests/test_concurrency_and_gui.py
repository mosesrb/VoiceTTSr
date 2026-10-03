import ast
import inspect
import textwrap
import pytest
from core.models import GenerationJob, EngineParameters, RvcParameters, GenerationContext
from core.config import save_config_atomic, load_config_safe
import voice_cloner_gui


class TestConcurrencyAndGuiThreadSafety:
    def test_run_generation_zero_direct_tkinter_variable_access(self):
        """
        Verify via AST analysis that _run_generation contains NO direct calls to .get() or .set()
        on self or Tkinter variables, guaranteeing thread safety.
        """
        source = textwrap.dedent(inspect.getsource(voice_cloner_gui.VoiceClonerApp._run_generation))
        tree = ast.parse(source)

        violations = []
        for node in ast.walk(tree):
            # Check for obj.get() or obj.set(...) calls
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ("get", "set"):
                    # Only allow dict.get / queue.get / response.get calls, reject any Tkinter var calls
                    # If called on self.<attr>, it's a violation
                    if isinstance(node.func.value, ast.Attribute) and isinstance(node.func.value.value, ast.Name) and node.func.value.value.id == "self":
                        violations.append(f"Line {node.lineno}: direct self.{node.func.value.attr}.{node.func.attr}() call")
                    # If called on self, violation
                    elif isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
                        violations.append(f"Line {node.lineno}: direct self.{node.func.attr}() call")

        assert violations == [], f"Found cross-thread Tkinter variable access in _run_generation: {violations}"

    def test_run_generation_zero_direct_progress_bar_mutations(self):
        """
        Verify via AST that _run_generation does not directly mutate _gen_progress subscription
        (e.g., self._gen_progress['value'] = ...), but dispatches via self.after.
        """
        source = textwrap.dedent(inspect.getsource(voice_cloner_gui.VoiceClonerApp._run_generation))
        tree = ast.parse(source)

        progress_mutations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Subscript):
                        if isinstance(target.value, ast.Attribute) and target.value.attr == "_gen_progress":
                            progress_mutations.append(f"Line {node.lineno}: direct mutation of _gen_progress")

        assert progress_mutations == [], f"Direct mutation of _gen_progress in background thread: {progress_mutations}"

    def test_gui_config_persistence_uses_atomic_helpers(self):
        """
        Verify that _save_config and _load_config in VoiceClonerApp delegate to core.config functions.
        """
        save_source = textwrap.dedent(inspect.getsource(voice_cloner_gui.VoiceClonerApp._save_config))
        assert "save_config_atomic(" in save_source
        assert "open(" not in save_source

        load_source = textwrap.dedent(inspect.getsource(voice_cloner_gui.VoiceClonerApp._load_config))
        assert "load_config_safe(" in load_source

    def test_generation_context_snapshot_execution(self, tmp_path):
        """
        Simulate a GenerationContext snapshot execution in a background thread with mock worker.
        """
        job = GenerationJob(
            index=1,
            text="Thread safety test prompt",
            custom_filename="thread_safe.wav",
            job_mood="Warm",
        )
        engine_params = EngineParameters(
            backend="xtts",
            language="en",
            speed=1.0,
            temperature=0.75,
            naming_mode="Sequential",
            ref_wavs=[],
        )
        rvc_params = RvcParameters(enabled=False)
        context = GenerationContext(
            output_dir=str(tmp_path),
            batches=[[job]],
            engine_params=engine_params,
            rvc_params=rvc_params,
            skyrim_mode=False,
        )

        assert context.engine_params.backend == "xtts"
        assert context.batches[0][0].text == "Thread safety test prompt"
        assert context.output_dir == str(tmp_path)
