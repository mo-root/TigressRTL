"""Unit tests for the pure logic in src/ and test/.

Runs in well under a second with no Ollama, no pulled model, no Icarus/Slang,
and no verilog-eval clone. That is the point: every other way of exercising
this project needs a GPU and a multi-hour sweep, so nothing here could be
checked by a reviewer who does not already have the full setup.

Scope is deliberately the logic that is deterministic and reachable without a
toolchain: path sandboxing, the edit contract, diagnostic truncation, config
validation, fenced-tool-call recovery, backend availability, and the scoring
decision in run_validation.py. Anything that shells out to a compiler or talks
to a model belongs in the integration harnesses, not here.

    python -m unittest discover -s test -p 'test_*.py'
"""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import config  # noqa: E402
import model  # noqa: E402
import tools  # noqa: E402


def _load_script(name: str):
    """Import a test/*.py harness by path; they are scripts, not a package."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "test" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_validation = _load_script("run_validation")


class SandboxedPaths(unittest.TestCase):
    """_resolve_safe_path is the only thing standing between a model-supplied
    path string and the rest of the filesystem."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = tools.GENERATED_DIR
        tools.GENERATED_DIR = self._tmp.name

    def tearDown(self):
        tools.GENERATED_DIR = self._saved
        self._tmp.cleanup()

    def test_plain_name_resolves_inside_the_sandbox(self):
        resolved = tools._resolve_safe_path("TopModule.sv")
        self.assertEqual(resolved.parent, Path(self._tmp.name).resolve())

    def test_subdirectories_are_allowed(self):
        resolved = tools._resolve_safe_path("pkg/types.svh")
        self.assertTrue(resolved.is_relative_to(Path(self._tmp.name).resolve()))

    def test_parent_traversal_is_rejected(self):
        for escape in ("../outside.sv", "../../etc/passwd", "a/../../b.sv"):
            with self.subTest(path=escape):
                with self.assertRaises(ValueError):
                    tools._resolve_safe_path(escape)

    def test_absolute_path_is_pulled_back_into_the_sandbox(self):
        # Models very predictably pass "/generated/foo.sv"; joining that with
        # pathlib would otherwise discard the base entirely and resolve to the
        # filesystem root.
        resolved = tools._resolve_safe_path("/generated/foo.sv")
        self.assertTrue(resolved.is_relative_to(Path(self._tmp.name).resolve()))
        self.assertEqual(resolved.name, "foo.sv")

    def test_leading_generated_segment_is_not_nested_twice(self):
        resolved = tools._resolve_safe_path("generated/foo.sv")
        self.assertEqual(resolved.parent, Path(self._tmp.name).resolve())


class EditFileBlockContract(unittest.TestCase):
    """edit_file_block must replace exactly one occurrence, and say so clearly
    when it cannot -- a silent no-op or a multi-site replace both leave the
    model believing an edit happened."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._saved = tools.GENERATED_DIR
        tools.GENERATED_DIR = self._tmp.name

    def tearDown(self):
        tools.GENERATED_DIR = self._saved
        self._tmp.cleanup()

    def _write(self, body: str) -> None:
        Path(self._tmp.name, "m.sv").write_text(body, encoding="utf-8")

    def _body(self) -> str:
        return Path(self._tmp.name, "m.sv").read_text(encoding="utf-8")

    def test_exactly_one_match_is_replaced(self):
        self._write("assign a = 1'b0;\n")
        result = tools.edit_file_block.invoke(
            {"file_path": "m.sv", "target_string": "1'b0", "replacement_string": "1'b1"}
        )
        self.assertIn("1'b1", self._body())
        self.assertNotIn("Error", result)

    def test_zero_matches_changes_nothing(self):
        self._write("assign a = 1'b0;\n")
        before = self._body()
        tools.edit_file_block.invoke(
            {"file_path": "m.sv", "target_string": "nonexistent", "replacement_string": "x"}
        )
        self.assertEqual(self._body(), before)

    def test_multiple_matches_changes_nothing(self):
        self._write("assign a = 1'b0;\nassign b = 1'b0;\n")
        before = self._body()
        tools.edit_file_block.invoke(
            {"file_path": "m.sv", "target_string": "1'b0", "replacement_string": "1'b1"}
        )
        self.assertEqual(self._body(), before)

    def test_missing_file_does_not_raise(self):
        result = tools.edit_file_block.invoke(
            {"file_path": "absent.sv", "target_string": "a", "replacement_string": "b"}
        )
        self.assertIsInstance(result, str)


class DiagnosticTruncation(unittest.TestCase):
    """Truncation decides what the model gets to read after a failed build, so
    dropping the errors defeats the whole fix cycle."""

    ERROR = "a.sv:9: error: real problem\n"
    NOTE = "a.sv:{}: sorry: constant selects in always_* processes\n"

    def test_under_budget_is_returned_untouched(self):
        log = self.NOTE.format(1) * 3
        self.assertEqual(tools._truncate_diagnostics(log), log)

    def test_over_budget_is_truncated(self):
        log = self.NOTE.format(1) * (tools.MAX_DIAG_BLOCKS + 4)
        self.assertLess(len(tools._truncate_diagnostics(log)), len(log))

    def test_errors_survive_a_flood_of_notes(self):
        # The regression this guards: "sorry:" notes are emitted while
        # elaborating, i.e. before an error raised by a later process, so a
        # keep-the-first-N policy silently drops every error.
        log = "".join(self.NOTE.format(i) for i in range(12)) + self.ERROR
        self.assertIn("error: real problem", tools._truncate_diagnostics(log))

    def test_kept_blocks_stay_in_source_order(self):
        log = self.NOTE.format(1) + self.ERROR + self.NOTE.format(2)
        log += "".join(self.NOTE.format(i) for i in range(3, 12))
        out = tools._truncate_diagnostics(log)
        self.assertLess(out.index("sorry"), out.index("error: real problem"))

    def test_omitted_counts_are_reported(self):
        log = "".join(self.NOTE.format(i) for i in range(12)) + self.ERROR
        self.assertIn("omitted", tools._truncate_diagnostics(log))

    def test_sorry_notes_are_recognised_as_diagnostics(self):
        self.assertTrue(tools._DIAG_START_RE.search(self.NOTE.format(4)))

    def test_a_sorry_note_is_not_counted_as_an_error(self):
        self.assertIsNone(tools._DIAG_IS_ERROR_RE.match(self.NOTE.format(4)))


class BackendAvailability(unittest.TestCase):
    """A missing backend binary is invisible to the agent's own failure
    detection, so the gate has to be right."""

    def setUp(self):
        self._saved = dict(tools.BUILD_TOOL_BINARIES)

    def tearDown(self):
        tools.BUILD_TOOL_BINARIES.clear()
        tools.BUILD_TOOL_BINARIES.update(self._saved)

    def test_absent_binary_is_reported(self):
        tools.BUILD_TOOL_BINARIES["icarus"] = "definitely-not-a-real-binary-xyz"
        self.assertIsNotNone(tools.missing_build_backend("icarus"))

    def test_a_directory_is_not_a_runnable_binary(self):
        # os.path.exists() would pass a directory named "slang" in the cwd.
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "slang"))
            tools.BUILD_TOOL_BINARIES["slang"] = os.path.join(tmp, "slang")
            self.assertIsNotNone(tools.missing_build_backend("slang"))

    def test_a_non_executable_file_is_not_runnable(self):
        with tempfile.NamedTemporaryFile(suffix="-slang", delete=False) as fh:
            fh.write(b"#!/bin/sh\n")
            path = fh.name
        try:
            os.chmod(path, 0o644)
            tools.BUILD_TOOL_BINARIES["slang"] = path
            self.assertIsNotNone(tools.missing_build_backend("slang"))
            os.chmod(path, 0o755)
            self.assertIsNone(tools.missing_build_backend("slang"))
        finally:
            os.unlink(path)


class ConfigValidation(unittest.TestCase):
    """load_config is the only place a typo'd experiment config can be caught
    before a multi-hour sweep runs with the wrong settings."""

    def _yaml(self, text: str) -> str:
        fh = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        fh.write(text)
        fh.close()
        self.addCleanup(os.unlink, fh.name)
        return fh.name

    def test_none_path_gives_dataclass_defaults(self):
        self.assertEqual(config.load_config(None), config.AgentConfig())

    def test_unknown_key_is_rejected(self):
        with self.assertRaises(ValueError):
            config.load_config(self._yaml("num_ctxx: 4096\n"))

    def test_invalid_build_tool_is_rejected(self):
        with self.assertRaises(ValueError):
            config.load_config(self._yaml("verilog_build_tool: islang\n"))

    def test_invalid_system_prompt_is_rejected(self):
        with self.assertRaises(ValueError):
            config.load_config(self._yaml("system_prompt: nonsense\n"))

    def test_omitted_field_falls_back_to_the_dataclass_not_the_shipped_yaml(self):
        # Documented behaviour worth pinning: configs/README.md tells users to
        # copy default.yaml and edit it, and deleting a line there reverts that
        # one field to the AgentConfig default rather than default.yaml's value.
        loaded = config.load_config(self._yaml("model: whatever\n"))
        self.assertEqual(loaded.num_ctx, config.AgentConfig().num_ctx)

    def test_every_shipped_config_loads(self):
        for path in sorted((REPO_ROOT / "configs").glob("*.yaml")):
            with self.subTest(config=path.name):
                config.load_config(path)


class SystemPromptStyles(unittest.TestCase):
    def test_both_styles_build_and_name_the_active_build_tool(self):
        for style in ("prose", "structured"):
            with self.subTest(style=style):
                prompt = model.build_system_prompt("lint_verilog", style)
                self.assertIn("lint_verilog", prompt)

    def test_default_style_is_the_prose_prompt(self):
        self.assertEqual(
            model.build_system_prompt("build_verilog"),
            model.build_system_prompt("build_verilog", "prose"),
        )


class FencedToolCallRecovery(unittest.TestCase):
    """Anything recovered here gets executed, so the negative cases matter at
    least as much as the positive ones."""

    VALID = {"write_file", "read_file", "edit_file_block", "list_directory", "build_verilog"}

    def _recover(self, text):
        return model.recover_tool_calls(text, self.VALID)

    def test_recovers_the_observed_shape(self):
        calls, _ = self._recover(
            'Sure.\n```json\n{"name": "write_file", "arguments": '
            '{"file_path": "TopModule.sv", "content": "module m; endmodule"}}\n```'
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "write_file")
        self.assertEqual(calls[0]["args"]["file_path"], "TopModule.sv")

    def test_recovers_an_untagged_fence(self):
        calls, _ = self._recover('```\n{"name":"build_verilog","arguments":{"file_path":"a.sv"}}\n```')
        self.assertEqual(len(calls), 1)

    def test_accepts_the_alternate_argument_spellings(self):
        for key in ("args", "parameters", "input"):
            with self.subTest(key=key):
                calls, _ = self._recover(
                    '```json\n{"name":"read_file","%s":{"file_path":"a.sv"}}\n```' % key
                )
                self.assertEqual(len(calls), 1)

    def test_unwraps_one_level(self):
        calls, _ = self._recover(
            '```json\n{"tool_call": {"name":"list_directory","arguments":{"path":"."}}}\n```'
        )
        self.assertEqual(len(calls), 1)

    def test_recovers_several_calls_from_one_reply(self):
        calls, _ = self._recover(
            '```json\n{"name":"write_file","arguments":{"file_path":"a.sv","content":"x"}}\n```\n'
            'then\n```json\n{"name":"build_verilog","arguments":{"file_path":"a.sv"}}\n```'
        )
        self.assertEqual(len(calls), 2)

    def test_consumed_fence_is_stripped_from_the_content(self):
        # Left in place it would sit in a transcript that never shrinks, as a
        # worked example of the format being recovered from.
        _, cleaned = self._recover(
            'Here.\n```json\n{"name":"write_file","arguments":{"file_path":"a.sv","content":"x"}}\n```\nDone.'
        )
        self.assertNotIn("write_file", cleaned)
        self.assertIn("Here.", cleaned)
        self.assertIn("Done.", cleaned)

    def test_ignores_unfenced_json(self):
        calls, _ = self._recover('Done. {"name": "write_file", "arguments": {"file_path": "a.sv"}}')
        self.assertEqual(calls, [])

    def test_ignores_a_status_object(self):
        calls, _ = self._recover('```json\n{"status":"success","files":1}\n```')
        self.assertEqual(calls, [])

    def test_ignores_a_tool_that_is_not_bound(self):
        calls, _ = self._recover(
            '```json\n{"name":"simulate_verilog","arguments":{"file_path":"a.sv"}}\n```'
        )
        self.assertEqual(calls, [])

    def test_ignores_a_non_mapping_arguments_value(self):
        calls, _ = self._recover('```json\n{"name":"write_file","arguments":"TopModule.sv"}\n```')
        self.assertEqual(calls, [])

    def test_ignores_malformed_json(self):
        calls, _ = self._recover('```json\n{"name": "write_file", oops}\n```')
        self.assertEqual(calls, [])

    def test_ignores_a_fenced_systemverilog_block(self):
        calls, _ = self._recover("```systemverilog\nmodule m; endmodule\n```")
        self.assertEqual(calls, [])

    def test_empty_input_is_safe(self):
        self.assertEqual(self._recover("")[0], [])
        self.assertEqual(self._recover(None)[0], [])


class ValidationScoring(unittest.TestCase):
    """The one defect class that turns a wrong answer into a passing one.

    These drive the real validate_problem() with the toolchain stubbed out, so
    they exercise the shipped decision rather than a copy of it -- an earlier
    version of this class reimplemented the pass/fail rule inline and kept
    passing when the rule itself was broken.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.dataset = root / "dataset"
        self.dataset.mkdir()
        (self.dataset / "P_ref.sv").write_text("module RefModule; endmodule\n")
        (self.dataset / "P_test.sv").write_text("module tb; endmodule\n")
        self.problem_dir = root / "P"
        (self.problem_dir / "generated").mkdir(parents=True)
        (self.problem_dir / "generated" / "TopModule.sv").write_text("module TopModule; endmodule\n")
        self._saved_run_step = run_validation.run_step

    def tearDown(self):
        run_validation.run_step = self._saved_run_step
        self._tmp.cleanup()

    def _score(self, sim_output: str) -> str:
        """Run the real validate_problem with compile and simulate stubbed."""
        calls = {"n": 0}

        self.commands = []

        def fake_run_step(cmd, cwd, timeout):
            calls["n"] += 1
            self.commands.append(cmd)
            # First call is the compile, second is the simulation.
            return (0, "" if calls["n"] == 1 else sim_output, False)

        run_validation.run_step = fake_run_step
        result, _log = run_validation.validate_problem("P", self.problem_dir, self.dataset, 5.0)
        return result["status"]

    def test_zero_samples_is_never_a_pass(self):
        self.assertEqual(self._score("Mismatches: 0 in 0 samples"), "unknown")

    def test_a_real_clean_run_still_passes(self):
        self.assertEqual(self._score("Mismatches: 0 in 20 samples"), "pass")

    def test_a_mismatching_run_fails(self):
        self.assertEqual(self._score("Mismatches: 10 in 20 samples"), "fail")

    def test_missing_summary_line_is_unknown(self):
        self.assertEqual(self._score("simulation produced no summary"), "unknown")

    def test_compile_uses_the_dataset_testbench_as_the_elaboration_root(self):
        # Without -s, iverilog elaborates every parentless module, including a
        # testbench the agent left in generated/, whose $finish then truncates
        # the run and scores the design on partial evidence.
        self._score("Mismatches: 0 in 20 samples")
        compile_cmd = self.commands[0]
        self.assertIn("-s", compile_cmd)
        self.assertEqual(
            compile_cmd[compile_cmd.index("-s") + 1], run_validation.DATASET_TB_MODULE
        )

    def test_unknown_is_a_counted_bucket_not_a_dropped_one(self):
        self.assertIn("unknown", run_validation.OTHER_STATUSES)

    def test_the_elaboration_root_is_pinned_to_the_dataset_testbench(self):
        # All 156 spec-to-rtl testbenches declare `module tb`; rooting there is
        # what stops a testbench the agent left behind from also running.
        self.assertEqual(run_validation.DATASET_TB_MODULE, "tb")


if __name__ == "__main__":
    unittest.main(verbosity=2)
