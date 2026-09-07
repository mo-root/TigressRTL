"""Unit tests for src/tools.py — the pure logic and the sandboxed file tools.

Nothing here shells out to iverilog/slang; build_verilog and lint_verilog
are covered only for the paths that return before subprocess.run (missing
file, sandbox escape), so the suite runs with neither tool installed.
"""

import pytest

import tools
from tools import (
    BUILD_FAILURE_PREFIXES,
    BUILD_TOOL_FUNCTIONS,
    MAX_DIAG_BLOCKS,
    _resolve_safe_path,
    _truncate_diagnostics,
    edit_file_block,
    list_directory,
    read_file,
    write_file,
)

BANNER = "slang version 6.0\nnote: resolving includes from ./inc\n"


def slang_diags(n, first=1):
    """n slang-style multi-line diagnostics (message + source line + caret)."""
    return "".join(
        f"top.sv:{i}:3: error: bad thing {i}\n   assign x = y;\n         ^\n"
        for i in range(first, first + n)
    )


# --------------------------------------------------------------------------
# _truncate_diagnostics
# --------------------------------------------------------------------------

def test_short_log_passes_through_byte_identical():
    log = BANNER + slang_diags(2)
    assert _truncate_diagnostics(log) == log


def test_log_exactly_at_cap_is_not_truncated():
    log = slang_diags(MAX_DIAG_BLOCKS)
    assert _truncate_diagnostics(log) == log


def test_truncation_keeps_first_max_blocks_and_drops_the_rest():
    out = _truncate_diagnostics(slang_diags(MAX_DIAG_BLOCKS + 3))
    assert out.count("error: bad thing") == MAX_DIAG_BLOCKS
    assert "bad thing 1" in out
    assert f"bad thing {MAX_DIAG_BLOCKS + 3}" not in out


def test_truncation_preserves_text_before_the_first_diagnostic():
    # Regression: the preamble falls outside every diagnostic block, so
    # rebuilding output from blocks alone silently dropped it — and only
    # when truncation triggered, which is exactly when context is scarcest.
    out = _truncate_diagnostics(BANNER + slang_diags(MAX_DIAG_BLOCKS + 2))
    assert out.startswith(BANNER)


def test_no_preamble_means_no_leading_junk():
    out = _truncate_diagnostics(slang_diags(MAX_DIAG_BLOCKS + 2))
    assert out.startswith("top.sv:1:3: error")


def test_omitted_counts_split_errors_from_warnings():
    log = slang_diags(MAX_DIAG_BLOCKS + 2) + "top.sv:99:1: warning: meh\n"
    out = _truncate_diagnostics(log)
    assert f"(2 more error(s), 1 more warning(s) omitted)" in out


def test_notes_fold_into_the_preceding_block_instead_of_consuming_a_slot():
    # A "note:" line is auxiliary info tied to the diagnostic above it, so
    # _DIAG_START_RE deliberately does not match it.
    log = "".join(
        f"top.sv:{i}:3: error: dup {i}\ntop.sv:{i}:1: note: previous definition here\n"
        for i in range(1, MAX_DIAG_BLOCKS + 3)
    )
    out = _truncate_diagnostics(log)
    assert out.count("error: dup") == MAX_DIAG_BLOCKS
    assert out.count("note: previous definition") == MAX_DIAG_BLOCKS


def test_icarus_single_line_format_is_handled():
    # Icarus emits a bare "syntax error" line followed by a specific
    # "error:" line at the same location — two matches for one real error,
    # which is why MAX_DIAG_BLOCKS is 6 rather than 3.
    log = "".join(f"top.sv:{i}: syntax error\ntop.sv:{i}: error: near ';'\n" for i in range(1, 6))
    out = _truncate_diagnostics(log)
    assert out.startswith("top.sv:1: syntax error")
    assert "omitted" in out


def test_empty_log_is_returned_unchanged():
    assert _truncate_diagnostics("") == ""


def test_log_with_no_diagnostics_at_all_is_untouched():
    prose = "Build succeeded: 0 errors, 0 warnings\n"
    assert _truncate_diagnostics(prose) == prose


# --------------------------------------------------------------------------
# _resolve_safe_path — the sandbox boundary
# --------------------------------------------------------------------------

def test_plain_relative_path_lands_inside_the_sandbox(sandbox):
    assert _resolve_safe_path("top.sv") == sandbox.resolve() / "top.sv"


def test_subdirectories_are_allowed(sandbox):
    assert _resolve_safe_path("pkg/types.svh") == sandbox.resolve() / "pkg" / "types.svh"


def test_leading_generated_segment_is_stripped_not_nested(sandbox):
    # Models routinely name the sandbox dir themselves; GENERATED_DIR
    # already *is* that directory, so it must not nest a second one.
    assert _resolve_safe_path("generated/top.sv") == sandbox.resolve() / "top.sv"
    assert _resolve_safe_path("/generated/top.sv") == sandbox.resolve() / "top.sv"


def test_absolute_path_is_rerooted_into_the_sandbox(sandbox):
    # The anchor is dropped rather than rejected, so an absolute path is
    # re-rooted instead of escaping or resolving to the filesystem root.
    assert _resolve_safe_path("/etc/passwd") == sandbox.resolve() / "etc" / "passwd"


@pytest.mark.parametrize("escape", ["../outside.sv", "../../etc/passwd", "pkg/../../outside.sv"])
def test_parent_traversal_is_rejected(sandbox, escape):
    with pytest.raises(ValueError, match="escapes"):
        _resolve_safe_path(escape)


@pytest.mark.parametrize("here", ["", "."])
def test_empty_and_dot_resolve_to_the_sandbox_root(sandbox, here):
    assert _resolve_safe_path(here) == sandbox.resolve()


def test_sandbox_dir_is_created_if_missing(tmp_path, monkeypatch):
    missing = tmp_path / "not-yet"
    monkeypatch.setattr(tools, "GENERATED_DIR", str(missing))
    _resolve_safe_path("top.sv")
    assert missing.is_dir()


# --------------------------------------------------------------------------
# write_file / read_file — @tool objects, invoked via .invoke(args)
# --------------------------------------------------------------------------

def test_write_then_read_round_trips(sandbox):
    body = "module TopModule;\nendmodule\n"
    result = write_file.invoke({"file_path": "top.sv", "content": body})
    assert result == f"Wrote {len(body)} characters to top.sv"
    assert read_file.invoke({"file_path": "top.sv"}) == body


def test_write_creates_missing_parent_directories(sandbox):
    write_file.invoke({"file_path": "pkg/types.svh", "content": "// pkg\n"})
    assert (sandbox / "pkg" / "types.svh").read_text() == "// pkg\n"


def test_write_overwrites_existing_content_completely(sandbox):
    write_file.invoke({"file_path": "top.sv", "content": "old long content\n"})
    write_file.invoke({"file_path": "top.sv", "content": "new\n"})
    assert read_file.invoke({"file_path": "top.sv"}) == "new\n"


def test_write_leaves_no_tmp_file_behind(sandbox):
    write_file.invoke({"file_path": "top.sv", "content": "x\n"})
    assert list(sandbox.glob("*.tmp")) == []


def test_write_escape_is_reported_not_raised(sandbox):
    result = write_file.invoke({"file_path": "../evil.sv", "content": "x"})
    assert result.startswith("Failed to write")
    assert "escapes" in result
    assert not (sandbox.parent / "evil.sv").exists()


def test_read_missing_file_reports_not_found(sandbox):
    assert read_file.invoke({"file_path": "nope.sv"}) == "File not found: nope.sv"


# --------------------------------------------------------------------------
# edit_file_block — exactly-one-match contract
# --------------------------------------------------------------------------

def test_edit_replaces_a_unique_block(sandbox):
    write_file.invoke({"file_path": "top.sv", "content": "wire a;\nlogic b;\n"})
    result = edit_file_block.invoke(
        {"file_path": "top.sv", "target_string": "wire a;", "replacement_string": "logic a;"}
    )
    assert result == "Replaced 1 occurrence in top.sv."
    assert read_file.invoke({"file_path": "top.sv"}) == "logic a;\nlogic b;\n"


def test_edit_refuses_an_ambiguous_target(sandbox):
    write_file.invoke({"file_path": "top.sv", "content": "wire a;\nwire a;\n"})
    result = edit_file_block.invoke(
        {"file_path": "top.sv", "target_string": "wire a;", "replacement_string": "logic a;"}
    )
    assert "appears 2 times" in result
    # Ambiguity must leave the file completely untouched.
    assert read_file.invoke({"file_path": "top.sv"}) == "wire a;\nwire a;\n"


def test_edit_reports_a_missing_target(sandbox):
    write_file.invoke({"file_path": "top.sv", "content": "logic a;\n"})
    result = edit_file_block.invoke(
        {"file_path": "top.sv", "target_string": "wire z;", "replacement_string": "logic z;"}
    )
    assert "not found" in result
    assert read_file.invoke({"file_path": "top.sv"}) == "logic a;\n"


def test_edit_on_a_missing_file_reports_not_found(sandbox):
    result = edit_file_block.invoke(
        {"file_path": "nope.sv", "target_string": "a", "replacement_string": "b"}
    )
    assert result == "File not found: nope.sv"


# --------------------------------------------------------------------------
# list_directory
# --------------------------------------------------------------------------

def test_list_marks_directories_with_a_trailing_slash(sandbox):
    write_file.invoke({"file_path": "top.sv", "content": "x"})
    write_file.invoke({"file_path": "pkg/types.svh", "content": "x"})
    assert list_directory.invoke({"path": "."}) == "pkg/\ntop.sv"


def test_list_reports_an_empty_directory(sandbox):
    assert list_directory.invoke({"path": "."}) == "(empty directory: .)"


def test_list_rejects_a_file_target(sandbox):
    write_file.invoke({"file_path": "top.sv", "content": "x"})
    assert list_directory.invoke({"path": "top.sv"}) == "Not a directory: top.sv"


def test_list_reports_a_missing_path(sandbox):
    assert list_directory.invoke({"path": "nope"}) == "Path not found: nope"


# --------------------------------------------------------------------------
# _project_sv_files — what the build tools actually compile
# --------------------------------------------------------------------------

def test_project_files_include_nested_sv_but_exclude_headers(sandbox):
    for name in ("top.sv", "pkg/sub.sv", "pkg/types.svh", "notes.txt"):
        write_file.invoke({"file_path": name, "content": "x"})
    found = [p.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] for p in tools._project_sv_files()]
    assert found == ["sub.sv", "top.sv"] or sorted(found) == ["sub.sv", "top.sv"]


# --------------------------------------------------------------------------
# Backend registry consistency
# --------------------------------------------------------------------------

def test_every_build_backend_has_a_failure_prefix():
    assert BUILD_TOOL_FUNCTIONS.keys() == BUILD_FAILURE_PREFIXES.keys()


def test_build_tools_report_a_missing_file_before_shelling_out(sandbox):
    # Guards the early return, so this passes with neither tool installed.
    for backend in BUILD_TOOL_FUNCTIONS.values():
        assert backend.invoke({"file_path": "nope.sv"}) == "File not found: nope.sv"
