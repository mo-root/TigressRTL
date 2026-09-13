import os
import re
import shutil
import subprocess
from pathlib import Path

from langchain_core.tools import tool

# Where the agent's RTL project lives. Kept separate from the script's own
# directory so it can be freely cleared/gitignored without touching the
# actual agent code.
GENERATED_DIR = os.path.join(os.path.dirname(__file__), "generated")

# Icarus Verilog's installer (unlike Ollama's) does not add itself to PATH,
# so a bare "iverilog" call would fail even in a fresh terminal. Check PATH
# first (covers Linux/macOS package managers, or a user who added it
# manually), then fall back to the default Windows install location.
_IVERILOG_FALLBACK = r"C:\iverilog\bin\iverilog.exe"
IVERILOG_PATH = shutil.which("iverilog") or (
    _IVERILOG_FALLBACK if os.path.exists(_IVERILOG_FALLBACK) else "iverilog"
)

# Slang (https://sv-lang.com) ships as a prebuilt binary with no installer —
# same PATH-then-fallback lookup as Icarus above, since there's no
# guarantee it ended up on PATH after extracting a release zip/tar.gz.
_SLANG_FALLBACK = r"C:\slang\slang.exe"
SLANG_PATH = shutil.which("slang") or (
    _SLANG_FALLBACK if os.path.exists(_SLANG_FALLBACK) else "slang"
)


# Matches the start of one diagnostic from either tool: slang's
# "file:line:col: error: msg" / "...warning: msg", or icarus's
# "file:line: error: msg" / bare "file:line: syntax error" (icarus's
# generic syntax-error line, always followed by a more specific "error:"
# line at the same location — so one real icarus error is two matches
# here, not one; MAX_DIAG_BLOCKS is set to 6 rather than 3 to compensate,
# giving ~3 full icarus errors or 6 full slang diagnostics). Deliberately
# does not match "note:" — a note is auxiliary info tied to the block
# before it (e.g. "previous definition here"), so it stays folded into
# that block instead of eating one of the kept slots.
# "sorry:" is icarus's third diagnostic verb, for a construct it parsed but
# won't fully support (e.g. "sorry: constant selects in always_* processes are
# not fully supported"). It is emitted on *successful* builds, one line per
# occurrence. Despite the wording of that message, only always_comb and
# always_latch trigger it — the two forms whose sensitivity list iverilog
# infers — and only for a constant bit-select read as an r-value; always_ff,
# always @(*) and an explicit sensitivity list do not. Unmatched here, a
# design that does hit it bypassed truncation entirely.
_DIAG_START_RE = re.compile(
    r"^\S+:\d+(?::\d+)?:\s*(?:(error|warning|sorry)\b|syntax error\b)", re.MULTILINE)
_DIAG_IS_ERROR_RE = re.compile(r"^\S+:\d+(?::\d+)?:\s*(error\b|syntax error\b)")
MAX_DIAG_BLOCKS = 6


def _truncate_diagnostics(stderr_log: str, max_blocks: int = MAX_DIAG_BLOCKS) -> str:
    # Both tools write every actual diagnostic to stderr (slang's stdout is
    # just a short fixed-size summary; icarus's stdout is empty), so
    # truncating stderr alone — before it's concatenated with stdout — caps
    # runaway logs (e.g. -Weverything's dozens of warnings) without
    # touching the cheap summary text. Diagnostics are multi-line for slang
    # (message + source snippet + caret) and single-line for icarus, so
    # this splits on _DIAG_START_RE rather than raw line count to avoid
    # slicing a block in half.
    starts = [m.start() for m in _DIAG_START_RE.finditer(stderr_log)]
    if len(starts) <= max_blocks:
        return stderr_log
    bounds = starts + [len(stderr_log)]
    blocks = [stderr_log[bounds[i]:bounds[i + 1]] for i in range(len(starts))]

    # Errors claim the kept slots first. Keeping simply the first N blocks was
    # fine while only errors and warnings matched, but icarus emits its "sorry:"
    # notes while elaborating a process — i.e. before an error raised by a later
    # process — so a design with a handful of them could push every actual error
    # out of the log. That leaves the model with a "Compilation failed" verdict
    # and nothing to act on, and FIX_PLAN_INSTRUCTION then asks it what the
    # error log says while the error log contains no error.
    #
    # Selection is by priority; emission stays in source order, so the log still
    # reads the way the compiler wrote it (and the trailing "N error(s) during
    # elaboration." line stays attached to the last error block, where icarus
    # put it).
    is_error = [bool(_DIAG_IS_ERROR_RE.match(b)) for b in blocks]
    order = [i for i in range(len(blocks)) if is_error[i]]
    order += [i for i in range(len(blocks)) if not is_error[i]]
    kept = set(order[:max_blocks])

    n_err = sum(1 for i in range(len(blocks)) if i not in kept and is_error[i])
    n_other = len(blocks) - len(kept) - n_err
    return (
        "".join(blocks[i] for i in sorted(kept))
        + f"... ({n_err} more error(s), {n_other} more warning/note(s) omitted) ...\n"
    )


def _project_sv_files() -> list[str]:
    # Every design file currently in the sandbox, not just the one the model
    # named — build_verilog/lint_verilog compile the whole project together
    # so a self-authored testbench in a second file that instantiates
    # TopModule resolves correctly instead of failing with a spurious
    # "unknown module" error (each run_benchmark.py problem gets its own
    # cleared GENERATED_DIR, so this never pulls in another problem's
    # files). .svh headers are excluded — they're meant to be `included,
    # not compiled as standalone top-level units.
    base = Path(GENERATED_DIR).resolve()
    return sorted(str(p) for p in base.rglob("*.sv"))


def _resolve_safe_path(path: str) -> Path:
    # Shared by every tool below. Resolves a model-supplied path against
    # GENERATED_DIR and rejects anything that would escape it (e.g. "../..",
    # or an absolute path) — the model's path arguments are untrusted input,
    # normalized and checked so nothing can escape the sandbox, while still
    # allowing safe subdirectories (needed for list_directory to browse a
    # project with packages/testbenches).
    os.makedirs(GENERATED_DIR, exist_ok=True)
    base = Path(GENERATED_DIR).resolve()

    # Models very predictably pass a path as if they need to name
    # GENERATED_DIR themselves — e.g. "/generated/foo.sv" — because a
    # request like "put it under the generated directory" reads naturally
    # as "include 'generated' in the path". Two things go wrong if we don't
    # correct for this: a leading "/" makes pathlib's `base / path` discard
    # `base` entirely (joining with an absolute path replaces the whole
    # thing, resolving to the filesystem root instead of our sandbox), and
    # even a relative "generated/foo.sv" would nest an extra generated/
    # subfolder inside GENERATED_DIR. So: drop any absolute anchor/drive,
    # and drop one leading "generated" path segment if present, before
    # resolving — since GENERATED_DIR already *is* that directory.
    parts = [p for p in Path(path).parts if p != Path(path).anchor]
    if parts and parts[0] == "generated":
        parts = parts[1:]

    candidate = base.joinpath(*parts).resolve() if parts else base
    if not candidate.is_relative_to(base):
        raise ValueError(f"path '{path}' escapes the generated/ directory")
    return candidate


@tool
def write_file(file_path: str, content: str) -> str:
    """Atomically create or overwrite a SystemVerilog (.sv/.svh) file with complete content."""
    try:
        target = _resolve_safe_path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file in the same directory first, then rename over
        # the real target. os.replace() is atomic on both POSIX and Windows,
        # so a crash or interrupted write can never leave a half-written or
        # corrupted .sv file at the target path — the file either has its
        # old complete contents or its new complete contents, never a mix.
        tmp_path = target.parent / (target.name + ".tmp")
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, target)
        rel = target.relative_to(Path(GENERATED_DIR).resolve())
        return f"Wrote {len(content)} characters to {rel}"
    except (OSError, ValueError) as e:
        return f"Failed to write {file_path}: {e}"


@tool
def read_file(file_path: str) -> str:
    """Read the entire contents of a file so the agent can review existing RTL or testbenches."""
    try:
        target = _resolve_safe_path(file_path)
        return target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except (OSError, ValueError) as e:
        return f"Failed to read {file_path}: {e}"


@tool
def edit_file_block(file_path: str, target_string: str, replacement_string: str) -> str:
    """Replace one exact block of code in a file with new code, without rewriting the whole file — use for small, targeted fixes."""
    try:
        target = _resolve_safe_path(file_path)
        text = target.read_text(encoding="utf-8")
        count = text.count(target_string)
        # Require exactly one match: zero means nothing to edit, more than
        # one means target_string is ambiguous about which occurrence to
        # change — both are reported back instead of guessing.
        if count == 0:
            return f"target_string not found in {file_path} — no changes made."
        if count > 1:
            return (
                f"target_string appears {count} times in {file_path} — "
                "make it more specific so exactly one occurrence matches."
            )
        new_text = text.replace(target_string, replacement_string, 1)
        tmp_path = target.parent / (target.name + ".tmp")
        tmp_path.write_text(new_text, encoding="utf-8")
        os.replace(tmp_path, target)
        return f"Replaced 1 occurrence in {file_path}."
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except (OSError, ValueError) as e:
        return f"Failed to edit {file_path}: {e}"


@tool
def build_verilog(file_path: str) -> str:
    """Compile a SystemVerilog file — together with every other .sv file in the project — with Icarus Verilog and return the real compilation log (errors/warnings), without running a simulation."""
    try:
        target = _resolve_safe_path(file_path)
        if not target.exists():
            return f"File not found: {file_path}"

        # -g2012 enables SystemVerilog-2012 syntax support (iverilog defaults
        # to plain Verilog otherwise). -t null elaborates the design and
        # reports errors/warnings without generating a runnable simulation
        # output — a compile/syntax check, not a simulation run. All project
        # .sv files are passed together (not just `target`) so a second file
        # that references the first — most commonly a self-authored
        # testbench instantiating TopModule — elaborates correctly instead
        # of a spurious "unknown module" error.
        result = subprocess.run(
            [IVERILOG_PATH, "-g2012", "-t", "null", *_project_sv_files()],
            capture_output=True, text=True, timeout=30,
        )
        log = (result.stdout + _truncate_diagnostics(result.stderr)).strip()

        # The verdict is stated unconditionally, never implied by the absence
        # of a log. Icarus exits 0 while still writing to stderr — "sorry:"
        # lines for unsupported constructs are the common case, and any
        # constant bit-select inside an always_* block triggers one — so
        # returning `log` alone handed the model a wall of diagnostics with no
        # indication the build had actually passed, while rtl_agent.py's
        # auto-build (which tests the failure prefix, not the log) recorded a
        # success. Model and harness disagreed about the same build.
        if result.returncode == 0:
            if not log:
                return "Compiled successfully — no errors or warnings."
            return f"Compiled successfully, with diagnostics:\n{log}"
        return f"Compilation failed (exit code {result.returncode}):\n{log}"

    except FileNotFoundError:
        # Raised if IVERILOG_PATH itself can't be executed at all (not just
        # a compile error in the .sv file) — distinct from the two errors
        # above, which mean iverilog ran fine but found a problem in the code.
        return (
            "iverilog is not installed or could not be found. Install it via "
            "`winget install Icarus.Verilog` (Windows) or your package "
            "manager, and note the installer may not add it to PATH."
        )
    except subprocess.TimeoutExpired:
        return f"Compilation of {file_path} timed out after 30s."
    except (OSError, ValueError) as e:
        return f"Failed to compile {file_path}: {e}"


@tool
def lint_verilog(file_path: str) -> str:
    """Compile a SystemVerilog file — together with every other .sv file in the project — with Slang and return the real build/lint log (errors/warnings)."""
    try:
        target = _resolve_safe_path(file_path)
        if not target.exists():
            return f"File not found: {file_path}"

        # -Weverything turns on every warning class, not just the default
        # subset — this is what makes it a genuine lint pass rather than
        # just a compile check. No SV-version flag is needed (unlike
        # Icarus's -g2012): slang parses modern SystemVerilog by default.
        # -Wno-newline-eof suppresses a cosmetic-only warning (missing
        # trailing newline) that write_file's model-supplied content
        # triggers constantly and that carries no signal about RTL
        # correctness. All project .sv files are passed together (not just
        # `target`) so a second file that references the first — most
        # commonly a self-authored testbench instantiating TopModule —
        # resolves correctly instead of a spurious "unknown module" error.
        result = subprocess.run(
            [SLANG_PATH, "-Weverything", "-Wno-newline-eof", *_project_sv_files()],
            capture_output=True, text=True, timeout=30,
        )
        log = (result.stdout + _truncate_diagnostics(result.stderr)).strip()

        # Unlike Icarus, slang always prints a "Build succeeded: N errors,
        # M warnings" summary line even on success, so `log` is virtually
        # never empty here. That summary is informative, but on its own it
        # still left the model to infer the verdict from log text; the
        # success sentence is now always present, for the same reason as in
        # build_verilog above.
        if result.returncode == 0:
            if not log:
                return "Linted successfully — no errors or warnings."
            return f"Linted successfully, with diagnostics:\n{log}"
        return f"Lint failed (exit code {result.returncode}):\n{log}"

    except FileNotFoundError:
        # Raised if SLANG_PATH itself can't be executed at all (not just a
        # lint error in the .sv file) — slang has no installer; grab a
        # prebuilt release binary and put it on PATH.
        return (
            "slang is not installed or could not be found. Download a prebuilt "
            "release from https://github.com/MikePopoloski/slang/releases and "
            "put slang(.exe) on PATH, or build from source per "
            "https://sv-lang.com/building.html."
        )
    except subprocess.TimeoutExpired:
        return f"Lint of {file_path} timed out after 30s."
    except (OSError, ValueError) as e:
        return f"Failed to lint {file_path}: {e}"


@tool
def list_directory(path: str = ".") -> str:
    """List files and subdirectories at a path, to discover project structure, packages (.svh), and testbenches."""
    try:
        target = _resolve_safe_path(path)
        if not target.exists():
            return f"Path not found: {path}"
        if not target.is_dir():
            return f"Not a directory: {path}"
        entries = sorted(target.iterdir())
        if not entries:
            return f"(empty directory: {path})"
        return "\n".join(entry.name + ("/" if entry.is_dir() else "") for entry in entries)
    except (OSError, ValueError) as e:
        return f"Failed to list {path}: {e}"


# Tools always available regardless of which build backend is selected.
# `@tool` turns each function into a StructuredTool object, not a plain
# function — it is NOT directly callable as write_file(**args) (that raises
# "object is not callable"). The correct call is
# TOOL_FUNCTIONS[name].invoke(args), passing the whole args dict rather
# than unpacking it as keyword arguments (see rtl_agent.py).
BASE_TOOLS = [write_file, read_file, edit_file_block, list_directory]

# The two interchangeable build/lint backends — AgentConfig.verilog_build_tool
# (see config.py) selects exactly one of these to bind to the model;
# rtl_agent.py builds its own TOOLS/TOOL_FUNCTIONS from BASE_TOOLS plus
# whichever one is active, rather than importing a fixed list here.
BUILD_TOOL_FUNCTIONS = {
    "icarus": build_verilog,
    "slang": lint_verilog,
}

# Return-value prefixes that mean "the build/lint failed", one per backend —
# build_verilog and lint_verilog intentionally keep their own distinct
# wording ("Compilation failed" vs "Lint failed", more informative than a
# generic message), so rtl_agent.py's auto-build enforcement needs this
# lookup to detect failure generically instead of hardcoding either string.
BUILD_FAILURE_PREFIXES = {
    "icarus": "Compilation failed",
    "slang": "Lint failed",
}

# The executable each backend shells out to. Resolved from the same constants
# the tools themselves use rather than re-deriving the PATH lookup, so this
# answers "would the active build tool actually find its binary?" and cannot
# drift from the answer.
BUILD_TOOL_BINARIES = {
    "icarus": IVERILOG_PATH,
    "slang": SLANG_PATH,
}


def missing_build_backend(name: str) -> str | None:
    """Why `name`'s executable can't be run, or None if it can be.

    Worth checking once up front rather than letting it surface per-build:
    when the binary is missing, build_verilog/lint_verilog return their
    "not installed" string, which does not start with the backend's
    BUILD_FAILURE_PREFIXES entry. rtl_agent.py's auto-build tests that prefix
    to decide whether a build failed, so a missing backend reads as a *clean
    build* — no fix-plan, no retries, and a run that reports success having
    never compiled anything. Failing closed at startup is the only point
    where that is still distinguishable from a design that simply works.
    """
    def _runnable(backend: str) -> bool:
        binary = BUILD_TOOL_BINARIES[backend]
        # Executability, not mere existence: shutil.which() accepts an absolute
        # path and applies the same exec test the shell would, and the second
        # arm covers the Windows fallback paths. A plain os.path.exists() would
        # pass a directory named "slang" in the working directory, or a
        # non-executable file at the fallback path, and the gate would then wave
        # through a backend that raises the moment subprocess tries to run it.
        return bool(shutil.which(binary)) or (
            os.path.isfile(binary) and os.access(binary, os.X_OK)
        )

    if _runnable(name):
        return None
    # Only suggest backends actually present — naming every option here would
    # include the one that just failed, which is what the user already tried.
    available = sorted(b for b in BUILD_TOOL_FUNCTIONS if _runnable(b))
    alternative = (
        f"switch verilog_build_tool to {' or '.join(available)}" if available
        else "no other configured backend is installed either"
    )
    return (
        f"build backend '{name}' is selected (verilog_build_tool in your config), but its "
        f"executable '{BUILD_TOOL_BINARIES[name]}' is not on PATH.\n"
        f"Install it, or {alternative}."
    )
