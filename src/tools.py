import os
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
    """Compile a SystemVerilog file with Icarus Verilog and return the real compilation log (errors/warnings), without running a simulation."""
    try:
        target = _resolve_safe_path(file_path)
        if not target.exists():
            return f"File not found: {file_path}"

        # -g2012 enables SystemVerilog-2012 syntax support (iverilog defaults
        # to plain Verilog otherwise). -t null elaborates the design and
        # reports errors/warnings without generating a runnable simulation
        # output — a compile/syntax check, not a simulation run.
        result = subprocess.run(
            [IVERILOG_PATH, "-g2012", "-t", "null", str(target)],
            capture_output=True, text=True, timeout=30,
        )
        log = (result.stdout + result.stderr).strip()

        if result.returncode == 0:
            return log or "Compiled successfully — no errors or warnings."
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
    """Compile a SystemVerilog file with Slang and return the real build/lint log (errors/warnings). Not yet wired into the agent's tool set — see build_verilog for the connected equivalent."""
    try:
        target = _resolve_safe_path(file_path)
        if not target.exists():
            return f"File not found: {file_path}"

        # -Weverything turns on every warning class, not just the default
        # subset — this is what makes it a genuine lint pass rather than
        # just a compile check. No SV-version flag is needed (unlike
        # Icarus's -g2012): slang parses modern SystemVerilog by default.
        result = subprocess.run(
            [SLANG_PATH, "-Weverything", str(target)],
            capture_output=True, text=True, timeout=30,
        )
        log = (result.stdout + result.stderr).strip()

        # Unlike Icarus, slang always prints a "Build succeeded: N errors,
        # M warnings" summary line even on success, so `log` is virtually
        # never empty here — that's fine, it's more informative than
        # build_verilog's silent-on-success behavior (it surfaces warning
        # counts even when the build passes).
        if result.returncode == 0:
            return log or "Linted successfully — no errors or warnings."
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


# name -> tool dispatch. `@tool` turns each function into a StructuredTool
# object, not a plain function — it is NOT directly callable as
# write_file(**args) (that raises "object is not callable"). The correct
# call is TOOL_FUNCTIONS[name].invoke(args), passing the whole args dict
# rather than unpacking it as keyword arguments. This dict is the seam for
# adding more tools later — rtl_agent.py's loop never needs to change, only
# this file does.
TOOLS = [write_file, read_file, edit_file_block, list_directory, build_verilog]
TOOL_FUNCTIONS = {
    "write_file": write_file,
    "read_file": read_file,
    "edit_file_block": edit_file_block,
    "list_directory": list_directory,
    "build_verilog": build_verilog,
}
