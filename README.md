# TigressRTL

An agent that generates, writes, and compiles SystemVerilog RTL. It runs a
ReAct-style loop against a local LLM (via [Ollama](https://ollama.com)) with
tools to write, read, and edit files, and to compile them with
[Icarus Verilog](http://iverilog.icarus.com/) — so generated code is checked
against a real compiler, not just eyeballed.

## Requirements

- **Ollama**, running locally, with a tool-capable model pulled:
  `ollama pull devstral` (the default — see [Design notes](#design-notes) for
  why). `ollama pull llama3.2` also works as a faster, less reliable
  fallback (`--model llama3.2`).
- **Icarus Verilog**, for the `build_verilog` tool. On Windows:
  `winget install Icarus.Verilog` — note the installer does not add itself
  to `PATH`; `src/tools.py` falls back to the default install location if
  `iverilog` isn't found on `PATH`.
- Python 3.10+.

## Install

```powershell
pip install -e .
```

## Run

```powershell
python src/rtl_agent.py                  # uses the default model (devstral)
python src/rtl_agent.py --model llama3.2 # override for one run
```

Type a request (e.g. *"Write a SystemVerilog module for a 4-bit synchronous
up-counter with active-low reset and enable."*) and the agent will plan,
write the file under `src/generated/`, and compile it automatically. Type
`exit` or `quit` to stop.

While it runs, three prefixes tell you what's actually happening:

- `[Plan]` — the model's stated plan, produced before any tool is available
  to it.
- `[Action]` / `[Result]` — a tool call and its real return value.
- `[Auto-Build]` — the compilation result, run automatically after every
  write/edit.

Always trust `[Result]`/`[Auto-Build]` over the `Assistant:` text that
follows — see [Design notes](#design-notes) below.

## Architecture

- **`src/model.py`** — the one swappable model/provider point (`build_llm()`,
  system prompt, default model).
- **`src/tools.py`** — the five tools the agent can call: `write_file`,
  `read_file`, `edit_file_block`, `list_directory`, `build_verilog`. All of
  them are sandboxed to `src/generated/` — a model-supplied path is
  untrusted input, normalized and checked so nothing can escape that
  directory.
- **`src/rtl_agent.py`** — the interactive loop: a planning call with no
  tools bound (so the model is structurally unable to act before planning),
  then a ReAct tool-calling loop that auto-runs `build_verilog` after every
  write/edit and forces a bounded number of fix attempts (`MAX_BUILD_RETRIES
  = 3`) if a build fails.

## Design notes

A few non-obvious things learned building this, worth knowing before
extending it:

- **Tool-calling reliability varies a lot by model**, independent of what
  `ollama show <model>` claims to support. Some models report `tools` as a
  capability but never actually populate `tool_calls` — they dump the call
  as raw JSON text instead. Others populate `tool_calls` but truncate or
  garble multi-line content. Verify empirically before trusting a model's
  declared capabilities.
- **Never trust the model's prose over a tool's actual return value.**
  Models have been observed describing a write or build that never
  happened, or fabricating a plausible-but-wrong description of a file's
  real contents — directly contradicting the tool result they'd just
  received. That's why every tool result is printed unconditionally via
  `[Result]`/`[Auto-Build]`, before the model gets a chance to say anything
  about it.
- **A prompt instruction to "plan before acting" isn't reliable enough on
  its own.** `rtl_agent.py` instead makes a first call with no tools bound
  at all, so planning happens before acting as a structural guarantee, not
  a hope.
- **Path handling needs normalization, not just a naive join.** Models
  reliably pass paths as if they need to name the sandbox directory
  themselves (e.g. `/generated/foo.sv` for "put it under the generated
  directory"), which breaks a naive `base / path` join. `_resolve_safe_path`
  in `src/tools.py` strips a redundant leading anchor/`generated` segment
  before resolving, while still rejecting genuine escape attempts like
  `../../etc/passwd`.
