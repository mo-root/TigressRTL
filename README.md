# TigressRTL

An agent that generates, writes, and compiles SystemVerilog RTL. It runs a
ReAct-style loop against a local LLM (via [Ollama](https://ollama.com)) with
tools to write, read, and edit files, and to compile them with
[Icarus Verilog](http://iverilog.icarus.com/) — so generated code is checked
against a real compiler, not just eyeballed.

## Requirements

- **Python 3.10+**
- **Ollama**, running locally, with a tool-capable model pulled — see
  [Installing Ollama](#installing-ollama) below.
- **Icarus Verilog**, for the `build_verilog` tool: `apt install iverilog`
  on Debian/Ubuntu, or `winget install Icarus.Verilog` on Windows (the
  Windows installer does not add itself to `PATH`; `src/tools.py` falls
  back to the default install location if `iverilog` isn't found on
  `PATH`).

These are all system-level installs — a Python virtual environment (below)
only isolates the Python packages (`langchain-core`, `langchain-ollama`),
not Ollama or Icarus Verilog, so both still need to be set up on whatever
machine you're actually running on, local or remote.

## Setup

Works the same whether you're setting this up locally or on a freshly
cloned remote server:

```bash
git clone <this-repo-url>
cd TigressRTL

python -m venv .venv

# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install --upgrade pip
pip install -e .
```

## Installing Ollama

The agent talks to a local Ollama server for all model inference — install
it before running the agent.

**Windows:** download and run the installer from
[ollama.com/download/windows](https://ollama.com/download/windows) (or
`winget install Ollama.Ollama`). It puts `ollama` on `PATH` and runs as a
background service automatically — no separate `ollama serve` step needed.

**Linux:**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

This is the official install script — it installs the `ollama` binary and
sets it up as a `systemd` service that starts automatically.

### Pulling a model

Once Ollama is installed and running, pull a tool-capable model:

```bash
ollama pull devstral   # default in configs/default.yaml — see Design notes for why
ollama pull llama3.2   # faster, less reliable fallback (--model llama3.2)
```

Newer, larger alternatives are also available if you want to try them:

```bash
ollama pull devstral-small-2   # updated 24B model, larger 384K context window
ollama pull devstral-2         # 123B flagship — needs significantly more RAM/VRAM
```

Run `ollama list` to confirm what's pulled locally.

## Run

```powershell
python src/rtl_agent.py                                    # uses configs/default.yaml
python src/rtl_agent.py --config configs/my-experiment.yaml # a different config file
python src/rtl_agent.py --model llama3.2                    # override just the model for one run
```

Settings (model, context window, build-retry cap, and more to come) are
loaded from a YAML config file — `configs/default.yaml` unless `--config`
points elsewhere. `--model` always wins over whatever the config says, so a
one-off override doesn't require creating a new file. To run an experiment
with different settings, copy `configs/default.yaml`, edit the copy, and
pass it via `--config` — see `src/config.py`'s `AgentConfig` for the full
list of fields and a typo'd key will raise a clear error instead of
silently using the wrong default.

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

## Config options

`configs/default.yaml` (or any file passed via `--config`) accepts the
fields defined by `AgentConfig` in `src/config.py`:

| Field | Default | What it controls |
|---|---|---|
| `model` | `devstral` | Which pulled Ollama model to use. |
| `num_ctx` | `8192` | Context window size, in tokens, given to Ollama. Can be set anywhere up to the chosen model's maximum context length (see table below) — larger values use more RAM/VRAM and run slower. |
| `max_build_retries` | `3` | How many times the agent is forced to retry after a build failure it didn't actually fix, before giving up for that turn. Any non-negative integer. |

### Models and their maximum context length

Verified locally via `ollama show <model>`:

| Model | Max context | Tool-calling | Notes |
|---|---|---|---|
| `devstral` (default) | 131072 (128K) | Reliable | See [Design notes](#design-notes) for why this is the default. |
| `llama3.2` | 131072 (128K) | Unreliable | Garbled/truncated tool-call arguments, fabricated results — see Design notes. |
| `qwen2.5-coder` | 32768 (32K) | Claims `tools`, doesn't use them | Dumps the call as plain-text JSON instead of populating `tool_calls`. |
| `llama2` | 4096 (4K) | None | No `tools` capability at all — Ollama rejects any request with tools bound. Not usable with this agent regardless of `num_ctx`. |

Newer alternatives from [Installing Ollama](#installing-ollama), per
[ollama.com](https://ollama.com/library) (not pulled/verified locally):

| Model | Max context (per ollama.com) |
|---|---|
| `devstral-small-2` | ~384K |
| `devstral-2` | ~256K |

Whatever model you choose, `num_ctx` in your config must not exceed its
maximum context length above — Ollama will error or silently clamp it
otherwise.

## Architecture

- **`src/config.py`** / **`configs/default.yaml`** — `AgentConfig`, a small
  dataclass (`model`, `num_ctx`, `max_build_retries`) loaded from YAML via
  `load_config()`. The single source of truth for experiment settings —
  see [Run](#run) above.
- **`src/model.py`** — the one swappable model/provider point (`build_llm()`
  takes an `AgentConfig`, plus the system prompt).
- **`src/tools.py`** — the five tools the agent can call: `write_file`,
  `read_file`, `edit_file_block`, `list_directory`, `build_verilog`. All of
  them are sandboxed to `src/generated/` — a model-supplied path is
  untrusted input, normalized and checked so nothing can escape that
  directory.
- **`src/rtl_agent.py`** — the interactive loop: a planning call with no
  tools bound (so the model is structurally unable to act before planning),
  then a ReAct tool-calling loop that auto-runs `build_verilog` after every
  write/edit and forces a bounded number of fix attempts
  (`config.max_build_retries`) if a build fails.

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
