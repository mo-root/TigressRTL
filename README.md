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
ollama pull devstral          # default in configs/default.yaml — see Design notes for why
ollama pull devstral-small-2  # newer 24B model, larger 384K context window (--model devstral-small-2)
ollama pull llama3.2          # faster, less reliable fallback (--model llama3.2)
```

A much larger alternative is also available if you have the RAM/VRAM for it:

```bash
ollama pull devstral-2   # 123B flagship, 75GB — not verified locally, see configs/README.md
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
one-off override doesn't require creating a new file. See
[Config options](#config-options) below for the full list of fields.

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

Settings (model, context window, build-retry cap), the full list of fields,
and a table of usable models with their maximum context length are
documented in [`configs/README.md`](configs/README.md).

## Benchmarking

`test/run_benchmark.py` runs the agent against the spec-to-rtl problems from
[NVlabs/verilog-eval](https://github.com/NVlabs/verilog-eval) — **generation
only**, it does not (yet) check the output against the dataset's reference
solutions or run any simulation/scoring. It exists to produce raw transcripts
and generated code for comparing setups; validating that output is separate,
future work.

Clone the dataset anywhere (it's not vendored into this repo, same as
Ollama/Icarus Verilog):

```bash
git clone https://github.com/NVlabs/verilog-eval.git
```

Run it:

```bash
# One config, all 156 problems
python test/run_benchmark.py --dataset-dir verilog-eval/dataset_spec-to-rtl --configs configs/default.yaml

# Sweep multiple configs (each gets the full problem set)
python test/run_benchmark.py --dataset-dir verilog-eval/dataset_spec-to-rtl \
    --configs configs/default.yaml configs/devstral-small-2.yaml

# Quick smoke test — first 2 problems only
python test/run_benchmark.py --dataset-dir verilog-eval/dataset_spec-to-rtl --configs configs/default.yaml --limit 2
```

Each invocation runs every problem as a **fresh `rtl_agent.py` subprocess**
(its own context, no history carried over between problems) and writes into
a new timestamped directory under `benchmark_runs/` (gitignored):

```
benchmark_runs/2026-08-04_23-45-28/
  command.txt              # the exact CLI invocation, for reproducibility
  configs/default.yaml      # a copy of every --configs file used in this run
  default/                  # config-stem = Path(config).stem
    Prob001_zero/
      prompt.txt            # the single-lined prompt piped into rtl_agent.py
      transcript.log        # full agent output ([Plan]/[Action]/[Result]/[Auto-Build]/Assistant)
      status.json           # {problem, model, num_ctx, status, sv_file_count, duration_s, ...}
      generated/             # whatever ended up in src/generated/ after this run
```

If a long sweep gets interrupted, resume it with
`--resume-dir benchmark_runs/<timestamp>`, which skips any (config, problem)
pair already marked `"completed"` and retries anything that timed out or
errored. Without `--resume-dir`, every invocation always starts a fresh
timestamped directory. See `--help` for `--problems` (filter by name/glob),
`--timeout` (per-problem subprocess timeout, none by default), and `--python`
(interpreter to launch `rtl_agent.py` with).

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
- **`test/run_benchmark.py`** — sweeps `rtl_agent.py` (one fresh subprocess
  per problem) over the verilog-eval dataset across one or more configs —
  see [Benchmarking](#benchmarking) above.

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
