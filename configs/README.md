# Config options

`configs/default.yaml` (or any file passed via `--config`) accepts the
fields defined by `AgentConfig` in `src/config.py`:

| Field | Default | What it controls |
|---|---|---|
| `model` | `devstral` | Which pulled Ollama model to use. |
| `num_ctx` | `8192` | Context window size, in tokens, given to Ollama. Can be set anywhere up to the chosen model's maximum context length (see table below) — larger values use more RAM/VRAM and run slower. |
| `max_build_retries` | `3` | How many times the agent is forced to retry after a build failure it didn't actually fix, before giving up for that turn. Any non-negative integer. |

To run an experiment with different settings, copy `default.yaml`, edit the
copy, and pass it via `--config` (e.g. `--config configs/my-experiment.yaml`).
`--model` always overrides just the model field for a one-off run without
needing a new file. A typo'd key in your YAML raises a clear error instead
of silently falling back to the wrong default — see `load_config()` in
`../src/config.py`.

## Models and their maximum context length

Verified locally via `ollama show <model>`:

| Model | Max context | Tool-calling | Notes |
|---|---|---|---|
| `devstral` (default) | 131072 (128K) | Reliable | See the root [README's Design notes](../README.md#design-notes) for why this is the default. |
| `devstral-small-2` | 393216 (384K) | Reliable | Newer 24B model, larger context than `devstral`. Real structured `tool_calls`, correctly self-corrected a build failure in testing. Also has `vision` capability (unused by this agent) and a lower built-in default temperature (0.15) than `devstral`. |
| `llama3.2` | 131072 (128K) | Unreliable | Garbled/truncated tool-call arguments, fabricated results — see Design notes. |
| `qwen2.5-coder` | 32768 (32K) | Claims `tools`, doesn't use them | Dumps the call as plain-text JSON instead of populating `tool_calls`. |
| `llama2` | 4096 (4K) | None | No `tools` capability at all — Ollama rejects any request with tools bound. Not usable with this agent regardless of `num_ctx`. |

Newer alternative from the root README's
[Installing Ollama](../README.md#installing-ollama) section, per
[ollama.com](https://ollama.com/library) (not pulled/verified locally — much
larger, 75GB):

| Model | Max context (per ollama.com) |
|---|---|
| `devstral-2` | ~256K |

Whatever model you choose, `num_ctx` in your config must not exceed its
maximum context length above — Ollama will error or silently clamp it
otherwise.
