from langchain_ollama import ChatOllama

# The one swappable model/provider point in this project — everything else
# (tools.py, rtl_agent.py) talks to `build_llm()`'s return value, not to
# ChatOllama directly, so switching providers later only means editing here.
#
# qwen2.5-coder *claims* tool support (`ollama show qwen2.5-coder` lists
# completion, tools, insert) but empirically does not reliably use it: tested
# directly against both the raw `ollama` library and langchain_ollama, it
# dumps the tool call as a raw JSON string in the response's plain text
# content instead of populating the structured tool_calls field — so
# response.tool_calls comes back empty/None and nothing actually executes.
# Declared capability metadata isn't proof of working behavior — verify
# empirically before trusting it.
#
# llama3.2 reliably produces structured tool_calls, but empirically fails in
# other ways: garbled/truncated multi-line content in tool-call arguments,
# narrating fake pseudo-tool-calls as plain text instead of invoking them,
# and confidently hallucinating tool results that contradict what the tool
# actually returned in the same response.
#
# devstral (Mistral's model purpose-built for agentic coding, via the
# OpenHands scaffold) is the first model tested here with none of those
# problems in a full end-to-end run: real structured tool_calls with
# complete, untruncated, valid SystemVerilog content; build succeeded on
# the first try; and its final answer was a byte-for-byte accurate
# description of what was actually written to disk. It's ~14GB and
# noticeably slower per turn on limited-GPU hardware (mostly CPU inference),
# but the reliability difference is large enough to make it the default.
# --model llama3.2 still works if speed matters more than reliability for a
# given session.
DEFAULT_MODEL = "devstral"

# Sized up from a typical 4096 default — generated SystemVerilog modules
# tend to run longer than plain prose/chat content that value is tuned for.
NUM_CTX = 8192

SYSTEM_PROMPT = (
    "You are an expert SystemVerilog RTL designer. Write clean, "
    "synthesizable RTL: use always_ff for sequential logic and "
    "always_comb for combinational logic, and non-blocking assignments "
    "(<=) inside always_ff blocks. If a request is ambiguous — missing "
    "bit widths, reset polarity, clock domain, or similar details — ask "
    "a clarifying question or clearly state the assumptions you're "
    "making, rather than silently guessing, systemverilog files all end with .sv. "
    "Whenever you write or edit a file, you must call build_verilog on it "
    "afterward. If the compilation log reports any errors, fix them with "
    "write_file or edit_file_block and call build_verilog again — repeat "
    "this build-and-fix cycle until the log is clean. Do not give your "
    "final answer, and do not claim the code is correct or complete, "
    "until build_verilog has actually reported no errors."
)

# Used for the planning-phase call only (see rtl_agent.py) — no tools are
# bound for that call, so the model is structurally unable to act yet no
# matter what it decides; this just shapes what it writes in that turn.
PLANNING_INSTRUCTION = (
    "Before any tool is available to you, write a short plan (3-6 bullet "
    "points) for how you will fulfill this request: what module(s) you'll "
    "create, their ports, and the order of write/build steps. Do not write "
    "SystemVerilog code yet — just the plan."
)


def build_llm(model_name: str | None = None) -> ChatOllama:
    # `model_name=None` means "use the default" — kept as a parameter
    # (rather than always reading DEFAULT_MODEL directly) so callers like
    # rtl_agent.py's --model flag can override it per run without touching
    # this file.
    return ChatOllama(
        model=model_name or DEFAULT_MODEL,
        num_ctx=NUM_CTX,
    )
