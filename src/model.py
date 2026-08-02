from langchain_ollama import ChatOllama

from config import AgentConfig

# The one swappable model/provider point in this project — everything else
# (tools.py, rtl_agent.py) talks to `build_llm()`'s return value, not to
# ChatOllama directly, so switching providers later only means editing here.
# Model choice and context window live in AgentConfig (see config.py) rather
# than as constants here, so they're driven by the YAML config instead of
# requiring a code edit to change.

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


def build_llm(config: AgentConfig) -> ChatOllama:
    return ChatOllama(
        model=config.model,
        num_ctx=config.num_ctx,
    )
