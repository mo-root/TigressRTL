import json
import re

from langchain_ollama import ChatOllama

from config import AgentConfig

# The one swappable model/provider point in this project — everything else
# (tools.py, rtl_agent.py) talks to `build_llm()`'s return value, not to
# ChatOllama directly, so switching providers later only means editing here.
# Model choice and context window live in AgentConfig (see config.py) rather
# than as constants here, so they're driven by the YAML config instead of
# requiring a code edit to change.

def build_system_prompt(build_tool_name: str) -> str:
    # A function rather than a plain constant so the build/fix cycle it
    # describes always names whichever tool is actually bound
    # (config.verilog_build_tool in config.py) — build_verilog or
    # lint_verilog — instead of hardcoding one.
    return (
        "You are an expert SystemVerilog RTL designer. Write clean, "
        "synthesizable RTL: use always_ff for sequential logic and "
        "always_comb or assign for combinational logic, and non-blocking assignments "
        "(<=) inside always_ff blocks (never inside always_comb). "
        "Declare every port and every signal assigned inside a procedural "
        "block as logic — never wire or bare output, and never declare a "
        "wire inside an always block. For a SYNCHRONOUS reset, put the "
        "clock alone in the sensitivity list (@(posedge clk)) and check "
        "the reset signal only inside the block body; only put reset in "
        "the sensitivity list (@(posedge clk or posedge/negedge reset)) "
        "for an explicitly ASYNCHRONOUS reset. Use the exact module name "
        "given in the request, character-for-character — never rename it, "
        "even when fixing or rewriting existing code. If a request "
        "includes a Karnaugh map or truth table, read the row/column "
        "labels carefully before deriving the logic equation. If a "
        "request is ambiguous — missing bit widths, reset polarity, clock "
        "domain, or similar details — ask a clarifying question or "
        "clearly state the assumptions you're making, rather than "
        "silently guessing, systemverilog files all end with .sv. "
        f"Whenever you write or edit a file, you must call {build_tool_name} on it "
        "afterward. If the compilation log reports any errors, fix them with "
        f"write_file or edit_file_block and call {build_tool_name} again — repeat "
        "this build-and-fix cycle until the log is clean. Do not give your "
        "final answer, and do not claim the code is correct or complete, "
        f"until {build_tool_name} has actually reported no errors."
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

# Used for the fix-plan call only (see rtl_agent.py), triggered every time
# an auto-build freshly fails — same structural guarantee as
# PLANNING_INSTRUCTION (no tools bound for this call), so the model can't
# skip straight to another guessed edit without first diagnosing the
# specific error in front of it.
FIX_PLAN_INSTRUCTION = (
    "The build/lint attempt above just failed. Before any tool is "
    "available to you, write a short plan (2-4 bullet points): what "
    "specifically the error log says is wrong, and the exact change "
    "you'll make to fix it. Do not write SystemVerilog code yet — just "
    "the plan, make sure to keep the signal names consistent with the original code"
)


def build_llm(config: AgentConfig) -> ChatOllama:
    return ChatOllama(
        model=config.model,
        num_ctx=config.num_ctx,
    )


# A fenced JSON object anywhere in a reply's text. Matched non-greedily and
# only inside a fence: bare {...} in prose is far too easy to hit on a model
# narrating what it is about to do, or emitting a status object after the
# fact, and executing that would be worse than missing a call.
_FENCED_JSON_RE = re.compile(r"```(?:json|tool_code)?\s*(\{.*?\})\s*```", re.DOTALL)

# Keys different models use for the same thing. "arguments" is what the
# OpenAI-style function-calling schema uses and is what the DeepSeek and Qwen
# coder models emit; "args" is langchain's own spelling; "parameters" shows up
# in Gemini-flavoured templates.
_ARG_KEYS = ("arguments", "args", "parameters", "input")


def recover_tool_calls(text: str, valid_names) -> tuple[list[dict], str]:
    """Pull tool calls a model wrote as fenced JSON text out of `text`.

    Returns (calls, text_without_the_consumed_fences). `calls` is in the shape
    rtl_agent.py already expects from AIMessage.tool_calls, minus the id.

    Some tool-capable models emit a correct call as a ```json block in the
    reply body instead of as a structured tool_call, which leaves
    response.tool_calls empty and makes the agent read a turn that intended to
    write a file as its final answer. See the model notes in config.py: with a
    tool-enabling Modelfile but no system prompt, deepseek-coder-v2 emits
    structured calls 4/4; with this project's system prompt in the message
    list, it emits fenced JSON instead. qwen2.5-coder fails the same way.

    Deliberately conservative, because anything recovered here gets executed:
    the JSON must be fenced, parse to an object, carry a "name" that is an
    actually-bound tool, and carry a mapping of arguments. Anything else is
    left alone.
    """
    calls, consumed = [], []
    for match in _FENCED_JSON_RE.finditer(text or ""):
        try:
            payload = json.loads(match.group(1))
        except ValueError:
            continue
        # Some templates wrap the call one level down.
        if isinstance(payload, dict) and len(payload) == 1:
            inner = next(iter(payload.values()))
            if isinstance(inner, dict) and "name" in inner:
                payload = inner
        if not isinstance(payload, dict):
            continue
        name = payload.get("name")
        if name not in valid_names:
            continue
        args = next(
            (payload[k] for k in _ARG_KEYS if isinstance(payload.get(k), dict)), None
        )
        if args is None:
            continue
        calls.append({"name": name, "args": args})
        consumed.append(match.span())

    # Strip what was consumed. Leaving it behind would keep a worked example of
    # the wrong output format in `messages` for the rest of the session, on a
    # transcript that is never trimmed, re-priming the behaviour being
    # recovered from.
    cleaned = text or ""
    for start, end in reversed(consumed):
        cleaned = cleaned[:start] + cleaned[end:]
    return calls, cleaned.strip()
