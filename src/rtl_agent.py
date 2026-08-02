import argparse
import sys

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from model import build_llm, SYSTEM_PROMPT, PLANNING_INSTRUCTION
from tools import TOOLS, TOOL_FUNCTIONS

# Windows terminals default to a codepage that can't render some characters —
# force UTF-8 output so nothing gets garbled.
sys.stdout.reconfigure(encoding="utf-8")

# --model lets you swap between locally-pulled Ollama models (e.g. llama3.2,
# codellama, qwen2.5-coder) without editing any file — useful since RTL
# quality and tool-calling reliability vary a lot between models (see
# model.py and README.md).
parser = argparse.ArgumentParser(description="Chat with a SystemVerilog RTL design assistant.")
parser.add_argument(
    "--model", default=None,
    help="Ollama model to use, e.g. codellama, llama3.2 (must already be pulled via `ollama pull`)",
)
args = parser.parse_args()

# Bounded retry cap for the build-failure nudge below — without a cap, a
# model that can never actually fix a given error would loop forever.
MAX_BUILD_RETRIES = 3

# `llm` is our handle to the model. `.bind_tools(TOOLS)` returns a new
# runnable that knows about each tool's schema and may respond with
# tool_calls instead of (or alongside) plain text — the model itself never
# executes anything, it only ever *requests* a call.
llm = build_llm(args.model)
llm_with_tools = llm.bind_tools(TOOLS)

# `messages` is the full conversation history sent on every request —
# the API is stateless, so the whole transcript is resent each time. Each
# turn is a typed message object:
#   SystemMessage(...) - instructions, sent once up front
#   HumanMessage(...)  - you
#   AIMessage(...)     - the model's turn (may carry .tool_calls)
#   ToolMessage(...)   - a tool's result, tagged with which call it answers
messages = [SystemMessage(content=SYSTEM_PROMPT)]

print("Chatting with", args.model or "the default model", "— a SystemVerilog RTL design assistant.")
print("Type 'exit' or 'quit' to stop.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in {"exit", "quit"}:
        break
    if not user_input:
        continue

    messages.append(HumanMessage(content=user_input))

    # Planning phase: call the PLAIN `llm` — no tools bound — so the model
    # is structurally incapable of acting yet, no matter what it decides.
    # This is a hard guarantee from the code, not a hope that a "plan
    # before acting" prompt instruction gets followed — local models have
    # repeatedly been observed skipping or misordering such instructions.
    plan_response = llm.invoke(messages + [HumanMessage(content=PLANNING_INSTRUCTION)])
    print("[Plan]", plan_response.content, "\n")
    messages.append(plan_response)

    # Without this, `messages` would end in two consecutive assistant turns
    # (the plan, then immediately another assistant response with no new
    # user turn in between) once the execution loop below calls the model
    # again — an unusual pattern outside normal alternating user/assistant
    # chat structure that some models handle poorly (observed: an entirely
    # empty response with zero tool_calls). A short synthetic user turn
    # restores normal alternation and gives an explicit cue to switch from
    # planning to acting.
    messages.append(HumanMessage(content="Proceed with your plan now, using the available tools."))

    # Tracks whether the most recent auto-build (below) failed and hasn't
    # been fixed yet, and how many times we've already nudged for a fix
    # this turn — reset per user turn. See the retry check further down.
    build_failed = False
    build_retry_count = 0

    # Inner loop: the ReAct cycle for this one turn — keep calling the
    # model and executing whatever tools it requests until a response has
    # no more tool_calls, which is the model's final answer for this turn.
    while True:
        try:
            response = llm_with_tools.invoke(messages)
        except Exception as e:
            # Some models have no tool-calling support in Ollama at all and
            # reject any request with tools bound. That 400 comes back as a
            # raw ResponseError traceback by default; recognize it and fail
            # with an actionable message instead, since every subsequent
            # turn would hit the same wall.
            if "does not support tools" in str(e):
                print(
                    f"\nError: model '{args.model or 'the default model'}' does not "
                    "support tool calling in Ollama.\nThis agent's tools (write_file, "
                    "read_file, edit_file_block, list_directory, build_verilog) require "
                    "a tool-capable model.\nTry --model llama3.2 instead — see "
                    "README.md for what's been tested."
                )
                sys.exit(1)
            raise
        messages.append(response)

        # `response.tool_calls` is a list of dicts:
        #   {"name": "write_file", "args": {...}, "id": "..."}
        if not response.tool_calls:
            # A response with no tool_calls is only treated as the real
            # final answer if the last known build actually succeeded (or
            # never ran). A model can correctly diagnose a build failure in
            # prose, say it will fix it, and then simply not call
            # write_file/edit_file_block in that same response — auto-build
            # enforces that a failure is *seen*, but nothing forces it to be
            # *acted on*. So otherwise, nudge for a genuine fix attempt and
            # keep the loop going, up to MAX_BUILD_RETRIES times.
            if build_failed and build_retry_count < MAX_BUILD_RETRIES:
                build_retry_count += 1
                print(f"[Retry] Build is still failing and no fix was applied — "
                      f"forcing attempt {build_retry_count}/{MAX_BUILD_RETRIES}.")
                messages.append(HumanMessage(content=(
                    "The last build_verilog result was a failure, and you did not "
                    "call write_file or edit_file_block to actually apply a fix — "
                    "you only described one. Call the appropriate tool now with "
                    "the corrected content."
                )))
                continue
            break  # model gave its final answer for this turn — inner loop ends

        for call in response.tool_calls:
            name = call["name"]
            print("[Action]", name, call["args"])

            # TOOL_FUNCTIONS[name] is a StructuredTool object (from @tool),
            # not a plain function — it must be called via .invoke(args),
            # passing the whole args dict, not unpacked as **args.
            tool_fn = TOOL_FUNCTIONS[name]
            result = tool_fn.invoke(call["args"])

            # Enforcement, not a request: SYSTEM_PROMPT asks the model to
            # always build after writing, but that isn't reliable — models
            # have skipped the write entirely, called build_verilog on a
            # file that doesn't exist yet, and narrated fake write_file(...)/
            # build_verilog(...) calls as plain text instead of actually
            # invoking them. So instead of trusting the model to remember,
            # the harness runs the build itself right after any successful
            # write/edit, and folds the result into the SAME tool response —
            # the model sees it whether or not it asked.
            if name in ("write_file", "edit_file_block") and result.startswith(("Wrote", "Replaced")):
                file_path = call["args"].get("file_path")
                build_result = TOOL_FUNCTIONS["build_verilog"].invoke({"file_path": file_path})
                print("[Auto-Build]", build_result)
                # Drives the retry check above: a real, current failure sets
                # this True; a successful build clears it (and resets the
                # retry count) so a later, unrelated failure gets its own
                # fresh set of attempts rather than inheriting an old count.
                build_failed = build_result.startswith("Compilation failed")
                if not build_failed:
                    build_retry_count = 0
                result = f"{result}\n\n[automatically ran build_verilog after {name}]\n{build_result}"

            # Print the tool's actual return value directly — never rely on
            # the model's own later paraphrase of it. Models have been
            # observed confidently misreporting a tool result (e.g.
            # describing a full module when read_file actually returned a
            # single "{"), so this is the ground truth, shown regardless of
            # what the model goes on to say.
            print("[Result]", result)

            # ToolMessage links the result back to the specific call it
            # answers via tool_call_id — the model matches these up itself.
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    final = messages[-1]
    print("Assistant:", final.content or "(no final text — see actions above)", "\n")
