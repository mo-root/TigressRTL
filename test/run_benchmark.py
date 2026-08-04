import argparse
import fnmatch
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Windows terminals default to a codepage that can't render some characters —
# force UTF-8 output so nothing gets garbled (same fix as rtl_agent.py).
sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent
RTL_AGENT = REPO_ROOT / "src" / "rtl_agent.py"

# test/run_benchmark.py lives outside src/, so tools.py/config.py aren't on
# sys.path the way they are for scripts run directly from inside src/ —
# there's no active editable install of this package on this machine (every
# other script here is just run directly, never `pip install -e .`'d), so
# don't assume `tools`/`config` are globally importable. Adding src/ here
# makes this script self-contained regardless of install state.
sys.path.insert(0, str(REPO_ROOT / "src"))

from config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from tools import GENERATED_DIR  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the RTL agent over the verilog-eval spec-to-rtl problems, "
                    "one fresh agent process per problem. Generation only — no "
                    "validation/scoring against the dataset's reference solutions yet."
    )
    parser.add_argument(
        "--dataset-dir", required=True,
        help="Path to a cloned verilog-eval repo's dataset_spec-to-rtl/ directory "
             "(git clone https://github.com/NVlabs/verilog-eval.git)",
    )
    parser.add_argument(
        "--configs", nargs="+", default=None,
        help="One or more AgentConfig YAML files to sweep (default: configs/default.yaml). "
             "The full problem set is run once per config.",
    )
    parser.add_argument(
        "--problems", nargs="*", default=None,
        help="Exact problem names or fnmatch-style globs (e.g. 'Prob00*') to filter the "
             "discovered set. Default: all problems in --dataset-dir.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the (post-filter) problem count — useful for a quick smoke test.",
    )
    parser.add_argument(
        "--output-dir", default=str(REPO_ROOT / "benchmark_runs"),
        help="Parent directory for timestamped run directories (default: benchmark_runs/).",
    )
    parser.add_argument(
        "--resume-dir", default=None,
        help="Continue a specific previous timestamped run directory instead of creating a "
             "new one — skips (config, problem) pairs already marked completed in it. "
             "Without this, every invocation always starts fresh in a new directory.",
    )
    parser.add_argument(
        "--timeout", type=float, default=None,
        help="Per-problem subprocess timeout in seconds (default: none — a problem can take "
             "however long the model needs).",
    )
    parser.add_argument(
        "--python", default=sys.executable,
        help="Interpreter used to launch rtl_agent.py (default: the current interpreter, so "
             "an activated venv is respected automatically).",
    )
    return parser.parse_args()


def discover_problems(dataset_dir: Path) -> list[str]:
    prompt_files = sorted(dataset_dir.glob("*_prompt.txt"))
    return [p.name[: -len("_prompt.txt")] for p in prompt_files]


def clear_generated_dir(generated_dir: str) -> None:
    path = Path(generated_dir)
    if not path.exists():
        return
    for entry in path.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def copy_generated_outputs(generated_dir: str, dest: Path) -> int:
    src = Path(generated_dir)
    if not src.exists():
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    for entry in src.rglob("*"):
        if entry.is_file():
            rel = entry.relative_to(src)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry, target)
            if entry.suffix in (".sv", ".svh"):
                count += 1
    return count


def run_one(python: str, config_path: Path, prompt_single_line: str, timeout: float | None):
    stdin_text = prompt_single_line + "\nexit\n"
    start = datetime.now(timezone.utc)
    try:
        proc = subprocess.run(
            [python, str(RTL_AGENT), "--config", str(config_path)],
            input=stdin_text, capture_output=True, text=True, encoding="utf-8",
            cwd=REPO_ROOT, timeout=timeout,
        )
        log = proc.stdout + proc.stderr
        returncode = proc.returncode
        status = "completed"
    except subprocess.TimeoutExpired as e:
        log = (e.stdout or "") + (e.stderr or "")
        returncode = None
        status = "timeout"
    except OSError as e:
        log = f"harness error launching subprocess: {e}"
        returncode = None
        status = "error"
    duration = (datetime.now(timezone.utc) - start).total_seconds()
    return status, returncode, log, duration


def main():
    args = parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    config_paths = (
        [Path(c).resolve() for c in args.configs]
        if args.configs
        else [DEFAULT_CONFIG_PATH]
    )

    if not dataset_dir.is_dir():
        sys.exit(f"--dataset-dir '{dataset_dir}' is not a directory.")
    all_problems = discover_problems(dataset_dir)
    if not all_problems:
        sys.exit(
            f"No *_prompt.txt files found in '{dataset_dir}'. Did you point at the "
            "verilog-eval repo root instead of its dataset_spec-to-rtl/ subdirectory?"
        )

    # Validate every config up front (raises ValueError on a typo'd key) and
    # reject duplicate stems, before any subprocess runs — a long sweep
    # shouldn't die on config #2 of 5 after config #1 already finished.
    stems_seen = {}
    for config_path in config_paths:
        try:
            load_config(config_path)
        except ValueError as e:
            sys.exit(str(e))
        stem = config_path.stem
        if stem in stems_seen:
            sys.exit(
                f"Two --configs paths both resolve to the stem '{stem}': "
                f"{stems_seen[stem]} and {config_path}. Rename one to avoid an "
                "output directory collision."
            )
        stems_seen[stem] = config_path

    problems = all_problems
    if args.problems:
        problems = [
            p for p in problems
            if any(fnmatch.fnmatch(p, pattern) or p == pattern for pattern in args.problems)
        ]
        if not problems:
            sys.exit(f"--problems filter matched none of the {len(all_problems)} discovered problems.")
    if args.limit is not None:
        problems = problems[: args.limit]

    if args.resume_dir:
        run_dir = Path(args.resume_dir).resolve()
        if not run_dir.is_dir():
            sys.exit(f"--resume-dir '{run_dir}' does not exist.")
        resuming = True
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = output_dir / timestamp
        run_dir.mkdir(parents=True)
        resuming = False

    if not resuming:
        (run_dir / "command.txt").write_text(" ".join(sys.argv), encoding="utf-8")
        configs_copy_dir = run_dir / "configs"
        configs_copy_dir.mkdir()
        for config_path in config_paths:
            shutil.copy2(config_path, configs_copy_dir / config_path.name)

    print(f"Run directory: {run_dir}")
    print(f"Configs: {[str(c) for c in config_paths]}")
    print(f"Problems: {len(problems)} of {len(all_problems)} discovered\n")

    summary = {}  # config_stem -> {status: count}

    for config_path in config_paths:
        config_stem = config_path.stem
        summary[config_stem] = {}
        cfg = load_config(config_path)

        for problem in problems:
            problem_dir = run_dir / config_stem / problem
            status_file = problem_dir / "status.json"

            if resuming and status_file.exists():
                existing = json.loads(status_file.read_text(encoding="utf-8"))
                if existing.get("status") == "completed":
                    print(f"[skip] {config_stem}/{problem} already completed")
                    summary[config_stem]["skipped"] = summary[config_stem].get("skipped", 0) + 1
                    continue

            problem_dir.mkdir(parents=True, exist_ok=True)
            prompt_path = dataset_dir / f"{problem}_prompt.txt"
            raw = prompt_path.read_text(encoding="utf-8")
            single_line = " ".join(raw.split())
            (problem_dir / "prompt.txt").write_text(single_line, encoding="utf-8")

            clear_generated_dir(GENERATED_DIR)

            status, returncode, log, duration = run_one(args.python, config_path, single_line, args.timeout)

            (problem_dir / "transcript.log").write_text(log, encoding="utf-8")
            sv_count = copy_generated_outputs(GENERATED_DIR, problem_dir / "generated")

            status_file.write_text(json.dumps({
                "problem": problem,
                "config_path": str(config_path),
                "model": cfg.model,
                "num_ctx": cfg.num_ctx,
                "status": status,
                "returncode": returncode,
                "sv_file_count": sv_count,
                "duration_s": round(duration, 1),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }, indent=2), encoding="utf-8")

            print(f"[{status}] {config_stem}/{problem} ({sv_count} file(s), {duration:.1f}s)")
            summary[config_stem][status] = summary[config_stem].get(status, 0) + 1

    print(f"\nRun directory: {run_dir}")
    for config_stem, counts in summary.items():
        print(f"  {config_stem}: {counts}")


if __name__ == "__main__":
    main()
