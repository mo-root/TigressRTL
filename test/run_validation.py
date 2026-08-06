import argparse
import fnmatch
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Windows terminals default to a codepage that can't render some characters —
# force UTF-8 output so nothing gets garbled (same fix as rtl_agent.py / run_benchmark.py).
sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parent.parent

# Same PATH-then-fallback lookup as src/tools.py's build_verilog, since Icarus
# Verilog's installer doesn't add itself to PATH on Windows.
IVERILOG_PATH = shutil.which("iverilog") or r"C:\iverilog\bin\iverilog.exe"
VVP_PATH = shutil.which("vvp") or r"C:\iverilog\bin\vvp.exe"

# Every verilog-eval testbench ends its `final` block with this exact line
# (see e.g. dataset_spec-to-rtl/Prob001_zero_test.sv) — it's the ground truth
# for whether the DUT matched the reference module.
MISMATCH_RE = re.compile(r"Mismatches:\s*(\d+)\s+in\s+(\d+)\s+samples")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate each problem's generated RTL (from a run_benchmark.py output "
                    "directory) against verilog-eval's reference solution and testbench, and "
                    "record pass/fail results. Companion to run_benchmark.py: that script "
                    "generates code, this one validates it — kept separate since generation "
                    "and validation are different concerns run at different times."
    )
    parser.add_argument(
        "--run-dir", required=True,
        help="A run_benchmark.py output directory (e.g. benchmark_runs/<timestamp>), "
             "containing <config-stem>/<problem>/generated/ subdirectories.",
    )
    parser.add_argument(
        "--dataset-dir", required=True,
        help="Path to a cloned verilog-eval repo's dataset_spec-to-rtl/ directory "
             "(needs <problem>_ref.sv and <problem>_test.sv).",
    )
    parser.add_argument(
        "--configs", nargs="+", default=None,
        help="Config-stem subdirectory names under --run-dir to validate (default: all found).",
    )
    parser.add_argument(
        "--problems", nargs="*", default=None,
        help="Exact problem names or fnmatch-style globs (e.g. 'Prob00*') to filter the set. "
             "Default: all problems found under each config.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-validate a problem even if validation.json already exists. Default: skip it.",
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0,
        help="Per-step (compile, simulate) subprocess timeout in seconds (default: 60). Every "
             "verilog-eval testbench has its own built-in simulated-time cutoff that prints "
             "TIMEOUT and $finish on its own, so a real hang shouldn't happen — this is just "
             "a safety net against something going wrong at the process level.",
    )
    return parser.parse_args()


def discover_config_dirs(run_dir: Path) -> list[Path]:
    # run_dir also contains command.txt and a configs/ subdir (copies of the
    # YAML files used) alongside the actual <config-stem>/ problem-set dirs —
    # exclude those, they're not something to validate.
    return sorted(
        d for d in run_dir.iterdir()
        if d.is_dir() and d.name != "configs"
    )


def discover_problem_dirs(config_dir: Path) -> list[Path]:
    return sorted(d for d in config_dir.iterdir() if d.is_dir())


def find_generated_files(problem_dir: Path) -> list[Path]:
    gen_dir = problem_dir / "generated"
    if not gen_dir.exists():
        return []
    return sorted(gen_dir.rglob("*.sv"))


def run_step(cmd: list[str], cwd: Path, timeout: float):
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", cwd=cwd, timeout=timeout,
        )
        return proc.returncode, proc.stdout + proc.stderr, False
    except subprocess.TimeoutExpired as e:
        return None, (e.stdout or "") + (e.stderr or ""), True


def validate_problem(problem: str, problem_dir: Path, dataset_dir: Path, timeout: float) -> dict:
    result = {"problem": problem, "timestamp": datetime.now(timezone.utc).isoformat()}
    log_parts = []
    start = datetime.now(timezone.utc)

    ref_file = dataset_dir / f"{problem}_ref.sv"
    test_file = dataset_dir / f"{problem}_test.sv"
    if not ref_file.exists() or not test_file.exists():
        result["status"] = "missing_dataset_files"
        return result, ""

    generated_files = find_generated_files(problem_dir)
    if not generated_files:
        result["status"] = "no_generated_code"
        return result, ""

    sim_vvp = problem_dir / "sim.vvp"
    compile_cmd = [IVERILOG_PATH, "-g2012", "-o", str(sim_vvp), str(test_file), str(ref_file)] + [
        str(f) for f in generated_files
    ]
    returncode, log, timed_out = run_step(compile_cmd, problem_dir, timeout)
    log_parts.append("=== compile ===\n" + log)
    if timed_out:
        result["status"] = "compile_timeout"
        result["duration_s"] = round((datetime.now(timezone.utc) - start).total_seconds(), 2)
        return result, "\n".join(log_parts)
    if returncode != 0:
        result["status"] = "compile_error"
        result["duration_s"] = round((datetime.now(timezone.utc) - start).total_seconds(), 2)
        return result, "\n".join(log_parts)

    returncode, log, timed_out = run_step([VVP_PATH, str(sim_vvp)], problem_dir, timeout)
    log_parts.append("=== simulate ===\n" + log)
    result["duration_s"] = round((datetime.now(timezone.utc) - start).total_seconds(), 2)
    if timed_out:
        result["status"] = "sim_timeout"
        return result, "\n".join(log_parts)

    if "TIMEOUT" in log:
        # The testbench's own built-in cutoff fired (see e.g. the "add timeout
        # after 100K cycles" block in Prob001_zero_test.sv) — the simulation
        # itself never reached $finish under normal conditions.
        result["status"] = "sim_timeout"
        return result, "\n".join(log_parts)

    match = MISMATCH_RE.search(log)
    if not match:
        # Testbench didn't print the expected summary line — can't tell
        # pass/fail from this, needs a human to look at the log.
        result["status"] = "unknown"
        return result, "\n".join(log_parts)

    mismatches, samples = int(match.group(1)), int(match.group(2))
    result["mismatches"] = mismatches
    result["samples"] = samples
    result["status"] = "pass" if mismatches == 0 else "fail"
    return result, "\n".join(log_parts)


# "Build failure" = never compiled at all. "Functional failure" = compiled
# and ran, but mismatched the reference. Everything else (sim timeout,
# nothing generated, dataset files missing, unparseable testbench output)
# is neither — surfaced separately so it isn't silently lost or miscounted
# as one of the other two.
BUILD_FAILURE_STATUSES = ("compile_error", "compile_timeout")
OTHER_STATUSES = ("sim_timeout", "no_generated_code", "missing_dataset_files", "unknown")


def print_summary(run_dir: Path, summary: dict) -> None:
    print(f"\n{'=' * 60}")
    print("Validation Summary")
    print(f"{'=' * 60}")
    print(f"Run directory: {run_dir}")

    for config_name, by_status in summary.items():
        total = sum(len(v) for v in by_status.values())
        passed = len(by_status.get("pass", []))
        rate = f"{100 * passed / total:.0f}%" if total else "n/a"

        print(f"\n--- {config_name} ---")
        print(f"Pass: {passed}/{total} ({rate})")

        build_failed = [(p, s) for s in BUILD_FAILURE_STATUSES for p, r in by_status.get(s, [])]
        if build_failed:
            print(f"\nBuild failures ({len(build_failed)}) — never compiled:")
            for problem, status in sorted(build_failed):
                print(f"  {problem} [{status}]")

        functional_failed = by_status.get("fail", [])
        if functional_failed:
            print(f"\nFunctional failures ({len(functional_failed)}) — compiled and ran, but mismatched the reference:")
            for problem, r in sorted(functional_failed):
                print(f"  {problem} ({r.get('mismatches')}/{r.get('samples')} mismatches)")

        other = [(p, s) for s in OTHER_STATUSES for p, r in by_status.get(s, [])]
        if other:
            print(f"\nOther / inconclusive ({len(other)}):")
            for problem, status in sorted(other):
                print(f"  {problem} [{status}]")


def main():
    args = parse_args()

    run_dir = Path(args.run_dir).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()

    if not run_dir.is_dir():
        sys.exit(f"--run-dir '{run_dir}' is not a directory.")
    if not dataset_dir.is_dir():
        sys.exit(f"--dataset-dir '{dataset_dir}' is not a directory.")

    config_dirs = discover_config_dirs(run_dir)
    if args.configs:
        config_dirs = [d for d in config_dirs if d.name in args.configs]
        if not config_dirs:
            sys.exit(f"--configs filter matched none of the config-stem directories under '{run_dir}'.")
    if not config_dirs:
        sys.exit(f"No config-stem subdirectories found under '{run_dir}'. Did you point at a run_benchmark.py output directory?")

    print(f"Run directory: {run_dir}")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Configs: {[d.name for d in config_dirs]}\n")

    # config_name -> {status: [(problem, result_dict), ...]} — every problem
    # lands here regardless of whether it was freshly validated this run or
    # already had a validation.json from a previous run, so the final
    # summary always reflects the true current state of the whole run
    # directory, not just what changed in this invocation.
    summary = {}

    for config_dir in config_dirs:
        config_name = config_dir.name
        summary[config_name] = {}

        problem_dirs = discover_problem_dirs(config_dir)
        if args.problems:
            problem_dirs = [
                d for d in problem_dirs
                if any(fnmatch.fnmatch(d.name, pattern) or d.name == pattern for pattern in args.problems)
            ]

        for problem_dir in problem_dirs:
            problem = problem_dir.name
            validation_file = problem_dir / "validation.json"

            if validation_file.exists() and not args.overwrite:
                result = json.loads(validation_file.read_text(encoding="utf-8"))
                print(f"[skip] {config_name}/{problem} already validated ({result.get('status')})")
            else:
                result, log = validate_problem(problem, problem_dir, dataset_dir, args.timeout)
                (problem_dir / "validation.log").write_text(log, encoding="utf-8")
                validation_file.write_text(json.dumps(result, indent=2), encoding="utf-8")

                extra = ""
                if "mismatches" in result:
                    extra = f" ({result['mismatches']}/{result['samples']} mismatches)"
                print(f"[{result['status']}] {config_name}/{problem}{extra} ({result.get('duration_s', 0):.1f}s)")

            summary[config_name].setdefault(result["status"], []).append((problem, result))

    print_summary(run_dir, summary)


if __name__ == "__main__":
    main()
