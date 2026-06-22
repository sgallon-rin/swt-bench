#!/usr/bin/env python3
"""
SWT-Bench Inference.

For each instance:
  1. Ensure repo in cache (clone if missing)
  2. Copy repo to <run-id>/<instance-id>/<repo> and checkout base commit
  3. Run opencode to generate a reproducing test
  4. Extract git diff from the agent's output
  5. Append to JSONL predictions file
  6. Cleanup workspace

Supports resume: re-running skips already-completed instances.

Usage:
    python inference/inference_opencode.py --max-instances 5 --model deepseek/deepseek-v4-flash

    # With specific agent
    python inference/inference_opencode.py --max-instances 5 --model deepseek/deepseek-v4-flash --agent agent_name

    # With custom dataset / git mirror
    GIT_BASE_URL=https://hub.nuaa.cf python inference/inference_opencode.py --max-instances 5

    # With run_id for resume (same run_id resumes, different run_id is independent)
    python inference/inference_opencode.py --max-instances 5 --model deepseek/deepseek-v4-flash --run-id 20260622
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

# --- Defaults ---
DEFAULT_DATASET = "eth-sri/SWT-bench_Lite_bm25_27k_zsb"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_TIMEOUT = None  # no limit

SCRIPT_DIR = Path(__file__).resolve().parent          # inference/
PROJECT_ROOT = SCRIPT_DIR.parent                       # swt-bench/
PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "prompts" / "opencode.txt"
REPO_CACHE_DEFAULT = PROJECT_ROOT / "repo-cache"
WORKSPACE_DIR_DEFAULT = PROJECT_ROOT / "tmp" / "workspaces"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"

GIT_BASE_URL = os.environ.get("GIT_BASE_URL", "https://github.com")
HUGGINGFACE_TOKEN = os.environ.get("HF_TOKEN")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


# ---------------------------------------------------------------------------
# Streaming subprocess helper
# ---------------------------------------------------------------------------

def _run_streamed(cmd, cwd, timeout, prefix="", err_prefix="ERR", env=None, check=True, quiet_stderr=False):
    """Run a command, streaming stdout/stderr to console while capturing both.

    If *quiet_stderr* is True, stderr is captured but NOT printed to console.

    Returns a CompletedProcess-like result with .stdout, .stderr, .returncode.
    """
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env or os.environ.copy(),
    )

    stdout_lines = []
    stderr_lines = []

    def _read(pipe, acc, label):
        for line in iter(pipe.readline, ""):
            acc.append(line)
            if label is not False and not quiet_stderr:
                fmt = f"  [{label}] {line.rstrip()}" if label else f"  {line.rstrip()}"
                sys.stdout.write(fmt + "\n")
                sys.stdout.flush()

    t_out = threading.Thread(target=_read, args=(proc.stdout, stdout_lines, prefix))
    t_err = threading.Thread(target=_read, args=(proc.stderr, stderr_lines, err_prefix))
    t_out.start()
    t_err.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        t_out.join()
        t_err.join()
        raise
    except TypeError:
        # timeout=None means wait forever
        proc.wait()

    t_out.join()
    t_err.join()

    result = subprocess.CompletedProcess(
        args=cmd,
        returncode=proc.returncode,
        stdout="".join(stdout_lines),
        stderr="".join(stderr_lines),
    )

    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, output=result.stdout, stderr=result.stderr,
        )

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_prompt_template():
    with open(PROMPT_TEMPLATE_PATH) as f:
        return f.read()


def get_completed_ids(output_path):
    if not output_path.exists():
        return set()
    completed = set()
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    completed.add(json.loads(line)["instance_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return completed


# ---------------------------------------------------------------------------
# Repo management
# ---------------------------------------------------------------------------

def ensure_repo(repo, repo_cache_dir):
    """Clone repo into cache if not already present. Returns path."""
    safe_name = repo.replace("/", "__")
    repo_path = repo_cache_dir / safe_name

    if not (repo_path / ".git").exists():
        repo_path.mkdir(parents=True, exist_ok=True)
        clone_url = f"{GIT_BASE_URL}/{repo}.git"
        print(f"  Cloning {clone_url} ...")
        _run_streamed(
            ["git", "clone", "--quiet", clone_url, str(repo_path)],
            cwd=repo_path.parent, timeout=600, prefix="git", err_prefix="git",
        )
        print(f"  Clone complete: {repo_path}")
    return repo_path


def setup_workspace(repo_path, base_commit, instance_id, workspace_dir):
    """Create an isolated workspace by copying from the cached repo.

    Copies the entire repo folder into <workspace_dir>/<instance_id>/<repo>.
    """
    safe_repo_name = repo_path.name  # e.g. "owner__repo"
    instance_dir = workspace_dir / instance_id
    worktree_path = instance_dir / safe_repo_name

    if instance_dir.exists():
        shutil.rmtree(instance_dir)

    instance_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Copying repo to workspace ...")
    shutil.copytree(str(repo_path), str(worktree_path), symlinks=True)

    _run_streamed(
        ["git", "-C", str(worktree_path), "checkout", "--detach", base_commit],
        cwd=str(instance_dir), timeout=60, prefix="git", err_prefix="git",
    )

    # Verify commit
    result = _run_streamed(
        ["git", "-C", str(worktree_path), "rev-parse", "HEAD"],
        cwd=str(instance_dir), timeout=10, prefix="git",
    )
    actual_commit = result.stdout.strip()
    print(f"  Verified commit: {actual_commit[:8]} (expected: {base_commit[:8]})")
    if not actual_commit.startswith(base_commit):
        raise RuntimeError(f"Commit mismatch! Expected {base_commit}, got {actual_commit}")

    return worktree_path


def cleanup_workspace(worktree_path):
    if worktree_path.exists():
        shutil.rmtree(worktree_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# opencode / patch extraction
# ---------------------------------------------------------------------------

def write_issue(worktree_path, issue_text):
    (worktree_path / "ISSUE.md").write_text(issue_text)


def run_opencode(worktree_path, prompt, model, timeout, instance_id, agent=None):
    env = os.environ.copy()
    if HUGGINGFACE_TOKEN:
        env["HF_TOKEN"] = HUGGINGFACE_TOKEN

    # Layer 2: Environment isolation - remove variables that could leak local paths
    env.pop("PYTHONPATH", None)
    env.pop("VIRTUAL_ENV", None)
    env.pop("CONDA_PREFIX", None)
    env.pop("CONDA_DEFAULT_ENV", None)
    env.pop("CONDA_PYTHON_EXE", None)
    env.pop("PIP_REQUIRE_VIRTUALENV", None)
    env.pop("PIP_USER", None)

    # Sanitize PATH: remove .venv and project directories
    path_entries = []
    project_root_str = str(PROJECT_ROOT)
    for p in env.get("PATH", "").split(os.pathsep):
        if ".venv" in p or project_root_str in p:
            continue
        path_entries.append(p)
    env["PATH"] = os.pathsep.join(path_entries)

    # Log sanitized environment for verification
    print(f"  Environment isolation: PYTHONPATH={env.get('PYTHONPATH', '(not set)')}")
    print(f"  PATH (sanitized): {env['PATH'][:120]}...")

    repo_dir = str(worktree_path)
    cmd = ["opencode", "run", prompt, "--model", model]
    if agent:
        cmd.extend(["--agent", agent])
    cmd.extend(["--title", instance_id, "--dir", repo_dir])

    return _run_streamed(
        cmd,
        cwd=repo_dir,
        timeout=timeout,
        prefix="opencode",
        err_prefix="opencode",
        env=env,
        check=False,
        quiet_stderr=True,
    )


def _filesystem_diff(worktree_path):
    """Fallback: run git diff in workspace."""
    subprocess.run(
        ["git", "add", "-N", "."],
        cwd=str(worktree_path),
        capture_output=True, text=True,
    )
    r = subprocess.run(
        ["git", "diff"],
        cwd=str(worktree_path),
        capture_output=True, text=True,
    )
    return r.stdout.strip()


def _validate_patch(patch, worktree_path):
    """Layer 3: Validate that all files in the patch are within the workspace.

    Returns (is_valid, reason) tuple.
    """
    if not patch:
        return False, "empty patch"

    worktree_resolved = worktree_path.resolve()

    for line in patch.split('\n'):
        if line.startswith('--- a/') or line.startswith('+++ b/'):
            file_path = line[6:]  # Remove '--- a/' or '+++ b/'
            # Skip /dev/null (new file additions)
            if file_path == '/dev/null':
                continue
            full_path = (worktree_path / file_path).resolve()
            try:
                full_path.relative_to(worktree_resolved)
            except ValueError:
                return False, f"file outside workspace: {file_path}"

    return True, None


def verify_workspace_isolation(worktree_path):
    """Verify that all git-tracked modifications are within the workspace.

    Returns list of violating files (should be empty).
    """
    result = subprocess.run(
        ["git", "-C", str(worktree_path), "diff", "--name-only"],
        capture_output=True, text=True,
    )
    modified_files = [f for f in result.stdout.strip().split('\n') if f]
    worktree_resolved = worktree_path.resolve()

    violations = []
    for f in modified_files:
        full_path = (worktree_path / f).resolve()
        try:
            full_path.relative_to(worktree_resolved)
        except ValueError:
            violations.append(f)

    return violations


def extract_patch(stdout, worktree_path):
    """Extract unified git diff from opencode stdout.

    Tries three strategies, in order:
      1. Delimiter markers  === PATCH_START === / === PATCH_END ===
      2. Raw diff --git  pattern anywhere in the output
      3. Filesystem fallback (git diff in workspace)

    Layer 3: Validates that all files in the patch are within the workspace.
    """
    # 1) Markers
    m = re.search(
        r"=== PATCH_START ===\s*\n(.*?)=== PATCH_END ===",
        stdout, re.DOTALL | re.IGNORECASE,
    )
    if m:
        patch = m.group(1).strip()
        is_valid, reason = _validate_patch(patch, worktree_path)
        if is_valid:
            return patch
        print(f"  WARNING: patch rejected - {reason}")

    # 2) Raw diff --git block
    m = re.search(r"(diff --git .+)", stdout, re.DOTALL)
    if m:
        patch = m.group(1).strip()
        is_valid, reason = _validate_patch(patch, worktree_path)
        if is_valid:
            return patch
        print(f"  WARNING: raw diff rejected - {reason}")

    # 3) Fallback
    patch = _filesystem_diff(worktree_path)
    is_valid, reason = _validate_patch(patch, worktree_path)
    if is_valid:
        return patch
    return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SWT-Bench inference"
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--output", default=None,
                        help="JSONL output path (default: predictions/opencode__<model>.jsonl)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help="Timeout per instance in seconds (default: no limit)")
    parser.add_argument("--repo-cache", default=str(REPO_CACHE_DEFAULT))
    parser.add_argument("--workspace-dir", default=str(WORKSPACE_DIR_DEFAULT))
    parser.add_argument("--instance-ids", nargs="+", default=None,
                        help="Run only these instance IDs")
    parser.add_argument("--agent", default=None,
                        help="opencode agent to use (default: none)")
    parser.add_argument("--run-id", default=None,
                        help="Run identifier for resume. Same run_id resumes (skip success, retry fail). Different run_id is independent.")

    args = parser.parse_args()

    # --- timestamp for this run --------------------------------------------
    run_ts = time.strftime("%Y%m%d_%H%M%S")

    # --- setup paths -------------------------------------------------------
    model_safe = args.model.replace("/", "_")
    model_name = f"opencode__{model_safe}"
    agent_safe = args.agent if args.agent else "build"
    run_tag = args.run_id if args.run_id else run_ts
    output_path = Path(args.output) if args.output else (
        PREDICTIONS_DIR / f"opencode__{model_safe}__{agent_safe}__{run_tag}.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    repo_cache_dir = Path(args.repo_cache)
    repo_cache_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir = Path(args.workspace_dir) / run_tag
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # --- load dataset ------------------------------------------------------
    print(f"Loading dataset: {args.dataset}")
    from datasets import load_dataset as hf_load
    ds = hf_load(args.dataset, split="test")

    completed_ids = get_completed_ids(output_path)
    is_resume = len(completed_ids) > 0

    print(f"\nRun ID  : {run_tag}")
    if not args.run_id:
        print(f"  (auto-generated; to resume: --run-id {run_tag})")
    print(f"Output  : {output_path}")
    if is_resume:
        print(f"Status  : RESUME ({len(completed_ids)} instance(s) already completed)")
    else:
        print(f"Status  : NEW RUN")

    if args.instance_ids:
        instances = [i for i in ds if i["instance_id"] in args.instance_ids and i["instance_id"] not in completed_ids]
        completed_in_selection = len(args.instance_ids) - len(instances)
    else:
        instances = [i for i in ds if i["instance_id"] not in completed_ids]
        completed_in_selection = len(completed_ids)

    if args.max_instances:
        instances = instances[:args.max_instances]

    print(f"Total in dataset : {len(ds)}")
    print(f"Already completed: {completed_in_selection}")
    print(f"To process       : {len(instances)}")
    if not instances:
        print("Nothing to do.")
        return

    prompt_template = load_prompt_template()

    # --- process instances -------------------------------------------------
    success_ids = []
    failed_ids = []
    for idx, instance in enumerate(instances):
        instance_id   = instance["instance_id"]
        repo          = instance["repo"]
        base_commit   = instance["base_commit"]
        issue         = instance["problem_statement"]

        print(f"\n{'=' * 60}")
        print(f"[{idx + 1}/{len(instances)}] {instance_id}")
        print(f"  Repo: {repo}  Commit: {base_commit[:8]}")

        worktree_path = None
        try:
            repo_path      = ensure_repo(repo, repo_cache_dir)
            worktree_path  = setup_workspace(repo_path, base_commit, instance_id, workspace_dir)
            write_issue(worktree_path, issue)

            print(f"  Workspace: {worktree_path}")

            prompt = prompt_template.format(issue=issue)

            print(f"  Running opencode (model={args.model}, agent={args.agent}, timeout={args.timeout}s) ...")
            log_file = workspace_dir / instance_id / f"{instance_id}_opencode.log"
            print(f"  Log → {log_file}")
            t0 = time.time()
            result = run_opencode(worktree_path, prompt, args.model, args.timeout, instance_id, args.agent)
            elapsed = time.time() - t0
            print(f"\n  --- opencode finished (exit={result.returncode}, elapsed={elapsed:.0f}s) ---")

            # Save raw opencode output for debugging
            log_file.write_text(
                f"=== STDOUT ===\n{result.stdout}\n\n=== STDERR ===\n{result.stderr}"
            )

            # Print stderr only on failure
            if result.returncode != 0 and result.stderr:
                lines = result.stderr.strip().splitlines()
                preview = "\n".join(lines[:20])
                print(f"  stderr (first 20 lines):\n{preview}")
                if len(lines) > 20:
                    print(f"  ... ({len(lines) - 20} more lines in log file)")

            model_patch = extract_patch(result.stdout, worktree_path)
            if not model_patch:
                print("  WARNING: empty patch extracted")
                failed_ids.append(instance_id)
            else:
                print(f"  Patch: {len(model_patch)} chars")

                # Verify workspace isolation
                violations = verify_workspace_isolation(worktree_path)
                if violations:
                    print(f"  ⚠️  WORKSPACE ISOLATION VIOLATION: {violations}")
                else:
                    print(f"  ✓ Workspace isolation verified (all modifications within workspace)")

                pred = {
                    "instance_id": instance_id,
                    "model_name_or_path": model_name,
                    "model_patch": model_patch,
                }
                with open(output_path, "a") as f:
                    f.write(json.dumps(pred) + "\n")
                    f.flush()
                print(f"  ✓ Saved")
                success_ids.append(instance_id)

        except subprocess.TimeoutExpired:
            print(f"  ✗ TIMEOUT ({args.timeout}s)")
            failed_ids.append(instance_id)
        except subprocess.CalledProcessError as e:
            msg = e.stderr[:500] if e.stderr else str(e)
            print(f"  ✗ Command failed: {msg}")
            failed_ids.append(instance_id)
        except Exception:
            print(f"  ✗ Error: {traceback.format_exc()}")
            failed_ids.append(instance_id)

        finally:
            if worktree_path is not None:
                cleanup_workspace(worktree_path)

    total_run = len(success_ids) + len(failed_ids)
    print(f"\n{'=' * 60}")
    print(f"Run Summary:")
    print(f"  Total run : {total_run}")
    print(f"  Success   : {len(success_ids)}")
    print(f"  Failed    : {len(failed_ids)}")
    if failed_ids:
        print(f"  Failed IDs: {', '.join(failed_ids)}")
    print(f"Run timestamp: {run_ts}")
    print(f"Done. Predictions → {output_path}")
    print()
    print(f"📋 Next steps:")
    print()
    if failed_ids:
        failed_ids_str = ' '.join(failed_ids)
        agent_flag = f'--agent {args.agent} ' if args.agent else ''
        print(f"⚠️  {len(failed_ids)} instance(s) failed. Retry:")
        print(f"   python inference/inference_opencode.py \\")
        print(f"       --instance-ids {failed_ids_str} \\")
        print(f"       --model {args.model} \\")
        print(f"       {agent_flag}\\")
        print(f"       --run-id {run_tag}")
        print()
        print(f"1️⃣  Evaluate (after retrying):")
    else:
        print(f"1️⃣  Evaluate:")
    print(f"   python -m src.main \\")
    print(f"       --dataset_name princeton-nlp/SWE-bench_Lite \\")
    print(f"       --predictions_path {output_path} \\")
    print(f"       --filter_swt \\")
    print(f"       --instance_ids $(cat dataset/swt_bench_lite_subset.txt | tr '\\n' ' ') \\")
    print(f"       --run_id {run_tag}")
    print()
    print(f"2️⃣  Report (after evaluation):")
    print(f"   python -m src.report_custom run_instance_swt_logs/{run_tag}/{model_name} --total 51")


if __name__ == "__main__":
    main()
