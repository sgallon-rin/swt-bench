#!/usr/bin/env python3
"""
SWT-Bench Inference.

For each instance:
  1. Clone the repo (cached) and checkout the base commit
  2. Run a code agent to generate a reproducing test
  3. Extract git diff from the agent's output
  4. Append to JSONL predictions file
  5. Cleanup workspace

Supports resume: re-running skips already-completed instances.

Usage:
    python inference/run.py --max-instances 5 --model deepseek-ve-flash

    # With custom dataset / git mirror
    GIT_BASE_URL=https://hub.nuaa.cf python inference/run.py --max-instances 5
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

# --- Defaults ---
DEFAULT_DATASET = "eth-sri/SWT-bench_Lite_bm25_27k_zsb"
DEFAULT_MODEL = "deepseek-ve-flash"
DEFAULT_TIMEOUT = 1800  # 30 minutes per instance

SCRIPT_DIR = Path(__file__).resolve().parent          # inference/
PROJECT_ROOT = SCRIPT_DIR.parent                       # swt-bench/
PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "prompts" / "opencode.txt"
REPO_CACHE_DEFAULT = PROJECT_ROOT / "repo-cache"
WORKSPACE_DIR_DEFAULT = PROJECT_ROOT / "tmp" / "workspaces"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"

GIT_BASE_URL = os.environ.get("GIT_BASE_URL", "https://github.com")
HUGGINGFACE_TOKEN = os.environ.get("HF_TOKEN")


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
        subprocess.run(
            ["git", "clone", clone_url, str(repo_path)],
            check=True, capture_output=True, text=True,
            timeout=600,
        )
        print(f"  Clone complete: {repo_path}")
    return repo_path


def setup_workspace(repo_path, base_commit, instance_id, workspace_dir):
    """Create an isolated workspace by cloning from the cached repo.

    Uses git's local clone optimisation (hardlinks).
    """
    worktree_path = workspace_dir / instance_id

    if worktree_path.exists():
        shutil.rmtree(worktree_path)

    workspace_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        ["git", "clone", str(repo_path), str(worktree_path)],
        check=True, capture_output=True, text=True,
    )
    subprocess.run(
        ["git", "-C", str(worktree_path), "checkout", "--detach", base_commit],
        check=True, capture_output=True, text=True,
    )
    return worktree_path


def cleanup_workspace(worktree_path):
    if worktree_path.exists():
        shutil.rmtree(worktree_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# opencode / patch extraction
# ---------------------------------------------------------------------------

def write_issue(worktree_path, issue_text):
    (worktree_path / "ISSUE.md").write_text(issue_text)


def run_opencode(worktree_path, prompt, model, timeout):
    env = os.environ.copy()
    if HUGGINGFACE_TOKEN:
        env["HF_TOKEN"] = HUGGINGFACE_TOKEN

    result = subprocess.run(
        ["opencode", "run", prompt, "--model", model],
        cwd=str(worktree_path),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    return result


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


def extract_patch(stdout, worktree_path):
    """Extract unified git diff from opencode stdout.

    Tries three strategies, in order:
      1. Delimiter markers  === PATCH_START === / === PATCH_END ===
      2. Raw diff --git  pattern anywhere in the output
      3. Filesystem fallback (git diff in workspace)
    """
    # 1) Markers
    m = re.search(
        r"=== PATCH_START ===\s*\n(.*?)=== PATCH_END ===",
        stdout, re.DOTALL | re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()

    # 2) Raw diff --git block
    m = re.search(r"(diff --git .+)", stdout, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 3) Fallback
    return _filesystem_diff(worktree_path)


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
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--repo-cache", default=str(REPO_CACHE_DEFAULT))
    parser.add_argument("--workspace-dir", default=str(WORKSPACE_DIR_DEFAULT))
    parser.add_argument("--instance-ids", nargs="+", default=None,
                        help="Run only these instance IDs")

    args = parser.parse_args()

    # --- setup paths -------------------------------------------------------
    model_name = f"opencode__{args.model}"
    output_path = Path(args.output) if args.output else (
        PREDICTIONS_DIR / f"opencode__{args.model}.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    repo_cache_dir = Path(args.repo_cache)
    repo_cache_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir = Path(args.workspace_dir)

    # --- load dataset ------------------------------------------------------
    print(f"Loading dataset: {args.dataset}")
    from datasets import load_dataset as hf_load
    ds = hf_load(args.dataset, split="test")

    completed_ids = get_completed_ids(output_path)

    if args.instance_ids:
        instances = [i for i in ds if i["instance_id"] in args.instance_ids]
    else:
        instances = [i for i in ds if i["instance_id"] not in completed_ids]

    if args.max_instances:
        instances = instances[:args.max_instances]

    print(f"Total in dataset : {len(ds)}")
    print(f"Already completed: {len(completed_ids)}")
    print(f"To process       : {len(instances)}")
    if not instances:
        print("Nothing to do.")
        return

    prompt_template = load_prompt_template()

    # --- process instances -------------------------------------------------
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

            prompt = prompt_template.format(issue=issue)

            print(f"  Running opencode (model={args.model}, timeout={args.timeout}s) ...")
            t0 = time.time()
            result = run_opencode(worktree_path, prompt, args.model, args.timeout)
            elapsed = time.time() - t0
            print(f"  opencode exited {result.returncode} in {elapsed:.0f}s")

            # Save raw opencode output for debugging
            log_file = workspace_dir / f"{instance_id}_opencode.log"
            log_file.write_text(
                f"=== STDOUT ===\n{result.stdout}\n\n=== STDERR ===\n{result.stderr}"
            )

            if result.stderr:
                print(f"  stderr (first 300 chars): {result.stderr[:300]}")

            model_patch = extract_patch(result.stdout, worktree_path)
            if not model_patch:
                print("  WARNING: empty patch extracted")
            else:
                print(f"  Patch: {len(model_patch)} chars")

            pred = {
                "instance_id": instance_id,
                "model_name_or_path": model_name,
                "model_patch": model_patch,
            }
            with open(output_path, "a") as f:
                f.write(json.dumps(pred) + "\n")
                f.flush()
            print(f"  ✓ Saved")

        except subprocess.TimeoutExpired:
            print(f"  ✗ TIMEOUT ({args.timeout}s)")
        except subprocess.CalledProcessError as e:
            msg = e.stderr[:500] if e.stderr else str(e)
            print(f"  ✗ Command failed: {msg}")
        except Exception:
            print(f"  ✗ Error: {traceback.format_exc()}")

        finally:
            if worktree_path is not None:
                cleanup_workspace(worktree_path)

    print(f"\n{'=' * 60}")
    print(f"Done. Predictions → {output_path}")
    print(f"Evaluate with:")
    print(f"  python -m src.main --dataset_name princeton-nlp/SWE-bench_Lite \\")
    print(f"      --predictions_path {output_path} --filter_swt --run_id opencode_exp1")


if __name__ == "__main__":
    main()
