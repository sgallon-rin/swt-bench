#!/usr/bin/env python3
"""
SWT-Bench Inference using pre-built evaluation images - Filesystem diff version.

This version extracts patches directly from the filesystem using git diff,
without requiring the agent to output <patch> tags. The agent only needs to
edit files and run `git add -N .` to ensure changes are tracked.

Architecture:
  src instance image (exec.eval.{arch}.{env_hash}.{instance_hash})
      ↓ already contains: conda env + deps + project code (/testbed)
      ↓ add opencode (~10s)
  inference image (swt-inf.eval.{arch}.{env_hash}.{instance_hash})
      ↓ run: --user nonroot, WORKDIR /testbed
      ↓ mount: ~/.opencode → /home/nonroot/.opencode
      ↓ execute: conda activate testbed && opencode run ...
      ↓ extract: git diff from worktree_path

Usage:
    python inference/inference_opencode_docker_fs.py --max-instances 5 --model deepseek/deepseek-v4-flash

    # With specific agent
    python inference/inference_opencode_docker_fs.py --max-instances 5 --model deepseek/deepseek-v4-flash --agent agent_name

    # With run_id for resume
    python inference/inference_opencode_docker_fs.py --max-instances 5 --model deepseek/deepseek-v4-flash --run-id 20260622
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import traceback

import docker

from docker_utils import get_base_image_name, get_arch
from src.exec_spec import make_exec_spec
from src.docker_build import build_instance_image_from_exec_spec
from src.constants import MAP_VERSION_TO_INSTALL

# --- Defaults ---
DEFAULT_DATASET = "eth-sri/SWT-bench_Lite_bm25_27k_zsb"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_TIMEOUT = None  # no limit

SCRIPT_DIR = Path(__file__).resolve().parent          # inference/
PROJECT_ROOT = SCRIPT_DIR.parent                       # swt-bench/
PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "prompts" / "opencode_fs.txt"
REPO_CACHE_DEFAULT = PROJECT_ROOT / "repo-cache"
WORKSPACE_DIR_DEFAULT = PROJECT_ROOT / "tmp" / "workspaces"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"

GIT_BASE_URL = os.environ.get("GIT_BASE_URL", "https://github.com")
HUGGINGFACE_TOKEN = os.environ.get("HF_TOKEN")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

OPCODE_VERSION = "v1.17.9"

# ---------------------------------------------------------------------------
# Streaming subprocess helper
# ---------------------------------------------------------------------------

def _run_streamed(cmd, cwd, timeout, prefix="", err_prefix="ERR", env=None, check=True, quiet_stderr=False):
    """Run a command, streaming stdout/stderr to console while capturing both."""
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
# Repo management (reused from inference_opencode.py)
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

    After copying, removes all commits after base_commit to prevent the agent
    from seeing newer commits that might contain the fix.
    """
    safe_repo_name = repo_path.name
    instance_dir = workspace_dir / instance_id
    worktree_path = instance_dir / safe_repo_name

    if instance_dir.exists():
        shutil.rmtree(instance_dir, ignore_errors=True)

    instance_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Copying repo to workspace ...")
    shutil.copytree(str(repo_path), str(worktree_path), symlinks=True)

    # Checkout the base commit
    _run_streamed(
        ["git", "-C", str(worktree_path), "checkout", "--detach", base_commit],
        cwd=str(instance_dir), timeout=60, prefix="git", err_prefix="git",
    )

    # Remove all branches to prevent access to commits after base_commit
    _run_streamed(
        ["git", "-C", str(worktree_path), "branch", "-D", "main", "master", "develop"],
        cwd=str(instance_dir), timeout=10, prefix="git", err_prefix="git",
        check=False,  # Ignore errors if branches don't exist
    )

    # Remove remote to prevent fetching newer commits
    _run_streamed(
        ["git", "-C", str(worktree_path), "remote", "remove", "origin"],
        cwd=str(instance_dir), timeout=10, prefix="git", err_prefix="git",
        check=False,
    )

    # Expire reflog and prune unreachable objects
    _run_streamed(
        ["git", "-C", str(worktree_path), "reflog", "expire", "--expire=now", "--all"],
        cwd=str(instance_dir), timeout=10, prefix="git", err_prefix="git",
        check=False,
    )
    _run_streamed(
        ["git", "-C", str(worktree_path), "gc", "--prune=now"],
        cwd=str(instance_dir), timeout=30, prefix="git", err_prefix="git",
        check=False,
    )

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


def write_issue(worktree_path, issue_text):
    (worktree_path / "ISSUE.md").write_text(issue_text)


# ---------------------------------------------------------------------------
# Image management
# ---------------------------------------------------------------------------

def ensure_src_instance_image(exec_spec) -> str:
    """Ensure src instance image exists. Build if missing.

    Uses src's build logic (build_instance_image_from_exec_spec) which
    handles base → env → instance image chain.

    Returns the src instance image key.
    """
    client = docker.from_env()
    image_key = exec_spec.instance_image_key

    try:
        client.images.get(image_key)
        print(f"  ✓ Src instance image exists: {image_key}")
        return image_key
    except docker.errors.ImageNotFound:
        pass

    print(f"  Building src instance image: {image_key}")
    print(f"  (This builds base → env → instance if needed)")
    try:
        build_instance_image_from_exec_spec(exec_spec, force_rebuild=False)
        print(f"  ✓ Src instance image built: {image_key}")
        return image_key
    except Exception as e:
        print(f"  ✗ Failed to build src instance image: {e}")
        raise


def build_inference_image(src_instance_key: str, arch: str) -> str:
    """Build inference image on top of src instance image.

    Adds opencode binary to the image. Reuses existing inference images
    if already built.

    Returns the inference image key.
    """
    # Derive inference image key from src image key
    inference_image_key = src_instance_key.replace("exec.eval", "swt-inf.eval")

    client = docker.from_env()
    try:
        client.images.get(inference_image_key)
        print(f"  ✓ Inference image exists: {inference_image_key}")
        return inference_image_key
    except docker.errors.ImageNotFound:
        pass

    print(f"  Building inference image: {inference_image_key}")

    # Determine opencode arch
    if arch == "arm64":
        ocode_arch = "arm64"
    else:
        ocode_arch = "x64"

    # Create Dockerfile that adds opencode on top of src instance image
    dockerfile = f"""FROM {src_instance_key}
RUN mkdir -p /home/nonroot/.local/state && chown -R nonroot:nonroot /home/nonroot/.local
RUN chown -R nonroot:nonroot /testbed
RUN curl -fsSL "https://github.com/anomalyco/opencode/releases/download/{OPCODE_VERSION}/opencode-linux-{ocode_arch}.tar.gz" \\
    | tar xz -C /usr/local/bin opencode
"""

    # Build the image
    with tempfile.TemporaryDirectory() as tmpdir:
        dockerfile_path = Path(tmpdir) / "Dockerfile"
        dockerfile_path.write_text(dockerfile)

        try:
            result = client.images.build(
                path=tmpdir,
                tag=inference_image_key,
                rm=True,
                forcerm=True,
                decode=False,
            )
            print(f"  ✓ Inference image built: {inference_image_key}")
            return inference_image_key
        except docker.errors.BuildError as e:
            print(f"  ✗ Failed to build inference image: {e}")
            raise


def check_agent_in_container(inference_image: str, agent: str, timeout: int = 30) -> bool:
    """Run `opencode agent list` in a container to verify the agent exists.

    Returns True if agent is found, False otherwise.
    """
    if agent is None:
        return True

    home_path = Path.home()
    auth_dir = home_path / ".local/share/opencode"
    config_dir = home_path / ".config/opencode"
    opencode_dir = home_path / ".opencode"

    cmd = ["docker", "run", "--rm", "--user", "nonroot"]

    # Mount config directories
    if auth_dir.exists():
        cmd.extend(["-v", f"{auth_dir}:/home/nonroot/.local/share/opencode"])
    if config_dir.exists():
        cmd.extend(["-v", f"{config_dir}:/home/nonroot/.config/opencode"])
    if opencode_dir.exists():
        cmd.extend(["-v", f"{opencode_dir}:/home/nonroot/.opencode"])

    cmd.extend([
        "--workdir", "/testbed",
        inference_image,
        "bash", "-c",
        f"source /opt/miniconda3/bin/activate && conda activate testbed && opencode agent list 2>&1 | grep '{agent} (primary)'",
    ])

    try:
        result = _run_streamed(
            cmd,
            cwd="/tmp",
            timeout=timeout,
            prefix="agent-check",
            err_prefix="agent-check",
            check=False,
            quiet_stderr=True,
        )
        if result.returncode == 0:
            return True  # grep found the agent

        # Failed - run again without grep to show available agents for debugging
        debug_cmd = cmd[:-1] + [
            "bash", "-c",
            "source /opt/miniconda3/bin/activate && conda activate testbed && opencode agent list 2>&1",
        ]
        debug_result = _run_streamed(
            debug_cmd,
            cwd="/tmp",
            timeout=timeout,
            prefix="agent-check-debug",
            err_prefix="agent-check-debug",
            check=False,
        )
        return False
    except subprocess.TimeoutExpired:
        print(f"  WARNING: Agent check timed out, proceeding anyway")
        return True


def run_opencode_docker_v2(
    inference_image: str,
    prompt: str,
    model: str,
    timeout: int,
    instance_id: str,
    agent: str = None,
) -> subprocess.CompletedProcess:
    """Run opencode inside a container based on inference image.

    Uses the same user (nonroot) and working directory (/testbed) as src images.
    Mounts opencode config from host.
    """
    prompt_b64 = base64.b64encode(prompt.encode()).decode()
    agent_flag = f"--agent {agent}" if agent else ""

    # Create the script to run inside the container
    setup_script = f"""#!/bin/bash
set -e
export HOME=/home/nonroot
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
echo "  [setup] Running opencode..."
OPCODE_PROMPT=$(echo {prompt_b64} | base64 -d)
opencode run "$OPCODE_PROMPT" --model {model} {agent_flag} --title {instance_id} --dir /testbed
"""

    home_path = Path.home()
    auth_dir = home_path / ".local/share/opencode"
    config_dir = home_path / ".config/opencode"
    opencode_dir = home_path / ".opencode"

    # Write script to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(setup_script)
        script_path = f.name

    try:
        cmd = ["docker", "run", "--rm", "--user", "nonroot"]

        # Mount the script
        cmd.extend(["-v", f"{script_path}:/run.sh"])

        # Mount opencode config directories
        if auth_dir.exists():
            cmd.extend(["-v", f"{auth_dir}:/home/nonroot/.local/share/opencode"])
        if config_dir.exists():
            cmd.extend(["-v", f"{config_dir}:/home/nonroot/.config/opencode"])
        if opencode_dir.exists():
            cmd.extend(["-v", f"{opencode_dir}:/home/nonroot/.opencode"])

        cmd.extend([
            "--workdir", "/testbed",
            inference_image,
            "bash", "/run.sh",
        ])

        return _run_streamed(
            cmd,
            cwd="/tmp",
            timeout=timeout,
            prefix="opencode",
            err_prefix="opencode",
            check=False,
            quiet_stderr=True,
        )
    finally:
        os.unlink(script_path)


# ---------------------------------------------------------------------------
# Patch extraction - Filesystem only
# ---------------------------------------------------------------------------

def extract_patch(worktree_path):
    """Extract unified git diff directly from the workspace filesystem.

    This is the ONLY extraction method - no parsing of agent output.
    The patch comes directly from `git diff` in the worktree.
    """
    # Ensure all changes are staged for diff
    subprocess.run(
        ["git", "add", "-N", "."],
        cwd=str(worktree_path),
        capture_output=True, text=True,
    )
    # Get the diff
    r = subprocess.run(
        ["git", "diff"],
        cwd=str(worktree_path),
        capture_output=True, text=True,
    )
    return r.stdout.strip()


def is_valid_patch(patch):
    """Validate that the patch is non-empty and contains a valid git diff.

    Returns (is_valid, reason) tuple.
    """
    if not patch:
        return False, "empty patch - agent did not make any changes"

    if not patch.startswith("diff --git"):
        return False, "patch does not start with 'diff --git'"

    return True, None


def get_patch_files(patch):
    """Extract list of files modified in the patch (for debugging)."""
    if not patch:
        return []
    files = []
    for line in patch.split('\n'):
        if line.startswith('diff --git a/'):
            parts = line.split('diff --git a/')[1].split(' b/')
            if len(parts) == 2:
                files.append(parts[0])
    return files


def verify_workspace_isolation(worktree_path):
    """Verify ALL modifications are within the workspace."""
    result = subprocess.run(
        ["git", "-C", str(worktree_path), "status", "--porcelain"],
        capture_output=True, text=True,
    )
    changed_files = []
    for line in result.stdout.strip().split('\n'):
        if line.strip():
            file_path = line[3:].strip()
            if file_path.startswith('.venv/'):
                continue
            changed_files.append(file_path)

    worktree_resolved = worktree_path.resolve()
    violations = []
    for f in changed_files:
        full_path = (worktree_path / f).resolve()
        try:
            full_path.relative_to(worktree_resolved)
        except ValueError:
            violations.append(f)

    return violations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SWT-Bench inference using pre-built evaluation images (filesystem diff)"
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
                        help="Run identifier for resume. Same run_id resumes, different run_id is independent.")

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

    print(f"\n  Using pre-built evaluation images for inference")
    print(f"  (src instance images + opencode)")
    print(f"  Patch extraction: filesystem git diff (no agent output parsing)\n")

    # --- process instances -------------------------------------------------
    success_ids = []
    failed_ids = []
    for idx, instance in enumerate(instances):
        instance_id   = instance["instance_id"]
        repo          = instance["repo"]
        base_commit   = instance["base_commit"]
        issue         = instance["problem_statement"]
        version       = instance.get("version", "")

        print(f"\n{'=' * 60}")
        print(f"[{idx + 1}/{len(instances)}] {instance_id}")
        print(f"  Repo: {repo}  Version: {version}  Commit: {base_commit[:8]}")

        worktree_path = None
        try:
            repo_path      = ensure_repo(repo, repo_cache_dir)
            worktree_path  = setup_workspace(repo_path, base_commit, instance_id, workspace_dir)
            write_issue(worktree_path, issue)

            prompt = prompt_template.format(issue=issue)

            # Map dataset fields to SWEbenchInstance format expected by make_exec_spec
            exec_instance = dict(instance)
            exec_instance["golden_code_patch"] = instance.get("patch", "")
            exec_instance["golden_test_patch"] = instance.get("test_patch", "")

            # Create exec_spec to get image keys
            exec_spec = make_exec_spec(exec_instance)

            # Step 1: Ensure src instance image exists
            print(f"  Checking src instance image...")
            src_instance_key = ensure_src_instance_image(exec_spec)

            # Step 2: Build inference image (adds opencode)
            print(f"  Building inference image...")
            inference_image = build_inference_image(src_instance_key, exec_spec.arch)

            # Step 3: Check agent availability
            if args.agent:
                print(f"  Checking if agent '{args.agent}' is available...")
                agent_available = check_agent_in_container(inference_image, args.agent)
                if not agent_available:
                    print(f"  ✗ Agent '{args.agent}' not found in container!")
                    print(f"  Ensure ~/.opencode is properly configured.")
                    failed_ids.append(instance_id)
                    continue
                print(f"  ✓ Agent '{args.agent}' found")

            # Step 4: Run opencode
            print(f"  Running opencode (model={args.model}, agent={args.agent}, timeout={args.timeout}s) ...")
            log_file = workspace_dir / instance_id / f"{instance_id}_opencode.log"
            print(f"  Log → {log_file}")
            t0 = time.time()
            result = run_opencode_docker_v2(
                inference_image, prompt, args.model, args.timeout, instance_id, args.agent
            )
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

            # Extract patch directly from filesystem (no agent output parsing)
            model_patch = extract_patch(worktree_path)

            # Validate patch
            is_valid, reason = is_valid_patch(model_patch)
            if not is_valid:
                print(f"  WARNING: {reason}")
                failed_ids.append(instance_id)
            else:
                print(f"  Patch: {len(model_patch)} chars")

                # Verify workspace isolation
                violations = verify_workspace_isolation(worktree_path)
                if violations:
                    print(f"  ⚠️  WORKSPACE ISOLATION VIOLATION: {violations}")
                else:
                    print(f"  ✓ Workspace isolation verified")

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
        print(f"   python inference/inference_opencode_docker_fs.py \\")
        print(f"       --instance-ids {failed_ids_str} \\")
        print(f"       --model {args.model} \\")
        print(f"       {agent_flag}\\")
        print(f"       --run-id {run_tag}")
        print()
        print(f"① Evaluate (after retrying):")
    else:
        print(f"① Evaluate:")
    print(f"   python -m src.main \\")
    print(f"       --dataset_name princeton-nlp/SWE-bench_Lite \\")
    print(f"       --predictions_path {output_path} \\")
    print(f"       --filter_swt \\")
    print(f"       --instance_ids $(cat dataset/swt_bench_lite_subset.txt | tr '\\n' ' ') \\")
    print(f"       --run_id {run_tag}")
    print()
    print(f"② Report (after evaluation):")
    print(f"   python -m src.report_custom run_instance_swt_logs/{run_tag}/{model_name} --total 51")


if __name__ == "__main__":
    main()
