#!/usr/bin/env python3
"""
SWT-Bench Inference using pre-built evaluation images with issue-driven-dt skill.

This script reuses src's instance images (exec.eval.*) and adds opencode
to create inference images (swt-inf-skill.eval.*). This avoids rebuilding
environments at runtime.

The build agent is instructed to load the "issue-driven-dt" skill to generate
regression tests following a structured workflow.

Architecture:
  src instance image (exec.eval.{arch}.{env_hash}.{instance_hash})
      ↓ already contains: conda env + deps + project code (/testbed)
      ↓ add opencode (~10s)
  inference image (swt-inf-skill.eval.{arch}.{env_hash}.{instance_hash})
      ↓ run: --user nonroot, WORKDIR /testbed
      ↓ mount: ~/.opencode → /home/nonroot/.opencode
      ↓ execute: conda activate testbed && opencode run ...

Usage:
    python inference/inference_opencode_docker_skill.py --max-instances 5 --model deepseek/deepseek-v4-flash

    # With specific agent
    python inference/inference_opencode_docker_skill.py --max-instances 5 --model deepseek/deepseek-v4-flash --agent agent_name

    # With run_id for resume
    python inference/inference_opencode_docker_skill.py --max-instances 5 --model deepseek/deepseek-v4-flash --run-id 20260622
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
import tarfile
import uuid

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
PROMPT_TEMPLATE_PATH = PROJECT_ROOT / "prompts" / "opencode_skill.txt"
REPO_CACHE_DEFAULT = PROJECT_ROOT / "repo-cache"
WORKSPACE_DIR_DEFAULT = PROJECT_ROOT / "tmp" / "workspaces"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
OPCODE_CONFIG_PATH = SCRIPT_DIR / "opencode.json"

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
    # Use 'swt-inf-skill.eval' to distinguish from non-skill inference images
    inference_image_key = src_instance_key.replace("exec.eval", "swt-inf-skill.eval")

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
RUN apt-get update && apt-get install -y ca-certificates && update-ca-certificates && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /home/nonroot/.local/state && chown -R nonroot:nonroot /home/nonroot/.local
RUN mkdir -p /home/nonroot/.local/share/opencode && chown -R nonroot:nonroot /home/nonroot/.local/share/opencode
RUN mkdir -p /home/nonroot/.local/share/uv && chown -R nonroot:nonroot /home/nonroot/.local/share/uv
RUN chown -R nonroot:nonroot /testbed
RUN curl -fsSL "https://github.com/anomalyco/opencode/releases/download/{OPCODE_VERSION}/opencode-linux-{ocode_arch}.tar.gz" \\
    | tar xz -C /usr/local/bin opencode
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \\
    cp /root/.local/bin/uv /usr/local/bin/uv && \\
    cp /root/.local/bin/uvx /usr/local/bin/uvx && \\
    chmod +x /usr/local/bin/uv /usr/local/bin/uvx
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


def _docker_mount_args():
    """Build the -v mount arguments for opencode config.

    auth.json is mounted to /tmp first, then copied by the nonroot user
    inside the container to avoid Docker creating root-owned parent dirs.
    """
    mounts = []
    auth_json = Path.home() / ".local/share/opencode/auth.json"
    config_dir = Path.home() / ".config/opencode"
    opencode_dir = Path.home() / ".opencode"

    if auth_json.exists():
        mounts.extend(["-v", f"{auth_json}:/tmp/auth.json:ro"])
    if config_dir.exists():
        mounts.extend(["-v", f"{config_dir}:/home/nonroot/.config/opencode"])
    if opencode_dir.exists():
        mounts.extend(["-v", f"{opencode_dir}:/home/nonroot/.opencode"])
    return mounts


def run_opencode_docker_v2(
    inference_image: str,
    prompt: str,
    model: str,
    timeout: int,
    instance_id: str,
    agent: str = None,
    opencode_config: Path = None,
    check_agent: bool = True,
    check_skill: bool = True,
) -> tuple[subprocess.CompletedProcess, Path | None, Path | None]:
    """Run opencode inside a container based on inference image.

    Uses the same user (nonroot) and working directory (/testbed) as src images.
    Mounts opencode config from host.
    Agent and skill checks are performed inside the container before running opencode.
    After opencode finishes, exports the session transcript and .ai-test-workspace
    to the host.

    Returns (result, session_export_path, workspace_export_path).
    """
    prompt_b64 = base64.b64encode(prompt.encode()).decode()
    agent_flag = f"--agent {agent}" if agent else ""
    container_name = f"opencode-{instance_id}-{uuid.uuid4().hex[:8]}"

    # Build check commands for the setup script
    agent_check = ""
    if check_agent and agent:
        agent_check = f"""
echo "  [check] Verifying agent '{agent}'..."
opencode agent list 2>&1 | grep '{agent} (primary)' || {{ echo "  ✗ Agent '{agent}' not found"; exit 1; }}
echo "  ✓ Agent '{agent}' found"
"""

    skill_check = ""
    if check_skill:
        skill_check = """
echo "  [check] Verifying skill 'issue-driven-dt'..."
test -f /home/nonroot/.opencode/skills/issue-driven-dt/SKILL.md || { echo "  ✗ Skill 'issue-driven-dt' not found"; exit 1; }
echo "  ✓ Skill 'issue-driven-dt' found"
"""

    # Create the script to run inside the container
    setup_script = f"""#!/bin/bash
set -e
export HOME=/home/nonroot
source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
git config --global --add safe.directory /testbed
if [ -f /tmp/auth.json ]; then
    mkdir -p /home/nonroot/.local/share/opencode
    cp /tmp/auth.json /home/nonroot/.local/share/opencode/auth.json
fi
{agent_check}{skill_check}
echo "  [setup] Running opencode..."
OPCODE_PROMPT=$(echo {prompt_b64} | base64 -d)
opencode run "$OPCODE_PROMPT" --model {model} {agent_flag} --title {instance_id} --dir /testbed || true

# Export session transcript
echo "  [export] Exporting session transcript..."
LATEST_SESSION=$(opencode session list --format json 2>/dev/null | python3 -c "import sys,json; sessions=json.load(sys.stdin); print(sessions[0]['id'] if sessions else '')" 2>/dev/null || true)
if [ -n "$LATEST_SESSION" ]; then
    opencode export "$LATEST_SESSION" > /tmp/opencode_session_export.json 2>/dev/null
    echo "  ✓ Session exported"
else
    echo "  ✗ No session found to export"
fi

# Export .ai-test-workspace (skill workflow artifacts)
echo "  [export] Exporting .ai-test-workspace..."
if [ -d /testbed/.ai-test-workspace ]; then
    tar -czf /tmp/ai-test-workspace.tar.gz -C /testbed .ai-test-workspace
    echo "  ✓ .ai-test-workspace archived"
else
    echo "  ✗ .ai-test-workspace not found"
fi
"""

    # Write script to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
        f.write(setup_script)
        script_path = f.name

    export_path = None
    workspace_export_path = None
    try:
        cmd = ["docker", "run", "--name", container_name, "--user", "nonroot"]

        # Mount the script
        cmd.extend(["-v", f"{script_path}:/run.sh"])

        # Mount project opencode config to workspace
        if opencode_config is not None and opencode_config.exists():
            cmd.extend(["-v", f"{opencode_config}:/testbed/opencode.json"])

        # Mount opencode config directories
        cmd.extend(_docker_mount_args())

        cmd.extend([
            "--workdir", "/testbed",
            inference_image,
            "bash", "/run.sh",
        ])

        result = _run_streamed(
            cmd,
            cwd="/tmp",
            timeout=timeout,
            prefix="opencode",
            err_prefix="opencode",
            check=False,
            quiet_stderr=True,
        )

        # Copy exported session from container to host
        try:
            tmp_export = Path(tempfile.gettempdir()) / f"opencode_session_{instance_id}.json"
            cp_result = subprocess.run(
                ["docker", "cp", f"{container_name}:/tmp/opencode_session_export.json", str(tmp_export)],
                capture_output=True, text=True, timeout=30,
            )
            if cp_result.returncode == 0 and tmp_export.exists():
                export_path = tmp_export
        except Exception:
            pass

        # Copy .ai-test-workspace archive from container to host and extract
        try:
            tmp_ws_archive = Path(tempfile.gettempdir()) / f"ai-test-workspace_{instance_id}.tar.gz"
            cp_result = subprocess.run(
                ["docker", "cp", f"{container_name}:/tmp/ai-test-workspace.tar.gz", str(tmp_ws_archive)],
                capture_output=True, text=True, timeout=30,
            )
            if cp_result.returncode == 0 and tmp_ws_archive.exists():
                workspace_export_path = tmp_ws_archive
        except Exception:
            pass

        return result, export_path, workspace_export_path
    finally:
        # Clean up container
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, timeout=10)
        os.unlink(script_path)


# ---------------------------------------------------------------------------
# Patch extraction (reused from inference_opencode.py)
# ---------------------------------------------------------------------------

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
    """Validate that all files in the patch are within the workspace."""
    if not patch:
        return False, "empty patch"

    worktree_resolved = worktree_path.resolve()

    for line in patch.split('\n'):
        if line.startswith('--- a/') or line.startswith('+++ b/'):
            file_path = line[6:]
            if file_path == '/dev/null':
                continue
            full_path = (worktree_path / file_path).resolve()
            try:
                full_path.relative_to(worktree_resolved)
            except ValueError:
                return False, f"file outside workspace: {file_path}"

    return True, None


def extract_patch(stdout, worktree_path):
    """Extract unified git diff from opencode stdout."""
    # 1) <patch> tags (must be on their own line)
    m = re.search(
        r"^\s*<patch>\s*\n?(.*?)\n?\s*</patch>\s*$",
        stdout, re.MULTILINE | re.DOTALL,
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


def is_valid_patch(stdout, patch):
    """Check if agent explicitly output a patch with <patch> tags.

    Returns (is_valid, reason) tuple.
    """
    if not patch:
        return False, "empty patch"

    # Check if agent explicitly output the patch with <patch> tags
    if "<patch>" not in stdout:
        return False, "agent did not output <patch> tag"

    # Verify tags contain actual diff content (not empty)
    m = re.search(r"^\s*<patch>\s*\n?(.*?)\n?\s*</patch>\s*$", stdout, re.MULTILINE | re.DOTALL)
    if m:
        tag_content = m.group(1).strip()
        if not tag_content:
            return False, "patch tags are empty - agent did not paste diff output between <patch> tags"
        if not tag_content.startswith("diff --git"):
            return False, "patch tags do not contain a valid git diff"

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
        description="SWT-Bench inference using pre-built evaluation images with issue-driven-dt skill"
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--output", default=None,
                        help="JSONL output path (default: predictions/opencode__<model>__skill.jsonl)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help="Timeout per instance in seconds (default: no limit)")
    parser.add_argument("--repo-cache", default=str(REPO_CACHE_DEFAULT))
    parser.add_argument("--workspace-dir", default=str(WORKSPACE_DIR_DEFAULT))
    parser.add_argument("--instance-ids", nargs="+", default=None,
                        help="Run only these instance IDs")
    parser.add_argument("--agent", default=None,
                        help="opencode agent to use (default: build)")
    parser.add_argument("--run-id", default=None,
                        help="Run identifier for resume. Same run_id resumes, different run_id is independent.")
    parser.add_argument("--skip-skill-check", action="store_true",
                        help="Skip the skill availability check")

    args = parser.parse_args()

    # --- timestamp for this run --------------------------------------------
    run_ts = time.strftime("%Y%m%d_%H%M%S")

    # --- setup paths -------------------------------------------------------
    model_safe = args.model.replace("/", "_")
    model_name = f"opencode__{model_safe}__skill"
    agent_safe = args.agent if args.agent else "build"
    run_tag = args.run_id if args.run_id else run_ts
    output_path = Path(args.output) if args.output else (
        PREDICTIONS_DIR / f"opencode__{model_safe}__skill__{agent_safe}__{run_tag}.jsonl"
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
    print(f"  (src instance images + opencode + issue-driven-dt skill)\n")

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

            # Step 3: Run opencode (agent/skill checks are performed inside the container)
            effective_agent = args.agent if args.agent else "build"
            print(f"  Running opencode (model={args.model}, agent={effective_agent}, timeout={args.timeout}s) ...")
            log_file = workspace_dir / instance_id / f"{instance_id}_opencode.log"
            session_export_file = workspace_dir / instance_id / f"{instance_id}_session.json"
            print(f"  Log → {log_file}")
            t0 = time.time()
            result, session_export, workspace_export = run_opencode_docker_v2(
                inference_image, prompt, args.model, args.timeout, instance_id, args.agent,
                opencode_config=OPCODE_CONFIG_PATH,
                check_agent=True,
                check_skill=not args.skip_skill_check,
            )
            elapsed = time.time() - t0
            print(f"\n  --- opencode finished (exit={result.returncode}, elapsed={elapsed:.0f}s) ---")

            # Save session export if available
            if session_export and session_export.exists():
                shutil.move(str(session_export), str(session_export_file))
                print(f"  Session export → {session_export_file}")

            # Extract .ai-test-workspace if available
            if workspace_export and workspace_export.exists():
                ws_dir = workspace_dir / instance_id / "ai-test-workspace"
                with tarfile.open(workspace_export, "r:gz") as tar:
                    tar.extractall(path=str(workspace_dir / instance_id))
                workspace_export.unlink()
                print(f"  .ai-test-workspace → {ws_dir}")

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

            # Validate: agent must explicitly output <patch> tag
            is_valid, reason = is_valid_patch(result.stdout, model_patch)
            if not is_valid:
                print(f"  WARNING: {reason}")
                if model_patch:
                    # Show what git diff captured (for debugging)
                    files = get_patch_files(model_patch)
                    print(f"  git diff captured: {', '.join(files)}")
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
        print(f"   python inference/inference_opencode_docker_skill.py \\")
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
