# SWT-Bench Inference

Generates reproducing test predictions for SWT-Bench instances using code agents.

Using mirror is STRONGLY RECOMMENDED for users having trouble connecting to `huggingface`, `docker`, `conda`, `github`, etc.

Supports two execution modes:
- **Local mode** (default): opencode runs directly on the host, isolated via venv + PATH sanitization
- **Docker mode** (`--docker`): opencode runs inside a Docker container with true sandboxing

## Prerequisite

Requires Docker.
Follow these steps to set up the evaluation environment.

```bash
git clone git@github.com:sgallon-rin/swt-bench.git
cd swt-bench
python -m venv .venv  # altenatively, use uv: `uv venv .venv`
source .venv/bin/activate
pip install -e .  # try `uv pip install -e .` instead if error occurs
```

Test your installation by running:
```bash
export DOCKER_HOST=unix:///$HOME/.docker/run/docker.sock  # required on MacOS; untested on other platforms
python -m src.main \
    --predictions_path gold \
    --max_workers 1 \
    --instance_ids sympy__sympy-20590 \
    --run_id validate-gold
```

On the first run, it will download and build docker images necessary for evaluation.

Running evaluation is resource-intensive. For each sample, a docker image of size around ~3GB will be built.


## Quick Start

Download SWT-Bench-Lite dataset

```bash
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com  # optional, use mirror
hf download eth-sri/SWT-bench_Lite_bm25_27k_zsb --repo-type dataset
```

Run Infernece

```bash
# LOCAL MODE — Run 5 instances with opencode (default)
python inference/inference_opencode.py --max-instances 5 --model deepseek/deepseek-v4-flash

# DOCKER MODE — Run 5 instances with opencode inside Docker (recommended for clean env)
DOCKER_HOST=unix:///$HOME/.docker/run/docker.sock \
python inference/inference_opencode.py --max-instances 5 --model deepseek/deepseek-v4-flash --docker

# Run with a specific agent
python inference/inference_opencode.py --max-instances 5 --model deepseek/deepseek-v4-flash --agent agent_name

# Run specific instances
python inference/inference_opencode.py --instance-ids sympy__sympy-20590 django__django-10087

# Run with a specific run_id (for resume)
python inference/inference_opencode.py --max-instances 5 --model deepseek/deepseek-v4-flash --run-id exp1

# Resume a previous run (skips completed, retries failed)
python inference/inference_opencode.py --model deepseek/deepseek-v4-flash --run-id exp1
```

## Run ID & Resume

Each run has a `run_id` that determines the output file and enables resume. The output filename includes both model and agent for clear identification.

| Scenario | Command | Behavior |
|----------|---------|----------|
| New run (auto ID) | `python inference_opencode.py --max-instances 5` | Uses timestamp as run_id, creates new file |
| New run (custom ID) | `python inference_opencode.py --run-id exp1` | Output saved to `opencode__<model>__<agent>__exp1.jsonl` |
| Resume | `python inference_opencode.py --run-id exp1` | Skips completed instances, retries failed ones |
| Independent runs | `--run-id exp1` vs `--run-id exp2` | Different output files, no interference |

When `--run-id` is not specified, an auto-generated timestamp is used as run_id. The script will print the run_id at startup:

```
Run ID  : 20260615_150838
  (auto-generated; to resume: --run-id 20260615_150838)
Output  : predictions/opencode__deepseek_deepseek-v4-flash__build__20260615_150838.jsonl
Status  : NEW RUN
```

To resume a previous run, use the same run_id:

```
Run ID  : exp1
Output  : predictions/opencode__deepseek_deepseek-v4-flash__dt-generation__exp1.jsonl
Status  : RESUME (3 instance(s) already completed)
```

After completion, the script prints the next steps automatically:

```
Done. Predictions → predictions/opencode__deepseek-v4-flash__dt-generation__exp1.jsonl

📋 Next steps:

1️⃣  Evaluate:
   python -m src.main \
       --dataset_name princeton-nlp/SWE-bench_Lite \
       --predictions_path predictions/opencode__deepseek-v4-flash__dt-generation__exp1.jsonl \
       --filter_swt \
       --instance_ids $(cat dataset/swt_bench_lite_subset.txt | tr '\n' ' ') \
       --run_id exp1

2️⃣  Report (after evaluation):
   python -m src.report_custom run_instance_swt_logs/exp1/opencode__deepseek-v4-flash --total 51
```

### Model & Agent in Filename

The output filename format is: `opencode__<model>__<agent>__<run-id>.jsonl`

- `<model>`: Model name with `/` replaced by `_`
- `<agent>`: Agent name, or `build` if not specified
- `<run-id>`: Custom run_id or auto-generated timestamp

This ensures different model/agent combinations produce separate output files, even with the same run_id.

## How It Works

### Local Mode

```
 For each SWT-Bench instance:
 ┌─ 1. Clone the target repo (cached in repo-cache/)
 ├─ 2. Checkout the buggy base commit in an isolated workspace
 ├─ 3. Create venv + install deps on host
 ├─ 4. Run the code agent with the task prompt
 │      Agent explores code, writes a reproducing test, verifies it FAILS
 ├─ 5. Extract the git diff from the agent's output
 ├─ 6. Append {instance_id, model_name_or_path, model_patch} to JSONL
 └─ 7. Cleanup workspace
```

### Docker Mode (`--docker`)

```
 For each SWT-Bench instance:
 ┌─ 1. Clone the target repo (cached in repo-cache/)          ← Host
 ├─ 2. Checkout the buggy base commit in an isolated workspace  ← Host
 ├─ 3. Write issue file to workspace                            ← Host
 ├─ 4. Spin up Docker container with workspace mounted          ← Docker boundary
 ├─ 5. Create venv + pip install -e . inside container          ← Container
 ├─ 6. Run opencode inside container                            ← Container
 │      Agent explores code, writes a reproducing test, verifies it FAILS
 ├─ 7. Container exits, all modifications stay in workspace     ← Docker boundary
 ├─ 8. Extract the git diff from the agent's output             ← Host
 ├─ 9. Append {instance_id, model_name_or_path, model_patch} to JSONL  ← Host
 └─10. Cleanup workspace                                        ← Host
```

Docker provides **true operating-system-level isolation**:
- opencode cannot write files outside the mounted workspace
- pip installs are sandboxed inside the container
- No PATH/env sanitization needed (Docker provides a clean environment)
- Workspace isolation is guaranteed by Docker, not Python-level checks

## Docker-Based Inference

### Architecture

The Docker inference follows the same three-layer image hierarchy as evaluation, with a
separate naming prefix (`swt-inf.*`) to avoid conflicts with evaluation images (`exec.*`):

```
swt-inf.base.x86_64:latest       ← Ubuntu 22.04 + opencode + git + python
        ↓                           Built once, shared by all instances
swt-inf.env.{repo}.{hash}:latest  ← (planned) Pre-installed repo dependencies
        ↓                           Shared by instances of the same repo
Container (per-instance)          ← Mount workspace, run opencode, destroy
```

Currently only the base image layer is implemented. Env-layer images with
pre-installed repo dependencies can be added later for faster per-instance setup.

### Prerequisites

The inference Docker image requires opencode CLI credentials. The container
automatically mounts your host's opencode config:

| Host path | Container path | Purpose |
|-----------|---------------|---------|
| `~/.config/opencode/` | `/home/agent/.config/opencode/` | opencode configuration |
| `~/.local/share/opencode/` | `/home/agent/.local/share/opencode/` | API auth credentials |

Make sure opencode is authenticated on the host before running with `--docker`:

```bash
opencode providers login  # or configure via opencode auth
```

### Image Build

The base image is built automatically on the first `--docker` run:

```bash
python inference/inference_opencode.py --max-instances 1 --model deepseek/deepseek-v4-flash --docker
```

Output:
```
  Building/ensuring inference Docker image ...
  Inference Docker image: swt-inf.base.x86_64:latest
  All opencode execution will run inside Docker containers
```

Subsequent runs skip the build check. The image is ~870 MB.

To force rebuild:
```bash
docker buildx build --platform linux/amd64 --tag swt-inf.base.x86_64:latest \
    --file inference/Dockerfile.base --load inference/
```

### macOS Setup

On macOS, Docker Desktop must be running and the socket path must be exported:

```bash
export DOCKER_HOST=unix:///$HOME/.docker/run/docker.sock
```

Add this to your shell profile (`.zshrc` / `.bashrc`) for convenience.

### Comparison: Local vs Docker

| Aspect | Local mode | Docker mode |
|--------|-----------|-------------|
| File isolation | PATH + venv hack | Docker boundary |
| pip install safety | Host venv (leaks) | Container sandbox |
| opencode isolation | Best-effort | Guaranteed |
| Reproduce version mismatch | Host Python | Fixed Python 3.10 |
| Host pollution risk | Moderate | None |
| Speed | Faster (no container overhead) | ~2-5s container startup |

## Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `eth-sri/SWT-bench_Lite_bm25_27k_zsb` | HuggingFace dataset name |
| `--model` | `deepseek/deepseek-v4-flash` | Model name (passed to the agent) |
| `--agent` | `None` | opencode agent to use (default: none) |
| `--docker` | `False` | Run opencode inside a Docker container for sandboxed inference |
| `--run-id` | `None` | Run identifier for resume. Same ID resumes (skip success, retry fail). Different IDs are independent. |
| `--max-instances` | all | Limit number of instances to process |
| `--instance-ids` | — | Run only these specific IDs (overrides max) |
| `--output` | `predictions/opencode__<model>__<agent>__<run-id-or-timestamp>.jsonl` | Output JSONL path |
| `--timeout` | `None` | Timeout per instance (seconds, default: no limit) |
| `--repo-cache` | `repo-cache/` | Directory to cache cloned repos |
| `--workspace-dir` | `tmp/workspaces/` | Temporary per-instance workspaces |

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `GIT_BASE_URL` | Base URL for git clones (default `https://github.com`). Set to a mirror if GitHub is unreachable. |
| `HF_TOKEN` | HuggingFace API token for dataset access |

## Evaluate Results

```bash
DOCKER_HOST=unix:///$HOME/.docker/run/docker.sock \
python -m src.main \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path predictions/opencode__deepseek-v4-flash.jsonl \
    --filter_swt \
    --run_id opencode_exp1
```


```bash
# evaluate with gold
DOCKER_HOST=unix:///$HOME/.docker/run/docker.sock \
python -m src.main \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path gold \
    --filter_swt \
    --instance_ids django__django-15202 django__django-17087 sphinx-doc__sphinx-8721 \
    --max_workers 1 \
    --run_id validate-gold
```


## Prompts

Prompt templates are in `prompts/`. The default prompt for opencode is `prompts/opencode.txt`, adapted from `prompts/SWE-Agent-plus.txt`.

## File Structure

```
inference/
├── README.md
├── inference_opencode.py    # Main inference script (supports --docker)
├── docker_utils.py          # Docker image build/cache + container helpers
├── Dockerfile.base          # Inference base image (opencode + python + git)
└── __init__.py

prompts/
├── SWE-Agent.txt            # Original SWE-Agent base prompt
├── SWE-Agent-plus.txt       # Original SWE-Agent plus prompt
└── opencode.txt             # Adapted prompt for opencode (based on plus)
```


## Subset Evaluation

A curated subset of 51 instances (5 per repo, seed=42) is available for quick validation. Unstable instances (appearing in any filter file) are automatically excluded during selection.

### Generate the subset (optional, already included)

```bash
python dataset/select_subset.py
```

Output: `dataset/swt_bench_lite_subset.txt` (51 instances after filtering)

### Run inference on the subset

```bash
# Local mode
python inference/inference_opencode.py \
    --instance-ids $(cat dataset/swt_bench_lite_subset.txt | tr '\n' ' ') \
    --model deepseek/deepseek-v4-flash \
    --agent dt-generation

# Docker mode (recommended)
DOCKER_HOST=unix:///$HOME/.docker/run/docker.sock \
python inference/inference_opencode.py \
    --instance-ids $(cat dataset/swt_bench_lite_subset.txt | tr '\n' ' ') \
    --model deepseek/deepseek-v4-flash \
    --agent dt-generation \
    --docker
```

### Evaluate the subset

```bash
# Optional: run on gold first to build reusable docker images for evaluation
python -m src.main \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path gold \
    --filter_swt \
    --instance_ids $(cat dataset/swt_bench_lite_subset.txt | tr '\n' ' ') \
    --run_id gold-subset

python -m src.main \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path predictions/your_predictions.jsonl \
    --filter_swt \
    --instance_ids $(cat dataset/swt_bench_lite_subset.txt | tr '\n' ' ') \
    --run_id opencode_subset
```

### Generate Report for the Subset

The standard `src/report.py` uses the full dataset size (~300) as denominator, which is incorrect for subsets. Use `src/report_custom.py` instead:

```bash
# Basic report (replace 51 with your actual subset size)
python -m src.report_custom run_instance_swt_logs/opencode_subset/your_model --total 51

# With Coverage Delta (requires a gold run for comparison)
python -m src.report_custom run_instance_swt_logs/opencode_subset/your_model \
    --total 51 \
    --gold-run-id opencode_subset \
    --name "My Model"

# LaTeX format
python -m src.report_custom run_instance_swt_logs/opencode_subset/your_model \
    --total 51 \
    --gold-run-id opencode_subset \
    --format latex
```

Example output:

|------------------------------------|---------------|
| Method                             | My Model      |
| Applicability (W)                  | 100.0         |
| Success Rate (S)                   | 88.2          |
| F->X                               | 98.0          |
| F->P                               | 88.2          |
| P->P                               | 19.6          |
| Coverage Delta (Δᵃˡˡ)              | 60.0          |
| Coverage Delta Resolved (Δᔆ)       | 60.4          |
| Coverage Delta Unresolved (Δⁿᵒᵗ ᔆ) | 50.0          |

| Flag | Description |
|------|-------------|
| `--total` | Total number of instances in your subset (used as denominator) |
| `--gold-run-id` | Run ID of the gold evaluation for coverage delta comparison |
| `--name` | Display name for the method (default: extracted from path) |
| `--format` | Output format: `github` (markdown table) or `latex` |
