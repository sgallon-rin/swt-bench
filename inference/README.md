# SWT-Bench Inference

Generates reproducing test predictions for SWT-Bench instances using code agents.

Using mirror is STRONGLY RECOMMENDED for users having trouble connecting to `huggingface`, `docker`, `conda`, `github`, etc.

## Prerequisite

Requires Docker.
Follow these steps to set up the evaluation environment.

```bash
git clone git@github.com:sgallon-rin/swt-bench.git
cd swt-bench
python -m venv .venv  # alternatively, use uv: `uv venv .venv`
source .venv/bin/activate
pip install -e .  # try `uv pip install -e .` instead if error occurs
```

Test your installation by running:
```bash
export DOCKER_HOST=unix:///$HOME/.docker/run/docker.sock  # required on MacOS
python -m src.main \
    --predictions_path gold \
    --max_workers 1 \
    --instance_ids sympy__sympy-20590 \
    --run_id validate-gold
```

On the first run, it will download and build docker images necessary for evaluation. Running evaluation is resource-intensive. For each sample, a docker image of size around ~3GB will be built.

## Quick Start

### Download Dataset

```bash
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com  # optional, use mirror
hf download eth-sri/SWT-bench_Lite_bm25_27k_zsb --repo-type dataset
```

### Run Inference

The inference script uses pre-built evaluation images from `src/`, adding opencode on top. This avoids rebuilding environments at runtime.

```bash
export RUN_ID=20260624_docker_subset_dt
export RUN_AGENT=dt-generation  # or: build

python inference/inference_opencode_docker.py \
    --instance-ids $(cat dataset/swt_bench_lite_subset.txt | tr '\n' ' ') \
    --model deepseek/deepseek-v4-flash \
    --agent ${RUN_AGENT} \
    --run-id $RUN_ID 2>&1 | tee -a tmp/log/${RUN_ID}.log
```

### Evaluate

```bash
python -m src.main \
       --dataset_name princeton-nlp/SWE-bench_Lite \
       --predictions_path predictions/opencode__deepseek_deepseek-v4-flash__${RUN_AGENT}__${RUN_ID}.jsonl \
       --filter_swt \
       --instance_ids $(cat dataset/swt_bench_lite_subset.txt | tr '\n' ' ') \
       --run_id ${RUN_ID}
```

### Generate Report

```bash
python -m src.report_custom run_instance_swt_logs/${RUN_ID}/opencode__deepseek_deepseek-v4-flash --total 51
```

## Architecture

```
src instance image (exec.eval.{arch}.{env_hash}.{instance_hash})
    ↓ already contains: conda env + deps + project code (/testbed)
    ↓ add opencode (~10s)
inference image (swt-inf.eval.{arch}.{env_hash}.{instance_hash})
    ↓ run: --user nonroot, WORKDIR /testbed
    ↓ mount: ~/.opencode → /home/nonroot/.opencode
    ↓ execute: conda activate testbed && opencode run ...
```

## Run ID & Resume

Each run has a `run_id` that determines the output file and enables resume. The output filename includes both model and agent for clear identification.

| Scenario | Command | Behavior |
|----------|---------|----------|
| New run (auto ID) | `python inference/inference_opencode_docker.py --max-instances 5` | Uses timestamp as run_id, creates new file |
| New run (custom ID) | `python inference/inference_opencode_docker.py --run-id exp1` | Output saved to `opencode__<model>__<agent>__exp1.jsonl` |
| Resume | `python inference/inference_opencode_docker.py --run-id exp1` | Skips completed instances, retries failed ones |
| Independent runs | `--run-id exp1` vs `--run-id exp2` | Different output files, no interference |

When `--run-id` is not specified, an auto-generated timestamp is used. The script prints the run_id at startup:

```
Run ID  : 20260624_094708
  (auto-generated; to resume: --run-id 20260624_094708)
Output  : predictions/opencode__deepseek_deepseek-v4-flash__build__20260624_094708.jsonl
Status  : NEW RUN
```

To resume a previous run, use the same run_id:

```
Run ID  : exp1
Output  : predictions/opencode__deepseek_deepseek-v4-flash__dt-generation__exp1.jsonl
Status  : RESUME (3 instance(s) already completed)
```

### Model & Agent in Filename

The output filename format is: `opencode__<model>__<agent>__<run-id>.jsonl`

- `<model>`: Model name with `/` replaced by `_`
- `<agent>`: Agent name, or `build` if not specified
- `<run-id>`: Custom run_id or auto-generated timestamp

## macOS Setup

On macOS, Docker Desktop must be running and the socket path must be exported:

```bash
export DOCKER_HOST=unix:///$HOME/.docker/run/docker.sock
```

Add this to your shell profile (`.zshrc` / `.bashrc`) for convenience.

## Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `eth-sri/SWT-bench_Lite_bm25_27k_zsb` | HuggingFace dataset name |
| `--model` | `deepseek/deepseek-v4-flash` | Model name (passed to the agent) |
| `--agent` | `None` | opencode agent to use (default: none, agent name or `build`) |
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

## Subset Evaluation

A curated subset of 51 instances (5 per repo, seed=42) is available for quick validation. Unstable instances (appearing in any filter file) are automatically excluded during selection.

### Generate the subset (optional, already included)

```bash
python dataset/select_subset.py
```

Output: `dataset/swt_bench_lite_subset.txt` (51 instances after filtering)

### Run Inference on the Subset

```bash
export RUN_ID=20260624_docker_subset_dt
export RUN_AGENT=dt-generation

python inference/inference_opencode_docker.py \
    --instance-ids $(cat dataset/swt_bench_lite_subset.txt | tr '\n' ' ') \
    --model deepseek/deepseek-v4-flash \
    --agent ${RUN_AGENT} \
    --run-id $RUN_ID 2>&1 | tee -a tmp/log/${RUN_ID}.log
```

### Evaluate the Subset

```bash
python -m src.main \
       --dataset_name princeton-nlp/SWE-bench_Lite \
       --predictions_path predictions/opencode__deepseek_deepseek-v4-flash__${RUN_AGENT}__${RUN_ID}.jsonl \
       --filter_swt \
       --instance_ids $(cat dataset/swt_bench_lite_subset.txt | tr '\n' ' ') \
       --run_id ${RUN_ID}
```

### Generate Report for the Subset

The standard `src/report.py` uses the full dataset size (~300) as denominator, which is incorrect for subsets. Use `src/report_custom.py` instead:

```bash
# Basic report (replace 51 with your actual subset size)
python -m src.report_custom run_instance_swt_logs/${RUN_ID}/opencode__deepseek_deepseek-v4-flash --total 51

# With Coverage Delta (requires a gold run for comparison)
python -m src.report_custom run_instance_swt_logs/${RUN_ID}/opencode__deepseek_deepseek-v4-flash \
    --total 51 \
    --gold-run-id ${RUN_ID} \
    --name "My Model"

# LaTeX format
python -m src.report_custom run_instance_swt_logs/${RUN_ID}/opencode__deepseek_deepseek-v4-flash \
    --total 51 \
    --gold-run-id ${RUN_ID} \
    --format latex
```

Example output:

|-------------------|------------------------------|
| Method            | 20260624_docker_subset_build |
| Applicability (W) | 100.0                        |
| Success Rate (S)  | 56.9                         |
| F->X              | 90.2                         |
| F->P              | 64.7                         |
| P->P              | 13.7                         |

| Flag | Description |
|------|-------------|
| `--total` | Total number of instances in your subset (used as denominator) |
| `--gold-run-id` | Run ID of the gold evaluation for coverage delta comparison |
| `--name` | Display name for the method (default: extracted from path) |
| `--format` | Output format: `github` (markdown table) or `latex` |

## Prompts

Prompt templates are in `prompts/`. The default prompt for opencode is `prompts/opencode.txt`.

## File Structure

```
inference/
├── README.md
├── inference_opencode_docker.py    # Main inference script (pre-built eval images)
├── docker_utils.py                 # Docker image helpers
├── Dockerfile.base                 # Inference base image
├── opencode.json                   # opencode configuration
└── __init__.py
```
