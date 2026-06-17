# SWT-Bench Inference

Generates reproducing test predictions for SWT-Bench instances using code agents.

Using mirror is STRONGLY RECOMMENDED for users having trouble connecting to `huggingface`, `docker`, `conda`, `github`, etc.

Currently, inferencing with opencode is in local environment without sandboxing.

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
# Run 5 instances with opencode
python inference/inference_opencode.py --max-instances 5 --model deepseek/deepseek-v4-flash

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

### Model & Agent in Filename

The output filename format is: `opencode__<model>__<agent>__<run-id>.jsonl`

- `<model>`: Model name with `/` replaced by `_`
- `<agent>`: Agent name, or `build` if not specified
- `<run-id>`: Custom run_id or auto-generated timestamp

This ensures different model/agent combinations produce separate output files, even with the same run_id.

## How It Works

```
 For each SWT-Bench instance:
 ┌─ 1. Clone the target repo (cached in repo-cache/)
 ├─ 2. Checkout the buggy base commit in an isolated workspace
 ├─ 3. Run the code agent with the task prompt
 │      Agent explores code, writes a reproducing test, verifies it FAILS
 ├─ 4. Extract the git diff from the agent's output
 ├─ 5. Append {instance_id, model_name_or_path, model_patch} to JSONL
 └─ 6. Cleanup workspace
```

Resume-safe: re-running skips instances already in the output JSONL.

## Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `eth-sri/SWT-bench_Lite_bm25_27k_zsb` | HuggingFace dataset name |
| `--model` | `deepseek/deepseek-v4-flash` | Model name (passed to the agent) |
| `--agent` | `None` | opencode agent to use (default: none) |
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
└── inference_opencode.py

prompts/
├── SWE-Agent.txt           # Original SWE-Agent base prompt
├── SWE-Agent-plus.txt      # Original SWE-Agent plus prompt
└── opencode.txt            # Adapted prompt for opencode (based on plus)
```


## Subset Evaluation

A curated subset of 52 instances (5 per repo, seed=42) is available for quick validation.

### Generate the subset (optional, already included)

```bash
python dataset/select_subset.py
```

Output: `dataset/swt_bench_lite_subset.txt`

### Run inference on the subset

```bash
python inference/inference_opencode.py \
    --instance-ids $(cat dataset/swt_bench_lite_subset.txt | tr '\n' ' ') \
    --model deepseek/deepseek-v4-flash \
    --agent dt-generation
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
# Basic report (replace 52 with your actual subset size)
python -m src.report_custom run_instance_swt_logs/opencode_subset/your_model --total 52

# With Coverage Delta (requires a gold run for comparison)
python -m src.report_custom run_instance_swt_logs/opencode_subset/your_model \
    --total 52 \
    --gold-run-id opencode_subset \
    --name "My Model"

# LaTeX format
python -m src.report_custom run_instance_swt_logs/opencode_subset/your_model \
    --total 52 \
    --gold-run-id opencode_subset \
    --format latex
```

Example output:

|------------------------------------|---------------|
| Method                             | My Model      |
| Applicability (W)                  | 98.1          |
| Success Rate (S)                   | 86.5          |
| F->X                               | 96.2          |
| F->P                               | 86.5          |
| P->P                               | 19.2          |
| Coverage Delta (Δᵃˡˡ)              | 60.0          |
| Coverage Delta Resolved (Δᔆ)       | 60.4          |
| Coverage Delta Unresolved (Δⁿᵒᵗ ᔆ) | 50.0          |

| Flag | Description |
|------|-------------|
| `--total` | Total number of instances in your subset (used as denominator) |
| `--gold-run-id` | Run ID of the gold evaluation for coverage delta comparison |
| `--name` | Display name for the method (default: extracted from path) |
| `--format` | Output format: `github` (markdown table) or `latex` |
