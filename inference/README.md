# SWT-Bench Inference

Generates reproducing test predictions for SWT-Bench instances using code agents.

Using mirror is STRONGLY RECOMMENDED for users having trouble connecting to `huggingface`, `docker`, `conda`, `github`, etc.

Currently, inferencing with opencode is in local environment without sandboxing.

## Prerequisite

Requires Docker.
Follow these steps to set up the evaluation environment.

```shell
git clone git@github.com:sgallon-rin/swt-bench.git
cd swt-bench
python -m venv .venv
source .venv/bin/activate
pip install -e .  # try `uv pip install -e .` instead if error occurs
```

Test your installation by running:
```shell
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

```bash
# Run 5 instances with opencode
python inference/inference_opencode.py --max-instances 5 --model deepseek/deepseek-v4-flash

# Run with a specific agent
python inference/inference_opencode.py --max-instances 5 --model deepseek/deepseek-v4-flash --agent agent_name

# Run specific instances
python inference/inference_opencode.py --instance-ids sympy__sympy-20590 django__django-10087


python inference/inference_opencode.py --max-instances 1 --model deepseek/deepseek-v4-flash --agent dt-generation
```

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
| `--max-instances` | all | Limit number of instances to process |
| `--instance-ids` | — | Run only these specific IDs (overrides max) |
| `--output` | `predictions/opencode__<model>_<timestamp>.jsonl` | Output JSONL path |
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


```
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
