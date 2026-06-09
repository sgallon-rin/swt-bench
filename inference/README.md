# SWT-Bench Inference

Generates reproducing test predictions for SWT-Bench instances using code agents.

## Quick Start

```bash
# Run 5 instances with opencode
python inference/inference_opencode.py --max-instances 5 --model deepseek-ve-flash

# Run specific instances
python inference/inference_opencode.py --instance-ids sympy__sympy-20590 django__django-10087
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
| `--model` | `deepseek-ve-flash` | Model name (passed to the agent) |
| `--max-instances` | all | Limit number of instances to process |
| `--instance-ids` | — | Run only these specific IDs (overrides max) |
| `--output` | `predictions/opencode__<model>.jsonl` | Output JSONL path |
| `--timeout` | `1800` | Timeout per instance (seconds) |
| `--repo-cache` | `repo-cache/` | Directory to cache cloned repos |
| `--workspace-dir` | `tmp/workspaces/` | Temporary per-instance workspaces |

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `GIT_BASE_URL` | Base URL for git clones (default `https://github.com`). Set to a mirror if GitHub is unreachable. |
| `HF_TOKEN` | HuggingFace API token for dataset access |

## Evaluate Results

```bash
python -m src.main \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path predictions/opencode__deepseek-ve-flash.jsonl \
    --filter_swt \
    --run_id opencode_exp1
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
