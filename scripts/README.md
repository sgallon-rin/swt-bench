# SWT-Bench Experiment Scripts

Unified scripts for running SWT-Bench experiments.

## Quick Start

```bash
# Default (skill enabled)
./scripts/run_experiment.sh infer --run-id 20260701_v1 --subset
./scripts/run_experiment.sh eval  --run-id 20260701_v1 --subset
./scripts/run_experiment.sh report --run-id 20260701_v1
```

Each `infer` and `eval` command prints elapsed time on completion, e.g.:
```
[OK]    Elapsed: 2h 15m 32s
```

## Commands

| Command | Description |
|---------|-------------|
| `infer` | Run inference — generates reproducing test predictions |
| `eval`  | Run evaluation harness — tests predictions against gold patches |
| `report`| Generate summary report — Applicability, Success Rate, F→X, F→P, P→P |
| `sync`  | Sync results from remote server |
| `help`  | Show usage help |

## Options

### Common (all commands)

| Option | Default | Description |
|--------|---------|-------------|
| `--run-id ID` | *(required)* | Unique run identifier |
| `--model MODEL` | `deepseek/deepseek-v4-flash` | LLM model |
| `--dataset NAME` | `eth-sri/SWT-bench_Lite_bm25_27k_zsb` | Inference dataset (HuggingFace) |
| `--subset` | — | Use the 51-instance subset from `dataset/swt_bench_lite_subset.txt` |
| `--instance-ids IDS...` | — | Instance IDs (space-separated, no quotes needed) |
| `--max-instances N` | all | Limit to N instances |
| `--timeout SECS` | none | Timeout per instance |
| `--no-skill` | off | Disable skill (default: skill **enabled**) |

### Eval only

| Option | Default | Description |
|--------|---------|-------------|
| `--max-workers N` | 4 | Parallel evaluation workers |

### Sync only

| Option | Default | Description |
|--------|---------|-------------|
| `--remote ADDR` | `vboxuser@zb` | Remote server SSH address |
| `--workspace PATH` | `/home/vboxuser/workspace/sjl/swt-bench` | Remote project root |
| `--stage STAGE` | `all` | `infer` / `eval` / `all` — which stage's artifacts to sync |
| `--dry-run` | off | Preview rsync without transferring |

## Experiment Workflow

The tool uses **skill by default**. Use `--no-skill` for the one-time baseline run.

### Skill runs (iterative development)

```bash
./scripts/run_experiment.sh infer --run-id 20260701_v1 --subset
./scripts/run_experiment.sh eval  --run-id 20260701_v1 --subset
./scripts/run_experiment.sh report --run-id 20260701_v1
```

### Baseline run (one-time, no skill)

```bash
./scripts/run_experiment.sh infer --run-id 20260701_baseline --subset --no-skill
./scripts/run_experiment.sh eval  --run-id 20260701_baseline --subset --no-skill
./scripts/run_experiment.sh report --run-id 20260701_baseline --no-skill
```

### Specify instance-ids

```bash
./scripts/run_experiment.sh infer --run-id test_run --instance-ids django__django-15202
./scripts/run_experiment.sh eval  --run-id test_run --instance-ids django__django-15202
./scripts/run_experiment.sh report --run-id test_run
```

### Custom dataset

```bash
./scripts/run_experiment.sh infer \
  --run-id 20260701_full \
  --dataset eth-sri/SWT-bench_bm25_27k_zsb \
  --max-instances 100
```

### Resume a run (same `--run-id` skips completed instances)

```bash
# First run (may have failures)
./scripts/run_experiment.sh infer --run-id 20260701_v1 --subset

# Re-run with same ID — skips successes, retries failures
./scripts/run_experiment.sh infer --run-id 20260701_v1 --subset
```

## Sync from Remote Server

### Stage-based sync

The `--stage` flag lets you sync only the artifacts from a specific experiment phase:

| Stage | Syncs | When to use |
|-------|-------|-------------|
| `infer` | workspace logs + predictions | After inference on server |
| `eval` | evaluation logs + evaluation results | After evaluation on server |
| `all` (default) | everything | One-shot pull of all artifacts |

```bash
# Default (skill) — sync everything
./scripts/run_experiment.sh sync --run-id 20260701_v1

# Baseline (no skill)
./scripts/run_experiment.sh sync --run-id 20260701_baseline --no-skill

# Sync only infer artifacts (after inference, before eval)
./scripts/run_experiment.sh sync --run-id 20260701_v1 --stage infer

# Sync only eval artifacts (after eval on server)
./scripts/run_experiment.sh sync --run-id 20260701_v1 --stage eval

# Custom server
./scripts/run_experiment.sh sync \
  --run-id 20260701_v1 \
  --remote user@myserver \
  --workspace /path/to/swt-bench

# Dry run (preview without transferring)
./scripts/run_experiment.sh sync --run-id 20260701_v1 --stage infer --dry-run
```

### Sync artifacts by stage

**`--stage infer`** downloads:

| Source (remote) | Destination (local) | Contents |
|-----------------|---------------------|----------|
| `tmp/workspaces/${RUN_ID}/` | `tmp/workspaces/${RUN_ID}/` | `*.log`, `*.json`, `.ai-test-workspace/` |
| `predictions/opencode__*.jsonl` | `predictions/` | Auto-derived filename |
| `tmp/log/${RUN_ID}.log` | `tmp/log/${RUN_ID}.log` | Full inference output log |

**`--stage eval`** downloads:

| Source (remote) | Destination (local) | Contents |
|-----------------|---------------------|----------|
| `run_instance_swt_logs/${RUN_ID}/` | `run_instance_swt_logs/${RUN_ID}/` | Per-instance eval logs, e.g. `opencode__deepseek_deepseek-v4-flash__skill/` |
| `evaluation_results/` | `evaluation_results/` | JSON report files |

### Predictions filename auto-derivation

The sync command computes the correct filename from `--no-skill`:

| Flag | Predictions filename |
|------|---------------------|
| (default) | `opencode__deepseek_deepseek-v4-flash__skill__build__${RUN_ID}.jsonl` |
| `--no-skill` | `opencode__deepseek_deepseek-v4-flash__build__${RUN_ID}.jsonl` |

### Manual sync commands (if you prefer)

If you need fine-grained control, here are the underlying commands:

```bash
# Configure your server
REMOTE="user@hostname"
REMOTE_WS="/path/to/swt-bench"
RUN_ID=20260701_v1

# ── Infer stage ──

# Workspace logs (dry run first)
rsync -avzPmn \
  --include='*/' --include='*.log' --include='*.json' \
  --include='.ai-test-workspace/***' --exclude='*' \
  ${REMOTE}:${REMOTE_WS}/tmp/workspaces/${RUN_ID}/ \
  ./tmp/workspaces/${RUN_ID}/

# Workspace logs (actual)
rsync -avzPm \
  --include='*/' --include='*.log' --include='*.json' \
  --include='.ai-test-workspace/***' --exclude='*' \
  ${REMOTE}:${REMOTE_WS}/tmp/workspaces/${RUN_ID}/ \
  ./tmp/workspaces/${RUN_ID}/

# Predictions (default, with skill)
scp ${REMOTE}:${REMOTE_WS}/predictions/opencode__deepseek_deepseek-v4-flash__skill__build__${RUN_ID}.jsonl \
  ./predictions/

# Predictions (baseline, no skill)
scp ${REMOTE}:${REMOTE_WS}/predictions/opencode__deepseek_deepseek-v4-flash__build__${RUN_ID}.jsonl \
  ./predictions/

# Run log
scp ${REMOTE}:${REMOTE_WS}/tmp/log/${RUN_ID}.log \
  ./tmp/log/${RUN_ID}.log

# ── Eval stage ──

# Evaluation logs (includes agent subdirectories like opencode__deepseek_deepseek-v4-flash__skill/)
rsync -avzP ${REMOTE}:${REMOTE_WS}/run_instance_swt_logs/${RUN_ID}/ \
  ./run_instance_swt_logs/${RUN_ID}/

# Evaluation results
rsync -avzP ${REMOTE}:${REMOTE_WS}/evaluation_results/ \
  ./evaluation_results/
```

## Retry Failed Instances

After inference, check the summary for failed IDs, then retry manually:

```bash
# From the inference output summary:
#   Failed IDs: astropy__astropy-12907 sympy__sympy-15678

# Retry (default, with skill)
python inference/inference_opencode_docker_skill.py \
  --instance-ids astropy__astropy-12907 sympy__sympy-15678 \
  --model deepseek/deepseek-v4-flash \
  --agent build \
  --run-id 20260701_v1

# Retry (baseline, no skill)
python inference/inference_opencode_docker.py \
  --instance-ids astropy__astropy-12907 \
  --model deepseek/deepseek-v4-flash \
  --agent build \
  --run-id 20260701_baseline
```

## Report with Coverage Delta

To include coverage delta (Δ), you need a gold evaluation run for comparison:

```bash
# First, run gold evaluation
python -m src.main \
  --dataset_name princeton-nlp/SWE-bench_Lite \
  --predictions_path gold \
  --filter_swt \
  --instance_ids $(cat dataset/swt_bench_lite_subset.txt | tr '\n' ' ') \
  --run-id gold_subset

# Then report with delta (total auto-detected from report.json files)
python -m src.report_custom \
  run_instance_swt_logs/20260701_v1/opencode__deepseek_deepseek-v4-flash__skill \
  --gold-run-id gold_subset \
  --name "v1"

python -m src.report_custom \
  run_instance_swt_logs/20260701_baseline/opencode__deepseek_deepseek-v4-flash \
  --gold-run-id gold_subset \
  --name "Baseline"
```

## Docker Setup

### macOS

```bash
export DOCKER_HOST=unix:///$HOME/.docker/run/docker.sock
```

Add to `~/.zshrc` for persistence.

### HuggingFace Mirror (China)

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Git Mirror

```bash
export GIT_BASE_URL=https://github.com  # or your mirror
```

## Report Metrics

| Metric | Description |
|--------|-------------|
| **Applicability (W)** | % of instances where the agent produced a valid patch |
| **Success Rate (S)** | % of instances that are resolved (test passes after patch, fails before) |
| **F→X** | % of instances where the test correctly fails before the patch |
| **F→P** | % of instances where the test fails before patch and passes after (ideal) |
| **P→P** | % where the test passes both before and after (test doesn't reproduce bug) |
