#!/usr/bin/env bash
# =============================================================================
# run_experiment.sh — SWT-Bench Experiment Runner
# =============================================================================
# Commands:
#   infer      Run inference on instances
#   eval       Run evaluation harness on predictions
#   report     Generate summary report from evaluation logs
#   sync       Sync results from remote server
#   help       Show help message
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ─── Defaults ───────────────────────────────────────────────────────────────
DEFAULT_MODEL="deepseek/deepseek-v4-flash"
DEFAULT_AGENT="build"
DEFAULT_SUBSET_FILE="dataset/swt_bench_lite_subset.txt"
DEFAULT_DATASET="eth-sri/SWT-bench_Lite_bm25_27k_zsb"
DEFAULT_EVAL_DATASET="princeton-nlp/SWE-bench_Lite"
DEFAULT_MAX_WORKERS=4
DEFAULT_REMOTE="vboxuser@zb"
DEFAULT_REMOTE_WS="/home/vboxuser/workspace/sjl/swt-bench"
DEFAULT_STAGE="all"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

elapsed_time() {
    local secs=$1
    if (( secs >= 3600 )); then
        printf "%dh %dm %ds" $((secs/3600)) $((secs%3600/60)) $((secs%60))
    elif (( secs >= 60 )); then
        printf "%dm %ds" $((secs/60)) $((secs%60))
    else
        printf "%ds" $secs
    fi
}

# ─── Usage ──────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $0 <command> [options]

Commands:
  infer      Run inference on instances
  eval       Run evaluation harness
  report     Generate summary report
  sync       Sync results from remote server
  help       Show this help

Common Options:
  --run-id ID          Run identifier (required)
  --model MODEL        Model (default: $DEFAULT_MODEL)
  --dataset NAME       Inference dataset (default: $DEFAULT_DATASET)
  --subset             Use the 51-instance subset (../dataset/swt_bench_lite_subset.txt)
  --instance-ids IDS...  Instance IDs (space-separated, no quotes needed)
  --max-instances N    Limit to N instances
  --timeout SECS       Timeout per instance (seconds)
  --no-skill           Disable skill (default: skill enabled)

Eval Options:
  --max-workers N      Parallel workers (default: $DEFAULT_MAX_WORKERS)

Sync Options:
  --remote ADDR        Remote server address (default: $DEFAULT_REMOTE)
  --workspace PATH     Remote workspace path (default: $DEFAULT_REMOTE_WS)
  --stage STAGE        infer | eval | all (default: $DEFAULT_STAGE)
  --dry-run            Preview rsync without transferring

Examples:
  # Default (skill enabled)
  $0 infer --run-id 20260701_v1 --subset
  $0 eval  --run-id 20260701_v1 --subset
  $0 report --run-id 20260701_v1
  $0 sync  --run-id 20260701_v1

  # Baseline (no skill, one-time run)
  $0 infer --run-id 20260701_baseline --subset --no-skill
  $0 eval  --run-id 20260701_baseline --subset --no-skill
  $0 report --run-id 20260701_baseline --no-skill
  $0 sync  --run-id 20260701_baseline --no-skill

  # Sync by stage
  $0 sync --run-id 20260701_v1 --stage infer
  $0 sync --run-id 20260701_v1 --stage eval

  # Custom server
  $0 sync --run-id 20260701_v1 --remote user@myserver --workspace /path/to/swt-bench
EOF
    exit 0
}

# ─── Parse ──────────────────────────────────────────────────────────────────
CMD="${1:-help}"
shift || true

RUN_ID=""
MODEL="$DEFAULT_MODEL"
DATASET="$DEFAULT_DATASET"
SUBSET=false
INSTANCE_IDS=""
MAX_INSTANCES=""
TIMEOUT=""
SKILL=true
MAX_WORKERS="$DEFAULT_MAX_WORKERS"
REMOTE="$DEFAULT_REMOTE"
REMOTE_WS="$DEFAULT_REMOTE_WS"
STAGE="$DEFAULT_STAGE"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id)        RUN_ID="$2"; shift 2 ;;
        --model)         MODEL="$2"; shift 2 ;;
        --dataset)       DATASET="$2"; shift 2 ;;
        --subset)        SUBSET=true; shift ;;
        --instance-ids)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                INSTANCE_IDS="${INSTANCE_IDS}${INSTANCE_IDS:+ }$1"
                shift
            done
            ;;
        --max-instances) MAX_INSTANCES="$2"; shift 2 ;;
        --timeout)       TIMEOUT="$2"; shift 2 ;;
        --no-skill)      SKILL=false; shift ;;
        --max-workers)   MAX_WORKERS="$2"; shift 2 ;;
        --remote)        REMOTE="$2"; shift 2 ;;
        --workspace)     REMOTE_WS="$2"; shift 2 ;;
        --stage)         STAGE="$2"; shift 2 ;;
        --dry-run)       DRY_RUN=true; shift ;;
        --help|-h)       usage ;;
        *)               log_error "Unknown option: $1"; usage ;;
    esac
done

# ─── Helpers ────────────────────────────────────────────────────────────────
resolve_instance_ids() {
    if [[ -n "$INSTANCE_IDS" ]]; then
        echo "$INSTANCE_IDS"
    elif [[ "$SUBSET" == true ]]; then
        cat "$DEFAULT_SUBSET_FILE" | tr '\n' ' '
    else
        echo ""
    fi
}

resolve_inference_script() {
    if [[ "$SKILL" == true ]]; then
        echo "inference/inference_opencode_docker_skill.py"
    else
        echo "inference/inference_opencode_docker.py"
    fi
}

model_safe() {
    echo "$MODEL" | tr '/' '_'
}

predictions_path() {
    local m_safe
    m_safe=$(model_safe)
    if [[ "$SKILL" == true ]]; then
        echo "$PROJECT_ROOT/predictions/opencode__${m_safe}__skill__${DEFAULT_AGENT}__${RUN_ID}.jsonl"
    else
        echo "$PROJECT_ROOT/predictions/opencode__${m_safe}__${DEFAULT_AGENT}__${RUN_ID}.jsonl"
    fi
}

remote_predictions_path() {
    local m_safe
    m_safe=$(model_safe)
    if [[ "$SKILL" == true ]]; then
        echo "${REMOTE_WS}/predictions/opencode__${m_safe}__skill__${DEFAULT_AGENT}__${RUN_ID}.jsonl"
    else
        echo "${REMOTE_WS}/predictions/opencode__${m_safe}__${DEFAULT_AGENT}__${RUN_ID}.jsonl"
    fi
}

setup_docker_host() {
    if [[ -z "${DOCKER_HOST:-}" ]] && [[ "$(uname)" == "Darwin" ]]; then
        export DOCKER_HOST="unix:///$HOME/.docker/run/docker.sock"
    fi
    log_info "DOCKER_HOST=${DOCKER_HOST:-<not set>}"
}

# ─── Commands ───────────────────────────────────────────────────────────────
cmd_infer() {
    if [[ -z "$RUN_ID" ]]; then
        log_error "--run-id is required"
        exit 1
    fi

    local script inst_ids
    script=$(resolve_inference_script)
    inst_ids=$(resolve_instance_ids)
    setup_docker_host

    local mode="skill"
    [[ "$SKILL" == false ]] && mode="baseline"

    log_info "Inference ($mode)"
    log_info "  Run ID    : $RUN_ID"
    log_info "  Model     : $MODEL"
    log_info "  Dataset   : $DATASET"
    log_info "  Agent     : $DEFAULT_AGENT"
    log_info "  Script    : $script"
    log_info "  Instances : ${inst_ids:-(all)}"
    [[ -n "$MAX_INSTANCES" ]] && log_info "  Max       : $MAX_INSTANCES"
    [[ -n "$TIMEOUT" ]]       && log_info "  Timeout   : ${TIMEOUT}s"

    local cmd=(python "$script" --dataset "$DATASET" --model "$MODEL" --agent "$DEFAULT_AGENT" --run-id "$RUN_ID")
    [[ -n "$inst_ids" ]]      && cmd+=(--instance-ids $inst_ids)
    [[ -n "$MAX_INSTANCES" ]] && cmd+=(--max-instances "$MAX_INSTANCES")
    [[ -n "$TIMEOUT" ]]       && cmd+=(--timeout "$TIMEOUT")

    mkdir -p tmp/log
    SECONDS=0
    "${cmd[@]}" 2>&1 | tee -a "tmp/log/${RUN_ID}.log"

    log_ok "Predictions → $(predictions_path)"
    log_ok "Elapsed: $(elapsed_time $SECONDS)"
}

cmd_eval() {
    if [[ -z "$RUN_ID" ]]; then
        log_error "--run-id is required"
        exit 1
    fi

    local inst_ids pred_file
    inst_ids=$(resolve_instance_ids)
    pred_file=$(predictions_path)

    if [[ ! -f "$pred_file" ]]; then
        log_error "Predictions not found: $pred_file"
        log_error "Run inference first: $0 infer --run-id $RUN_ID $([ "$SKILL" == false ] && echo '--no-skill')"
        exit 1
    fi

    setup_docker_host

    log_info "Evaluation"
    log_info "  Run ID      : $RUN_ID"
    log_info "  Predictions : $pred_file"
    log_info "  Max workers : $MAX_WORKERS"

    local cmd=(
        python -m src.main
        --dataset_name "$DEFAULT_EVAL_DATASET"
        --predictions_path "$pred_file"
        --filter_swt
        --max_workers "$MAX_WORKERS"
        --run_id "$RUN_ID"
    )
    [[ -n "$inst_ids" ]] && cmd+=(--instance_ids $inst_ids)

    SECONDS=0
    "${cmd[@]}"

    log_ok "Results → evaluation_results/"
    log_ok "Logs    → run_instance_swt_logs/${RUN_ID}/"
    log_ok "Elapsed: $(elapsed_time $SECONDS)"
}

cmd_report() {
    if [[ -z "$RUN_ID" ]]; then
        log_error "--run-id is required"
        exit 1
    fi

    local m_safe model_name log_dir
    m_safe=$(model_safe)
    model_name="opencode__${m_safe}"
    [[ "$SKILL" == true ]] && model_name="${model_name}__skill"

    log_dir="run_instance_swt_logs/${RUN_ID}/${model_name}"

    if [[ ! -d "$log_dir" ]]; then
        log_error "Log dir not found: $log_dir"
        log_error "Run evaluation first: $0 eval --run-id $RUN_ID"
        exit 1
    fi

    log_info "Report"
    log_info "  Log dir : $log_dir"

    python -m src.report_custom "$log_dir"
}

cmd_sync() {
    if [[ -z "$RUN_ID" ]]; then
        log_error "--run-id is required"
        exit 1
    fi

    case "$STAGE" in
        infer|eval|all) ;;
        *)
            log_error "Invalid stage: $STAGE (must be infer, eval, or all)"
            exit 1
            ;;
    esac

    local pred_file remote_pred
    pred_file=$(predictions_path)
    remote_pred=$(remote_predictions_path)

    local rsync_flags=(-avzPm)
    [[ "$DRY_RUN" == true ]] && rsync_flags=(-avzPmn)

    local do_infer=false
    local do_eval=false
    case "$STAGE" in
        infer) do_infer=true ;;
        eval)  do_eval=true ;;
        all)   do_infer=true; do_eval=true ;;
    esac

    log_info "Sync from ${REMOTE}:${REMOTE_WS}"
    log_info "  Run ID    : $RUN_ID"
    log_info "  Stage     : $STAGE"
    log_info "  Mode      : $([ "$DRY_RUN" == true ] && echo "dry-run" || echo "actual")"
    echo ""

    SECONDS=0

    # ── Infer stage: workspace logs + predictions + run log ──
    if [[ "$do_infer" == true ]]; then
        local ws_src="${REMOTE}:${REMOTE_WS}/tmp/workspaces/${RUN_ID}/"
        local ws_dst="$PROJECT_ROOT/tmp/workspaces/${RUN_ID}/"
        log_info "[1/3] Workspace logs..."
        log_info "  → ${ws_src}"
        log_info "  → ${ws_dst}"
        rsync "${rsync_flags[@]}" \
            --include='*/' \
            --include='*.log' \
            --include='*.json' \
            --include='.ai-test-workspace/***' \
            --exclude='*' \
            "$ws_src" \
            "$ws_dst" || \
            log_warn "No workspace logs found on server"
        echo ""

        log_info "[2/3] Predictions..."
        log_info "  → ${remote_pred}"
        log_info "  → ${pred_file}"
        mkdir -p "$PROJECT_ROOT/predictions"
        scp "${REMOTE}:${remote_pred}" "$pred_file" 2>/dev/null || \
            log_warn "Could not fetch predictions (may not exist yet)"
        echo ""

        local log_src="${REMOTE}:${REMOTE_WS}/tmp/log/${RUN_ID}.log"
        local log_dst="$PROJECT_ROOT/tmp/log/${RUN_ID}.log"
        log_info "[3/3] Run log..."
        log_info "  → ${log_src}"
        log_info "  → ${log_dst}"
        mkdir -p "$PROJECT_ROOT/tmp/log"
        scp "$log_src" "$log_dst" 2>/dev/null || \
            log_warn "Could not fetch run log (may not exist yet)"
        echo ""
    fi

    # ── Eval stage: evaluation logs + evaluation results ──
    if [[ "$do_eval" == true ]]; then
        local eval_log_src="${REMOTE}:${REMOTE_WS}/run_instance_swt_logs/${RUN_ID}/"
        local eval_log_dst="$PROJECT_ROOT/run_instance_swt_logs/${RUN_ID}/"
        log_info "[1/2] Evaluation logs..."
        log_info "  → ${eval_log_src}"
        log_info "  → ${eval_log_dst}"
        rsync -avzP "$eval_log_src" \
            "$eval_log_dst" 2>/dev/null || \
            log_warn "No evaluation logs found on server (run eval first)"
        echo ""

        local eval_res_src="${REMOTE}:${REMOTE_WS}/evaluation_results/"
        local eval_res_dst="$PROJECT_ROOT/evaluation_results/"
        log_info "[2/2] Evaluation results..."
        log_info "  → ${eval_res_src}"
        log_info "  → ${eval_res_dst}"
        rsync -avzP "$eval_res_src" \
            "$eval_res_dst" 2>/dev/null || \
            log_warn "No evaluation results found on server"
        echo ""
    fi

    log_ok "Sync complete."
    log_ok "Elapsed: $(elapsed_time $SECONDS)"
}

# ─── Dispatch ───────────────────────────────────────────────────────────────
case "$CMD" in
    infer)   cmd_infer ;;
    eval)    cmd_eval ;;
    report)  cmd_report ;;
    sync)    cmd_sync ;;
    help)    usage ;;
    *)       log_error "Unknown command: $CMD"; usage ;;
esac
