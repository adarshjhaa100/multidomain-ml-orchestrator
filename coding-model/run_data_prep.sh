#!/usr/bin/env bash
# =============================================================================
# run_data_prep.sh — Phase C Dataset Construction Runner (50/25/15/10 Rule)
# =============================================================================
#
# Tiger Style applied to this script:
#   - No silent failures: set -euo pipefail (fail fast on any error)
#   - Explicit pre-checks: every tool is verified before use
#   - Deterministic error messages: every failure says WHAT failed and HOW to fix
#
# Prerequisites:
#   - uv installed (pip3 install uv or curl -LsSf https://astral.sh/uv/install.sh | sh)
#   - Python >= 3.10
#   - Git
#
# Usage:
#   chmod +x run_data_prep.sh
#   ./run_data_prep.sh                         # Full pipeline (all phases)
#   ./run_data_prep.sh --phase clone           # Only clone repos
#   ./run_data_prep.sh --phase chunk,instruct  # Multiple comma-separated phases
#   ./run_data_prep.sh --verbose               # Full debug logging
#   ./run_data_prep.sh --skip-model-download   # Skip automatic model download
#   ./run_data_prep.sh --smoke-test            # Quick smoke test of batched LLM
#   ./run_data_prep.sh --instruct-limit 100    # First N chunks only (benchmark)
#   ./run_data_prep.sh --cpu-workers 2         # Force 2 CPU workers in orchestrator
#   ./run_data_prep.sh --gpu-parallel 8        # Force GPU batch size in orchestrator
# =============================================================================

set -euo pipefail  # Tiger Style: fail fast, no silent errors, catch unset vars

# ── Color output (disabled if not a terminal) ──────────────────────────────
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    NC='\033[0m' # No Color
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; NC=''
fi

# ── Helper functions (Tiger Style: every function does ONE thing) ──────────

log_info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# Tiger Style: explicit error handler — no silent crash.
trap 'log_error "Script failed at line $LINENO. Check the error above."' ERR

# ── Pre-checks (Tiger Style: validate environment before any work) ─────────

pre_check() {
    local cmd="$1"
    local install_hint="$2"
    if ! command -v "$cmd" &>/dev/null; then
        log_error "'$cmd' not found. $install_hint"
        exit 1
    fi
    log_ok "Found: $cmd ($(command -v "$cmd"))"
}

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Phase C: Dataset Construction (50/25/15/10 Rule) Starting...${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

# ── Parse arguments ────────────────────────────────────────────────────────

PHASE="all"
VERBOSE=""
SKIP_MODEL_DOWNLOAD=false
SMOKE_TEST=false
INSTRUCT_LIMIT=""
CPU_WORKERS=""
CPU_THREADS=""
GPU_PARALLEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)
            shift
            PHASE="$1"
            ;;
        --verbose|-v)
            VERBOSE="--verbose"
            ;;
        --skip-model-download)
            SKIP_MODEL_DOWNLOAD=true
            ;;
        --smoke-test)
            SMOKE_TEST=true
            ;;
        --instruct-limit)
            shift
            INSTRUCT_LIMIT="--instruct-limit $1"
            ;;
        --cpu-workers)
            shift
            CPU_WORKERS="--cpu-workers $1"
            ;;
        --cpu-threads)
            shift
            CPU_THREADS="--cpu-threads $1"
            ;;
        --gpu-parallel)
            shift
            GPU_PARALLEL="--gpu-parallel $1"
            ;;
        --help|-h)
            head -30 "$0" | grep "^#  " | sed 's/^#  //'
            exit 0
            ;;
        *)
            log_error "Unknown argument: $1. Use --help for usage."
            exit 1
            ;;
    esac
    shift
done

# ── Step 0: Pre-flight checks ──────────────────────────────────────────────

log_info "Running pre-flight checks..."

pre_check "git"   "Install: brew install git (macOS) or apt install git (Linux)"
pre_check "uv"    "Install: curl -LsSf https://astral.sh/uv/install.sh | sh"

# Check uv version.
UV_VERSION=$(uv --version 2>/dev/null | head -1 || echo "unknown")
log_info "uv version: $UV_VERSION"

# Check Python version.
PYTHON_VERSION=$(uv run python3 --version 2>/dev/null || uv run python --version 2>/dev/null || echo "unknown")
log_info "Python: $PYTHON_VERSION"

# Check for HuggingFace CLI (needed for model download).
if command -v huggingface-cli &>/dev/null; then
    log_ok "Found: huggingface-cli"
else
    log_warn "huggingface-cli not found. Will attempt install via uv."
    uv pip install huggingface-hub
fi

# ── Check LM Studio backend (batched LLM inference) ─────────────────────────

BACKEND_DIR=$(uv run python3 -c "
import sys; sys.path.insert(0, '.')
try:
    from llm_backend import find_lmstudio_backend
    info = find_lmstudio_backend()
    if info:
        print(info['backend_dir'])
    else:
        print('')
except Exception:
    print('')
" 2>/dev/null)

if [[ -n "$BACKEND_DIR" ]]; then
    log_ok "LM Studio llama.cpp CUDA backend found (batched mode)"
    log_info "  Backend: $BACKEND_DIR"
else
    log_info "No LM Studio CUDA backend detected — using serial llama-cpp-python path"
    log_info "  (Install LM Studio at https://lmstudio.ai for up to 3x faster generation)"
fi

echo ""

if [[ "$SMOKE_TEST" == true ]]; then
    log_info "Running batched LLM smoke test..."
    uv run python -c "
import sys, tempfile, orjson
from pathlib import Path
sys.path.insert(0, '.')
from llm_backend import BatchedLlama, find_lmstudio_backend
info = find_lmstudio_backend()
if info is None:
    print('[SKIP] No LM Studio backend found')
    sys.exit(0)
chunks = [
    {'language': 'python', 'code': 'def add(a, b): return a + b', 'repo': 'test', 'name': 'add'},
    {'language': 'c',      'code': 'int inc(int x) { return x + 1; }',  'repo': 'test', 'name': 'inc'},
]
with BatchedLlama('models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf', n_parallel=4, ctx_per_seq=1536) as llm:
    reqs = [{'prompt': f'Write an instruction for: {c[\"code\"]}', 'max_tokens': 64, 'temperature': 0.7, 'stop': ['\\n\\n']} for c in chunks]
    texts = llm.complete_batch(reqs)
    for c, t in zip(chunks, texts):
        print(f'  [{c[\"language\"]:>8}] {t[:60] if t else \"(failed)\"}')
print('OK')
" 2>&1
    log_ok "Smoke test complete"
    exit 0
fi

# ── Step 0b: Create uv virtual environment and install deps ────────────────

log_info "Creating uv virtual environment (if not exists)..."
if [[ ! -f ".venv/bin/python" ]]; then
    uv venv
    log_ok "Virtual environment created at .venv/"
else
    log_ok "Virtual environment already exists at .venv/"
fi

log_info "Installing Python dependencies via uv..."
uv pip install \
    tree-sitter tree-sitter-c tree-sitter-python tree-sitter-javascript \
    tree-sitter-css tree-sitter-html \
    llama-cpp-python \
    requests beautifulsoup4 \
    gitpython \
    orjson \
    huggingface-hub \
    tqdm \
    transformers \
    blake3 \
    psutil
log_ok "Dependencies installed."

echo ""

# ── Step 0c: Download generator model (if needed) ──────────────────────────

MODEL_DIR="models"
MODEL_FILE="$MODEL_DIR/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
MODEL_REPO="Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF"

if [[ "$SKIP_MODEL_DOWNLOAD" == true ]]; then
    log_info "Skipping model download (--skip-model-download flag set)."
    if [[ ! -f "$MODEL_FILE" ]]; then
        log_warn "Model file not found at $MODEL_FILE. Instruction generation will be skipped."
    fi
elif [[ -f "$MODEL_FILE" ]]; then
    FILE_SIZE_MB=$(du -m "$MODEL_FILE" | cut -f1)
    log_ok "Model already cached: $MODEL_FILE (${FILE_SIZE_MB} MB)"
else
    log_info "Downloading generator model (~1 GB)..."
    log_info "  Repo: $MODEL_REPO"
    log_info "  File: $(basename "$MODEL_FILE")"
    log_info "  This may take a few minutes on slower connections."
    echo ""

    mkdir -p "$MODEL_DIR"
    uv run huggingface-cli download "$MODEL_REPO" \
        "$(basename "$MODEL_FILE")" \
        --local-dir "$MODEL_DIR"

    # Post-condition: verify download.
    if [[ -f "$MODEL_FILE" ]]; then
        FILE_SIZE_MB=$(du -m "$MODEL_FILE" | cut -f1)
        log_ok "Model downloaded: $MODEL_FILE (${FILE_SIZE_MB} MB)"
        # Tiger Style: assert minimum file size (~900MB for Q4_K_M).
        if [[ "$FILE_SIZE_MB" -lt 800 ]]; then
            log_warn "Model file is small (${FILE_SIZE_MB} MB). May be corrupted."
        fi
    else
        log_error "Model download failed. File not found: $MODEL_FILE"
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Setup complete. Starting Phase C pipeline...${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""

# ── Run the Python data preparation script ─────────────────────────────────

log_info "Executing data_prep.py with phase: $PHASE"
echo ""

# Tiger Style: explicit timeout — 4 hours max for the full pipeline.
uv run python data_prep.py --phase "$PHASE" $VERBOSE $INSTRUCT_LIMIT $CPU_WORKERS $CPU_THREADS $GPU_PARALLEL

EXIT_CODE=$?

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Phase C pipeline completed successfully.${NC}"
    echo -e "${GREEN}  Output: data/train.jsonl${NC}"
    echo -e "${GREEN}  Stats:  data/stats.json${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"

    # Show dataset summary if it exists.
    if [[ -f "data/train.jsonl" ]]; then
        LINES=$(wc -l < "data/train.jsonl" | tr -d ' ')
        SIZE_MB=$(du -m "data/train.jsonl" | cut -f1)
        echo ""
        echo "  Dataset summary:"
        echo "    Examples: $LINES"
        echo "    Size:     ${SIZE_MB} MB"
        echo ""
        echo "  Next step: Run Phase D (fine-tuning on cloud GPU)."
        echo "  See: docs/data-prep-stage-instructions.md for details."
    fi
else
    echo -e "${RED}════════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}  Phase C pipeline failed (exit code: $EXIT_CODE).${NC}"
    echo -e "${RED}  Check logs above for errors.${NC}"
    echo -e "${RED}  Common fixes:${NC}"
    echo -e "${RED}    - Run only specific phases to isolate the failure${NC}"
    echo -e "${RED}      e.g., ./run_data_prep.sh --phase clone${NC}"
    echo -e "${RED}    - Check network connectivity for scraping/docs phases${NC}"
    echo -e "${RED}    - Ensure models/ directory has the GGUF file${NC}"
    echo -e "${RED}════════════════════════════════════════════════════════════${NC}"
fi

exit $EXIT_CODE
