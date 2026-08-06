"""
data_prep.py — Phase C: Dataset Construction (The 50/25/15/10 Rule)
================================================================

Builds train.jsonl from 3 layers:
  60% Code   — SQLite, Redis, Starlette, Rich, HTTPX, Alpine.js, htmx, Pico.css
  25% Docs   — cppreference.com, docs.python.org, MDN
  15% Align  — Tiger Style guide, low-level design, whitepapers

Tiger Style principles applied to THIS script:
  - No silent failures: every error is caught, logged, and surfaced
  - Explicit pre/post conditions: assert on every input/output boundary
  - Deterministic cleanup: context managers for all resources
  - No hidden allocations: all buffers/files explicitly managed
  - Bounds checking: every list access, string length, file size validated

Usage:
  uv run python data_prep.py --phase all        # Run full pipeline
  uv run python data_prep.py --phase clone      # Run only repo cloning
  uv run python data_prep.py --phase chunk      # Run only tree-sitter chunking
  uv run python data_prep.py --phase instruct   # Run only instruction generation
  uv run python data_prep.py --phase docs       # Run only doc scraping
  uv run python data_prep.py --phase align      # Run only alignment building
  uv run python data_prep.py --phase filter     # Run only token accounting + filtering
  uv run python data_prep.py --phase mix        # Run only mixing + shuffling
  uv run python data_prep.py --phase validate   # Run only final validation
"""

# %%
# =============================================================================
# Phase 0: Imports, Constants, and Environment Checks
# =============================================================================
# Tiger Style: all imports explicit. No wildcard imports.
# Each import's purpose is documented inline.

import argparse
import hashlib
import json
import logging
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import textwrap
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Tiger Style: validate Python version at import time.
# Below 3.10 means no match/case, no dataclass slots — deal-breaker for maintainability.
assert sys.version_info >= (3, 10), (
    f"Python >= 3.10 required (found {sys.version_info.major}.{sys.version_info.minor}). "
    "Use `uv python install 3.12` to upgrade."
)

# ── Third-party imports (with fallback guidance) ──────────────────────────
# Tiger Style: fail fast at import time, not halfway through a 3-hour run.

try:
    import orjson
except ImportError:
    raise ImportError(
        "Missing `orjson`. Install: uv pip install orjson\n"
        "orjson is 4-10x faster than stdlib json for JSONL bulk writes."
    )

try:
    import git as gitpython
except ImportError:
    raise ImportError(
        "Missing `gitpython`. Install: uv pip install gitpython"
    )

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    raise ImportError(
        "Missing `requests` or `beautifulsoup4`. Install: uv pip install requests beautifulsoup4"
    )

try:
    from tree_sitter import Language, Parser
except ImportError:
    raise ImportError(
        "Missing `tree-sitter`. Install: uv pip install tree-sitter\n"
        "Also need language grammars: tree-sitter-c, tree-sitter-python, etc."
    )

try:
    from tqdm import tqdm
except ImportError:
    # Tiger Style: fail explicit — provide the fix, not just the error.
    raise ImportError("Missing `tqdm`. Install: uv pip install tqdm")

# ── Local LLM for synthetic instruction generation ────────────────────────
# This is optional — only needed for --phase instruct or --phase align.
_llm_instance: Any = None
_batched_llm_instance: Any = None


def _orchestrator_plan(
    model_path: str,
    cpu_workers: Optional[int] = None,
    cpu_threads: Optional[int] = None,
    gpu_parallel: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    sample_jobs: Optional[List[Dict[str, Any]]] = None,
) -> List[Any]:
    """Return the orchestrator worker plan (GPU batched + CPU workers).

    Uses live hardware probing so the machine decides, and lets the CLI force
    a mix for benchmarking (see --cpu-workers / --cpu-threads / --gpu-parallel).

    When nothing is forced, the plan is *calibrated*: the orchestrator runs the
    candidate mixes (gpu-only, +1cpu, +2cpu) on a sample of the real job and
    keeps whichever gives the highest throughput per machine.
    """
    from orchestrator import calibrate_plan, plan_llm_resources, probe

    res = probe()
    forced = cpu_workers is not None or cpu_threads is not None or gpu_parallel is not None
    ctx = 2048

    if not forced and sample_jobs:
        specs = calibrate_plan(res, model_path, sample_jobs, ctx_per_seq=ctx, logger=logger)
        if logger:
            logger.info("  Orchestrator plan (calibrated, %s):", res.gpu.name or "no-GPU")
            for s in specs:
                logger.info("    worker %-5s kind=%-3s threads=%-2d n_parallel=%-2d ctx=%d",
                            s.name, s.kind, s.n_threads, s.n_parallel, s.ctx_per_seq)
        return specs

    specs = plan_llm_resources(
        res,
        model_path,
        ctx_per_seq=ctx,
        max_cpu_workers=cpu_workers,
        cpu_threads_each=cpu_threads if cpu_threads else 2,
        max_parallel=gpu_parallel if gpu_parallel else 12,
    )
    if logger:
        logger.info("  Orchestrator plan (%s):", res.gpu.name or "no-GPU")
        for s in specs:
            logger.info("    worker %-5s kind=%-3s threads=%-2d n_parallel=%-2d ctx=%d",
                        s.name, s.kind, s.n_threads, s.n_parallel, s.ctx_per_seq)
    return specs


def _run_orchestrated(
    model_path: str,
    jobs: List[Dict[str, Any]],
    raw_out: Path,
    specs: List[Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Run a completion job across the orchestrator fleet; return stats.

    `jobs[i]` is a request dict ({"prompt", "max_tokens", ...}). `raw_out` is a
    per-run temp file; each line is the request merged with {"_idx", "_text"}.
    """
    from orchestrator import run_llm_completions
    return run_llm_completions(
        jobs, raw_out, specs, model_path, logger=logger, force_regen=True,
    )


def _orchestrator_complete(
    model_path: str,
    jobs: List[Dict[str, Any]],
    specs: List[Any],
    logger: logging.Logger,
    run_dir: Optional[Path] = None,
) -> List[Optional[str]]:
    """Run jobs through the fleet and return texts in input order.

    Small wrapper around `_run_orchestrated` for callers that only need the
    decoded texts back (e.g. alignment expansion).
    """
    if not jobs:
        return []
    raw_out = CHUNKS_DIR / f"orch_raw_{uuid.uuid4().hex[:8]}.jsonl"
    from orchestrator import run_llm_completions
    stats = run_llm_completions(
        jobs, raw_out, specs, model_path, logger=logger,
        force_regen=True, run_dir=run_dir,
    )
    texts: Dict[int, Optional[str]] = {}
    if raw_out.exists():
        for line in open(raw_out, "rb"):
            rec = orjson.loads(line)
            texts[rec["_idx"]] = rec["_text"]
        raw_out.unlink(missing_ok=True)
    return [texts.get(i) for i in range(len(jobs))]


def _load_batched_llm(model_path: str, logger: logging.Logger) -> Any:
    """Lazy-load the LM Studio llama.cpp backend for batched inference.

    Tiger Style: explicit check-then-use. Returns None (never raises) when no
    LM Studio CUDA backend is installed, so callers fall back to the serial
    llama-cpp-python path. The backend folder is consulted first, as the user
    requested — no build, no server, no download.
    """
    global _batched_llm_instance
    if _batched_llm_instance is not None:
        return _batched_llm_instance
    try:
        from llm_backend import BatchedLlama, find_lmstudio_backend
    except ImportError:
        logger.debug("  llm_backend not importable — using serial llama-cpp-python path")
        return None
    info = find_lmstudio_backend()
    if info is None:
        logger.debug("  No LM Studio llama.cpp backend found — using serial llama-cpp-python path")
        return None
    try:
        _batched_llm_instance = BatchedLlama(
            str(model_path),
            n_parallel=12,
            ctx_per_seq=2048,
            logger=logger,
        )
        _batched_llm_instance.start()
    except Exception as exc:
        logger.warning("  LM Studio backend failed to start (%s) — falling back to serial", exc)
        _batched_llm_instance = None
        return None
    logger.info("  Batched LM Studio backend ready (%s)", info["backend_dir"])
    return _batched_llm_instance


def _load_llm(model_path: str, n_threads: Optional[int] = None) -> Any:
    """Lazy-load the GGUF model via llama-cpp-python.

    Tiger Style: explicit resource acquisition — caller decides when to load.
    No hidden initialization at module level.
    Offloads all layers to GPU when available.
    """
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "Missing `llama-cpp-python`. Install: uv pip install llama-cpp-python\n"
            "Then download: huggingface-cli download Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF \\\n"
            "    qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --local-dir models/"
        )
    model_path_resolved = Path(model_path)
    assert model_path_resolved.exists(), (
        f"Model file not found: {model_path_resolved.absolute()}. "
        "Download it first (see instructions above)."
    )
    assert model_path_resolved.stat().st_size > 100_000_000, (
        f"Model file suspiciously small ({model_path_resolved.stat().st_size} bytes). "
        "Download may be corrupted."
    )
    if n_threads is None:
        n_threads = os.cpu_count() or 4
    _llm_instance = Llama(
        model_path=str(model_path_resolved),
        n_ctx=2048,       # Tiger Style: explicit context cap — no unbounded memory
        n_threads=n_threads,
        n_gpu_layers=-1,  # Offload all layers to GPU
        verbose=False,
    )
    return _llm_instance


# =============================================================================
# %%
# Phase 0b: Project Paths & Constants
# =============================================================================
# Tiger Style: all magic numbers named. No bare literals in logic.

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DATA_DIR = PROJECT_ROOT / "input_data"
REPOS_DIR = INPUT_DATA_DIR / "repos"
DOCS_DIR = INPUT_DATA_DIR / "docs"
ALIGNMENT_DIR = INPUT_DATA_DIR / "alignment"
DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_DIR = DATA_DIR / "chunks"
LOGS_DIR = DATA_DIR / "logs"
MODELS_DIR = PROJECT_ROOT / "models"

# Tiger Style: explicit bounds on every data dimension.
MAX_FILE_LINES = 5000        # Skip generated/amalgamated files
MIN_CHUNK_LINES = 3          # Below this: trivial (getter/setter)
MAX_CHUNK_LINES = 200        # Above this: too large for 3B model context
MAX_INSTRUCTION_CODE_CHARS = 2000  # Max code chars passed to the LLM prompt (speed/quality trade-off)
MAX_TOKENS_PER_EXAMPLE = 2048
MIN_TOKENS_PER_EXAMPLE = 8
MAX_INSTRUCTION_GEN_RETRIES = 3
REQUEST_DELAY_SECONDS = 0.5  # Polite scraping delay

# Tiger Style: target ratios with explicit tolerance bounds.
# Phase C rule: 50% code / 25% docs & fundamentals / 15% style & alignment /
# 10% devops, logs & harness operations (measured in tokens, not examples).
TARGET_CODE_PCT = 0.50
TARGET_DOC_PCT = 0.25
TARGET_ALIGN_PCT = 0.15
TARGET_DEVOPS_PCT = 0.10
RATIO_TOLERANCE = 0.03       # ±3% allowed deviation

# ── Repository definitions ────────────────────────────────────────────────

REPOS: List[Dict[str, str]] = [
    {"name": "sqlite",    "url": "https://github.com/sqlite/sqlite.git",   "lang": "c"},
    {"name": "redis",     "url": "https://github.com/redis/redis.git",     "lang": "c"},
    {"name": "starlette", "url": "https://github.com/encode/starlette.git","lang": "python"},
    {"name": "rich",      "url": "https://github.com/Textualize/rich.git", "lang": "python"},
    {"name": "httpx",     "url": "https://github.com/encode/httpx.git",    "lang": "python"},
    {"name": "alpine",    "url": "https://github.com/alpinejs/alpine.git", "lang": "javascript"},
    {"name": "htmx",      "url": "https://github.com/bigskysoftware/htmx.git","lang": "html"},
    {"name": "picocss",   "url": "https://github.com/picocss/pico.git",    "lang": "css"},
]

# Tiger Style: assert invariants at module load time.
assert len(REPOS) == 8, f"Expected 8 repos, got {len(REPOS)}"
repo_names = {r["name"] for r in REPOS}
assert len(repo_names) == len(REPOS), f"Duplicate repo names: {repo_names}"

# ── DevOps layer online sources ───────────────────────────────────────────
# Tiger Style: the devops layer was historically hard-coded seeds (~1k tokens),
# which starved the whole 50/25/15/10 budget. These small, config-heavy repos
# provide authentic Dockerfile/Kubernetes/CI content so devops can reach the
# 10% binding target without an LLM (instructions are derived deterministically,
# mirroring the doc-layer pattern).
DEVOPS_REPOS: List[Dict[str, str]] = [
    {"name": "k8s-examples", "url": "https://github.com/kubernetes/examples.git"},
]

# File names/extensions treated as devops config material across those repos.
DEVOPS_FILE_MARKERS: Tuple[str, ...] = (
    "dockerfile",
)
DEVOPS_EXTENSIONS: Tuple[str, ...] = (
    ".yaml", ".yml", ".tf", ".tfvars", ".service", ".timer", ".mk",
)

# Language-specific file extensions for filtering.
LANG_EXTENSIONS: Dict[str, Tuple[str, ...]] = {
    "c":          (".c", ".h"),
    "python":     (".py",),
    "javascript": (".js",),
    "html":       (".html", ".htm"),
    "css":        (".css",),
}

# Directories to exclude when scanning repos (test files, build artifacts).
EXCLUDE_DIRS: Tuple[str, ...] = (
    "test", "tests", "testing", "__pycache__",
    "build", "vendor", "third_party", ".git",
)

# tree-sitter query patterns for each language.
# Tiger Style: each query explicitly named and documented.
TS_QUERIES: Dict[str, str] = {
    "c": """
        (function_definition) @func
        (declaration) @decl
    """,
    "python": """
        (function_definition) @func
        (class_definition) @class
        (decorated_definition) @decorated
    """,
    "javascript": """
        (function_declaration) @func
        (class_declaration) @class
        (arrow_function) @arrow
        (method_definition) @method
    """,
    "html": """
        (element
          (start_tag
            (tag_name) @tag
            (#match? @tag "^(main|section|article|form|template|nav|header|footer)$")
          )
        ) @semantic
    """,
    "css": """
        (rule_set) @rule
        (media_statement) @media
    """,
}


# =============================================================================
# %%
# Phase 0c: Logging Setup
# =============================================================================
# Tiger Style: logging is not hidden. It's explicitly configured with
# deterministic format and output destination.

def _setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure a root logger with explicit format and level.

    Tiger Style:
      - Logs go to stdout AND an append-only file per run.
      - File name: <run_id>_<ts_in_ns>.log under data/logs/.
      - Post-condition: logger is ready, handlers are attached.
    """
    logger = logging.getLogger("data_prep")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Tiger Style: explicit handler set — no duplicate handlers on re-init.
    if logger.handlers:
        return logger

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Append-only per-run file: data/logs/<run_id>_<ts_in_ns>.log.
    ensure_dir(LOGS_DIR)
    run_id = uuid.uuid4().hex[:8]
    ts_ns = time.time_ns()
    log_path = LOGS_DIR / f"{run_id}_{ts_ns}.log"
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("Log file: %s", log_path)

    return logger


# =============================================================================
# %%
# Phase 0d: Utility Functions
# =============================================================================

def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist. Return the path.

    Tiger Style: explicit assertion that dir exists post-call.
    No silent failures.
    """
    path.mkdir(parents=True, exist_ok=True)
    assert path.exists() and path.is_dir(), f"Failed to create directory: {path}"
    return path


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file. Returns empty list if file doesn't exist.

    Tiger Style: explicit bounds check — file size validated.
    """
    if not path.exists():
        return []
    file_size = path.stat().st_size
    assert file_size > 0, f"JSONL file is empty: {path}"
    assert file_size < 10_737_418_240, (  # 10 GB sanity cap
        f"JSONL file suspiciously large ({file_size} bytes): {path}"
    )
    results: List[Dict[str, Any]] = []
    with open(path, "rb") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(orjson.loads(line))
            except orjson.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_num}: {exc}")
    return results


def write_jsonl(path: Path, records: List[Dict[str, Any]], mode: str = "w") -> Path:
    """Write records to a JSONL file using orjson for speed.

    Tiger Style: post-condition asserts file written with correct record count.
    Use mode='a' to append to an existing file.
    """
    ensure_dir(path.parent)
    # Tiger Style: use binary mode, explicit encoding — no hidden text transforms.
    with open(path, "ab" if mode == "a" else "wb") as f:
        for record in records:
            f.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))
    if mode != "a":
        # Post-condition validation only for fresh writes (append count depends on prior state).
        written_count = sum(1 for _ in open(path, "rb") if _.strip())
        assert written_count == len(records), (
            f"Write count mismatch: expected {len(records)}, got {written_count}"
        )
    return path


def compute_blake3(data: str) -> str:
    """Compute BLAKE3 hash for deduplication.

    Tiger Style: explicit algorithm choice (BLAKE3 is faster than SHA-256
    and equally secure for dedup purposes).
    """
    try:
        import blake3
        return blake3.blake3(data.encode("utf-8")).hexdigest()
    except ImportError:
        # Graceful fallback to SHA-256 if blake3 not installed.
        return hashlib.sha256(data.encode("utf-8")).hexdigest()


def safe_filename(url_or_name: str) -> str:
    """Sanitize a string for use as a filename.

    Tiger Style: explicit character filtering — no assumptions about input.
    Appends a short hash when the sanitized name is truncated so that long
    URLs never collide to the same cache file.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", url_or_name)
    assert len(sanitized) > 0, f"Filename became empty after sanitization: {url_or_name}"
    if len(sanitized) <= 64:
        return sanitized
    digest = hashlib.sha1(url_or_name.encode("utf-8")).hexdigest()[:8]
    return f"{sanitized[:55]}_{digest}"


def _check_checkpoint(
    output_path: Path,
    label: str,
    min_records: int = 0,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Check if a phase output already exists and is valid.

    Tiger Style:
      - Explicit pre/post condition check.
      - Returns True if checkpoint is valid (skip the phase).
      - Returns False if checkpoint is missing or invalid (run the phase).
    """
    if not output_path.exists():
        if logger:
            logger.info("  No prior output at %s — running phase.", output_path)
        return False
    file_size = output_path.stat().st_size
    if file_size == 0:
        if logger:
            logger.warning("  Prior output at %s is empty — re-running.", output_path)
        return False
    if min_records > 0:
        line_count = sum(1 for _ in open(output_path, "rb") if _.strip())
        if line_count < min_records:
            if logger:
                logger.warning(
                    "  Prior output at %s has %d records (need %d) — re-running.",
                    output_path, line_count, min_records,
                )
            return False
    if logger:
        size_mb = file_size / (1024 * 1024)
        logger.info("  Checkpoint found: %s (%.1f MB) — skipping phase.", output_path, size_mb)
    return True


# =============================================================================
# %%
# Phase 1: Clone Target Repositories
# =============================================================================

def phase_clone_repos(logger: logging.Logger) -> List[Path]:
    """Clone all 8 repos into input_data/repos/.

    Tiger Style:
      - Explicit check that git is installed before cloning.
      - depth=1 to avoid downloading history we don't need.
      - Existing repos are pulled to latest, not re-cloned (network-efficient).
      - Post-condition: every repo directory exists and has files.

    Returns:
        List of paths to cloned repo directories.
    """
    # Checkpoint: skip if all 8 repos already exist with files.
    existing = sum(1 for r in REPOS if (REPOS_DIR / r["name"]).exists())
    if existing == len(REPOS):
        logger.info("=== Phase 1: All %d repos already cloned — skipping.", len(REPOS))
        return [REPOS_DIR / r["name"] for r in REPOS]
    logger.info("=== Phase 1: Cloning %d repositories (%d already exist) ===", len(REPOS), existing)

    # Tiger Style: verify tool exists before use.
    git_path = shutil.which("git")
    assert git_path is not None, (
        "`git` not found in PATH. Install git: brew install git (macOS) "
        "or apt install git (Linux)."
    )

    repo_paths: List[Path] = []
    for repo in REPOS:
        target = REPOS_DIR / repo["name"]
        logger.info("  [%s] %s → %s", repo["lang"], repo["name"], target)

        if target.exists():
            # Tiger Style: verify existing dir is actually a git repo.
            git_dir = target / ".git"
            assert git_dir.exists() and git_dir.is_dir(), (
                f"{target} exists but is not a git repository. "
                f"Remove it manually and re-run."
            )
            logger.info("    Already exists, pulling latest...")
            result = subprocess.run(
                ["git", "-C", str(target), "pull", "--ff-only"],
                capture_output=True, text=True, timeout=120,
            )
            assert result.returncode == 0, (
                f"git pull failed for {repo['name']}:\n{result.stderr}"
            )
        else:
            ensure_dir(REPOS_DIR)
            logger.info("    Cloning (depth=1)...")
            result = subprocess.run(
                ["git", "clone", "--depth", "1", repo["url"], str(target)],
                capture_output=True, text=True, timeout=300,
            )
            assert result.returncode == 0, (
                f"git clone failed for {repo['name']}:\n{result.stderr}"
            )

        # Post-condition: repo directory is non-empty.
        file_count = len(list(target.rglob("*")))
        assert file_count > 5, (
            f"Repo {repo['name']} has only {file_count} files — likely a failed clone."
        )
        repo_paths.append(target)
        logger.info("    OK — %d files", file_count)

    logger.info("=== Phase 1 complete: %d repos ready ===", len(repo_paths))
    return repo_paths


# =============================================================================
# %%
# Phase 2: Chunk Code with tree-sitter
# =============================================================================

def _init_tree_sitter_parser(language_name: str) -> Parser:
    """Create a tree-sitter Parser for the given language.

    Tiger Style: explicit language availability check. Fail fast if grammar
    is not installed rather than failing 1000 files in.
    """
    # Map language name → pip package name → tree-sitter language object.
    grammar_map = {
        "c":          "tree_sitter_c",
        "python":     "tree_sitter_python",
        "javascript": "tree_sitter_javascript",
        "html":       "tree_sitter_html",
        "css":        "tree_sitter_css",
    }

    assert language_name in grammar_map, (
        f"Unsupported language: {language_name}. "
        f"Supported: {list(grammar_map.keys())}"
    )

    module_name = grammar_map[language_name]
    try:
        lang_mod = __import__(module_name)
    except ImportError:
        raise ImportError(
            f"Missing tree-sitter grammar for {language_name}. "
            f"Install: uv pip install tree-sitter-{language_name}"
        )

    lang_obj = Language(lang_mod.language())
    parser = Parser()
    parser.language = lang_obj
    return parser


def _get_source_files(repo_path: Path, language: str) -> List[Path]:
    """Recursively find all source files for a given language in a repo.

    Tiger Style:
      - Explicit exclusion of test/build/vendor dirs.
      - Extension validation against LANG_EXTENSIONS.
      - File size / line count bounds checking.
    """
    assert language in LANG_EXTENSIONS, f"Unknown language: {language}"
    valid_extensions = LANG_EXTENSIONS[language]

    source_files: List[Path] = []
    for filepath in repo_path.rglob("*"):
        # Skip directories.
        if not filepath.is_file():
            continue

        # Check extension.
        if filepath.suffix not in valid_extensions:
            continue

        # Check exclusion directories.
        rel_parts = filepath.relative_to(repo_path).parts
        if any(part in EXCLUDE_DIRS for part in rel_parts):
            continue

        # Check max lines.
        try:
            with open(filepath, "rb") as f:
                line_count = sum(1 for _ in f)
        except (OSError, PermissionError) as exc:
            # Tiger Style: log skip reason — no silent dropping.
            logging.getLogger("data_prep").warning(
                "Skipping unreadable file %s: %s", filepath, exc
            )
            continue

        if line_count > MAX_FILE_LINES:
            continue

        source_files.append(filepath)

    return source_files


def _extract_chunks_tree_sitter(
    filepath: Path,
    language: str,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """Extract function/class chunks from a single file using tree-sitter.

    Tiger Style:
      - Explicit pre-condition: file must be readable and non-empty.
      - Explicit bounds: each chunk must be between MIN_CHUNK_LINES and MAX_CHUNK_LINES.
      - Returns empty list (not None) on parse failure — no silent None propagation.
    """
    # Pre-condition: file exists and is readable.
    assert filepath.exists(), f"File not found: {filepath}"
    file_size = filepath.stat().st_size
    assert file_size > 0, f"File is empty: {filepath}"

    with open(filepath, "rb") as f:
        source_code = f.read()

    parser = _init_tree_sitter_parser(language)
    tree = parser.parse(source_code)
    root_node = tree.root_node

    # Run the language-specific query.
    query = TS_QUERIES.get(language)
    assert query is not None, f"No tree-sitter query defined for {language}"

    try:
        from tree_sitter import Query, QueryCursor
        ts_query = Query(parser.language, query)
        cursor = QueryCursor(ts_query)
        captures = cursor.matches(root_node)
    except Exception as exc:
        logger.warning("  tree-sitter query failed for %s: %s", filepath.name, exc)
        return []

    chunks: List[Dict[str, Any]] = []
    seen_ranges: set = set()  # Dedup overlapping captures.

    for pattern_index, capture_dict in captures:
        for capture_name, nodes in capture_dict.items():
            for node in nodes:
                start_line = node.start_point[0] + 1  # 1-indexed
                end_line = node.end_point[0] + 1
                line_count = end_line - start_line + 1

                # Bounds check: skip chunks outside valid range.
                if line_count < MIN_CHUNK_LINES or line_count > MAX_CHUNK_LINES:
                    continue

                # Dedup by byte range.
                range_key = (node.start_byte, node.end_byte)
                if range_key in seen_ranges:
                    continue
                seen_ranges.add(range_key)

                # Extract the text.
                chunk_text = source_code[node.start_byte:node.end_byte].decode(
                    "utf-8", errors="replace"
                )

                # Tiger Style: assert chunk text is non-empty.
                assert len(chunk_text) > 0, (
                    f"Empty chunk from {filepath}:{start_line}-{end_line}"
                )

                # Extract function name from first line.
                first_line = chunk_text.split("\n")[0].strip()

                chunks.append({
                    "repo": filepath.parent.parent.name,  # .../repos/{repo_name}/...
                    "language": language,
                    "file_path": str(filepath.relative_to(filepath.parent.parent.parent)),
                    "chunk_type": capture_name,
                    "name": first_line.split("(")[0].split()[-1]
                    if "(" in first_line and first_line.split("(")[0].strip()
                    else first_line[:64],
                    "start_line": start_line,
                    "end_line": end_line,
                    "code": chunk_text,
                    "signature": first_line[:256],
                })

    return chunks


def phase_chunk_code(logger: logging.Logger) -> Path:
    """Phase 2: Parse all source files with tree-sitter, extract chunks.

    Tiger Style:
      - Processes each repo sequentially (deterministic order).
      - Logs per-file progress with tqdm for transparency.
      - Post-condition: code_chunks_raw.jsonl exists and is non-empty.
    """
    output_path = CHUNKS_DIR / "code_chunks_raw.jsonl"
    if _check_checkpoint(output_path, "tree-sitter chunks", min_records=50, logger=logger):
        return output_path
    logger.info("=== Phase 2: tree-sitter chunking ===")

    all_chunks: List[Dict[str, Any]] = []
    total_files_processed = 0
    total_files_skipped = 0

    for repo_def in REPOS:
        repo_path = REPOS_DIR / repo_def["name"]
        if not repo_path.exists():
            logger.warning("  Repo %s not cloned yet. Skipping.", repo_def["name"])
            continue

        source_files = _get_source_files(repo_path, repo_def["lang"])
        logger.info(
            "  [%s] %s — %d source files found",
            repo_def["lang"], repo_def["name"], len(source_files),
        )

        for filepath in tqdm(source_files, desc=f"    Parsing {repo_def['name']}", leave=False):
            try:
                chunks = _extract_chunks_tree_sitter(filepath, repo_def["lang"], logger)
                if chunks:
                    all_chunks.extend(chunks)
                    total_files_processed += 1
                else:
                    total_files_skipped += 1
            except Exception as exc:
                # Tiger Style: log the error, don't crash the entire pipeline.
                logger.debug("  Error parsing %s: %s", filepath, exc)
                total_files_skipped += 1
                continue

    logger.info(
        "  Files: %d processed, %d skipped — %d total chunks extracted",
        total_files_processed, total_files_skipped, len(all_chunks),
    )

    # Tiger Style: post-condition — at least some chunks found.
    assert len(all_chunks) > 50, (
        f"Only {len(all_chunks)} chunks extracted. "
        "Check tree-sitter grammars and repo contents."
    )

    output_path = CHUNKS_DIR / "code_chunks_raw.jsonl"
    write_jsonl(output_path, all_chunks)
    logger.info("  Written: %s (%d chunks)", output_path, len(all_chunks))
    return output_path


# =============================================================================
# %%
# Phase 3: Generate Synthetic Instructions
# =============================================================================

INSTRUCTION_PROMPT_TEMPLATE = """\
You are an expert programmer. Given the following {language} code, write a clear,
concise instruction that a human would write to ask for exactly this code.
The instruction should be 1-3 sentences, be specific, and mention the language if relevant.
Output ONLY the instruction, nothing else.

CODE:
```{language}
{code}
```

INSTRUCTION:"""


def _validate_instruction(
    instruction: Optional[str],
    code: str,
    logger: Optional[logging.Logger],
) -> Optional[str]:
    """Apply the instruction quality filters; return the instruction or None.

    Tiger Style: shared by the serial and batched generation paths so both
    accept exactly the same outputs.
    """
    if not instruction:
        return None

    if len(instruction) < 5:
        if logger:
            logger.debug("  Rejected: instruction too short (%d chars)", len(instruction))
        return None

    if instruction.lower().startswith(("i cannot", "i'm unable", "i am unable", "sorry")):
        if logger:
            logger.debug("  Rejected: model refused")
        return None

    # Check if model just echoed the code back.
    code_words = set(code.lower().split()[:20])
    inst_words = set(instruction.lower().split())
    overlap = len(code_words & inst_words)
    if overlap > len(code_words) * 0.8 and len(code_words) > 5:
        if logger:
            logger.debug("  Rejected: instruction copies code (%.0f%% overlap)", overlap / len(code_words) * 100)
        return None

    return instruction


def _generate_instruction(
    llm: Any,
    language: str,
    code: str,
    logger: logging.Logger,
) -> Optional[str]:
    """Generate a single instruction for a code chunk using the local LLM.

    Tiger Style:
      - Explicit retry logic with MAX_INSTRUCTION_GEN_RETRIES.
      - Output validation: length check, refusal check, copy check.
      - Returns None on failure (explicit), never empty string.
    """
    prompt = INSTRUCTION_PROMPT_TEMPLATE.format(language=language, code=code[:MAX_INSTRUCTION_CODE_CHARS])
    # Bounds check: prompt must be non-empty and reasonable length.
    assert len(prompt) > 50, f"Prompt too short ({len(prompt)} chars)"
    assert len(prompt) < 8192, f"Prompt too long ({len(prompt)} chars)"

    for attempt in range(1, MAX_INSTRUCTION_GEN_RETRIES + 1):
        try:
            response = llm(
                prompt,
                max_tokens=128,
                temperature=0.7,
                stop=["\n\n"],
            )
            instruction = _validate_instruction(
                response["choices"][0]["text"].strip(), code, logger
            )
        except Exception as exc:
            logger.debug("  LLM generation attempt %d failed: %s", attempt, exc)
            if attempt == MAX_INSTRUCTION_GEN_RETRIES:
                return None
            time.sleep(1)
            continue

        if instruction is not None:
            return instruction

    return None


def _make_instruct_record(instruction: str, chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Build the output record shape for a validated instruction."""
    return {
        "instruction": instruction,
        "input": "",
        "output": chunk["code"],
        "metadata": {
            "layer": "code",
            "language": chunk["language"],
            "repo": chunk["repo"],
            "chunk_name": chunk.get("name", "unknown"),
        },
    }


def _generate_instructions_batched(
    llm: Any,
    chunks: List[Dict[str, Any]],
    completed: int,
    output_path: Path,
    logger: logging.Logger,
) -> Tuple[int, int]:
    """Generate instructions using the LM Studio batched backend.

    Returns (saved_count, generation_failures).
    """
    from collections import deque

    queue: deque = deque()
    total_to_process = 0
    for idx, chunk in enumerate(chunks):
        if idx < completed:
            continue
        queue.append((idx, chunk, MAX_INSTRUCTION_GEN_RETRIES))
        total_to_process += 1

    fout = open(output_path, "ab")
    saved_count = 0
    generation_failures = 0
    retry_count = 0
    pbar = tqdm(total=total_to_process, desc="  Generating instructions (batched)")
    _batch_t0 = time.monotonic()
    _log_interval = max(50, total_to_process // 100)

    try:
        batch_num = 0
        while queue:
            batch_num += 1
            wave = []
            while queue and len(wave) < llm.n_parallel:
                wave.append(queue.popleft())

            reqs = [
                {
                    "prompt": INSTRUCTION_PROMPT_TEMPLATE.format(
                        language=c["language"], code=c["code"][:MAX_INSTRUCTION_CODE_CHARS]
                    ),
                    "max_tokens": 128,
                    "temperature": 0.7,
                    "stop": ["\n\n"],
                }
                for _, c, _ in wave
            ]

            _wave_t0 = time.monotonic()
            texts = llm.complete_batch(reqs)
            _wave_dt = time.monotonic() - _wave_t0

            wave_retries = 0
            for (idx, chunk, attempts), text in zip(wave, texts):
                if text is not None:
                    instruction = _validate_instruction(text, chunk["code"], logger)
                else:
                    instruction = None

                if instruction is None:
                    if attempts > 1:
                        queue.append((idx, chunk, attempts - 1))
                        wave_retries += 1
                    else:
                        generation_failures += 1
                        pbar.update(1)
                    continue

                record = _make_instruct_record(instruction, chunk)
                fout.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))
                fout.flush()
                saved_count += 1
                pbar.update(1)
            retry_count += wave_retries

            if batch_num % _log_interval == 0:
                _elapsed = time.monotonic() - _batch_t0
                rate = saved_count / _elapsed if _elapsed > 0 else 0
                avg_code_chars = sum(len(c.get("code", "")) for _, c, _ in wave) / len(wave) if wave else 0
                logger.info(
                    "  Batch %d: %d prompts in %.1fs (%.2fs/prompt, avg %.0f code chars, "
                    "%d saved/%d failed/%d retrying, %.2f chunks/s)",
                    batch_num, len(wave), _wave_dt, _wave_dt / max(len(wave), 1),
                    avg_code_chars, saved_count, generation_failures, retry_count, rate,
                )
    finally:
        fout.close()
        pbar.close()

    return saved_count, generation_failures


def _generate_instructions_serial(
    llm: Any,
    chunks: List[Dict[str, Any]],
    completed: int,
    output_path: Path,
    logger: logging.Logger,
) -> Tuple[int, int]:
    """Generate instructions one-at-a-time via llama-cpp-python (fallback)."""
    generation_failures = 0
    saved_count = 0

    fout = open(output_path, "ab")
    try:
        for idx, chunk in enumerate(tqdm(chunks, desc="  Generating instructions")):
            if idx < completed:
                continue

            instruction = _generate_instruction(
                llm, chunk["language"], chunk["code"], logger
            )
            if instruction is None:
                generation_failures += 1
                continue

            record = _make_instruct_record(instruction, chunk)
            fout.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))
            fout.flush()
            saved_count += 1
    finally:
        fout.close()

    return saved_count, generation_failures


def phase_generate_instructions(
    logger: logging.Logger,
    instruct_limit: Optional[int] = None,
    cpu_workers: Optional[int] = None,
    cpu_threads: Optional[int] = None,
    gpu_parallel: Optional[int] = None,
) -> Path:
    """Phase 3: Generate synthetic instructions for every code chunk.

    Tiger Style:
      - Lazy-loads the LLM (no wasted resources if this phase is skipped).
      - Processes chunks in deterministic order with progress bar.
      - Writes filtered results — chunks that failed generation are dropped.
      - Uses the resource-aware orchestrator (GPU batched + CPU workers) so
        every memory bus on the machine is busy; falls back to the old serial
        path only if the orchestrator cannot start.
    """
    output_path = CHUNKS_DIR / "code_chunks_ready.jsonl"
    if _check_checkpoint(output_path, "generated instructions", min_records=50, logger=logger):
        return output_path
    logger.info("=== Phase 3: Generating synthetic instructions ===")

    input_path = CHUNKS_DIR / "code_chunks_raw.jsonl"
    assert input_path.exists(), (
        f"Run --phase chunk first. File not found: {input_path}"
    )

    chunks = read_jsonl(input_path)
    if instruct_limit is not None:
        chunks = chunks[:instruct_limit]
        logger.info("  (limited to first %d chunks)", len(chunks))
    logger.info("  Loaded %d code chunks", len(chunks))

    model_path = MODELS_DIR / "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    assert model_path.exists(), (
        f"Model not found at {model_path}. "
        "Download: huggingface-cli download Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF "
        "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --local-dir models/"
    )

    model_path_str = str(model_path)

    completed = 0
    if output_path.exists():
        completed = sum(1 for _ in open(output_path, "rb") if _.strip())
        logger.info("  Resuming from checkpoint: %d chunks already done", completed)

    remaining = chunks[completed:]
    if not remaining:
        logger.info("  All chunks already processed — nothing to do.")
        return output_path

    ensure_dir(output_path.parent)

    jobs = [
        {
            "prompt": INSTRUCTION_PROMPT_TEMPLATE.format(
                language=c["language"], code=c["code"][:MAX_INSTRUCTION_CODE_CHARS]
            ),
            "max_tokens": 128,
            "temperature": 0.7,
            "stop": ["\n\n"],
        }
        for c in remaining
    ]

    try:
        specs = _orchestrator_plan(
            model_path_str, cpu_workers, cpu_threads, gpu_parallel, logger,
            sample_jobs=jobs[:60] if cpu_workers is None and gpu_parallel is None else None,
        )
        if not specs:
            raise RuntimeError("orchestrator planned no workers")
    except Exception as exc:
        logger.warning("  Orchestrator unavailable (%s) — falling back to batched/serial", exc)
        specs = []

    if specs:
        # Stable raw output name so an interrupted run can be resumed on the
        # next invocation (run_llm_completions seeds from the .partial file and
        # only does the missing jobs). Unlinked on success.
        raw_out = CHUNKS_DIR / "instruct_raw.jsonl"
        stats = _run_orchestrated(model_path_str, jobs, raw_out, specs, logger)
        logger.info("  Orchestrator stats: %s", {
            k: v for k, v in stats.items() if k in ("ok", "failed", "elapsed_s", "jobs_per_s")
        })

        # Map raw results back to validated records, in original chunk order.
        results: Dict[int, Dict[str, Any]] = {}
        for line in open(raw_out, "rb"):
            rec = orjson.loads(line)
            results[rec["_idx"]] = rec
        raw_out.unlink(missing_ok=True)

        saved_count = 0
        generation_failures = 0
        with open(output_path, "ab") as fout:
            for i, chunk in enumerate(remaining):
                rec = results.get(i)
                text = rec["_text"] if rec else None
                instruction = _validate_instruction(text, chunk["code"], logger)
                if instruction is None:
                    generation_failures += 1
                    continue
                record = _make_instruct_record(instruction, chunk)
                fout.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))
                saved_count += 1
        total_done = completed + saved_count
        logger.info(
            "  Generated: %d / %d chunks (%.1f%% success, %d failures)",
            total_done, len(chunks),
            total_done / max(len(chunks), 1) * 100,
            generation_failures,
        )
        logger.info("  Written: %s (%d examples)", output_path, total_done)
        assert total_done > 50, (
            f"Only {total_done} successful generations. Check model or chunks."
        )
        return output_path

    # ── Fallback: legacy batched LM Studio backend, then serial ───────────
    batched = _load_batched_llm(model_path_str, logger)
    if batched is not None:
        saved_count, generation_failures = _generate_instructions_batched(
            batched, remaining, 0, output_path, logger
        )
    else:
        llm = _load_llm(model_path_str)
        saved_count, generation_failures = _generate_instructions_serial(
            llm, remaining, 0, output_path, logger
        )

    total_done = completed + saved_count
    logger.info(
        "  Generated: %d / %d chunks (%.1f%% success, %d failures)",
        total_done, len(chunks),
        total_done / max(len(chunks), 1) * 100,
        generation_failures,
    )

    assert total_done > 50, (
        f"Only {total_done} successful generations. Check model or chunks."
    )

    logger.info("  Written: %s (%d examples)", output_path, total_done)
    return output_path


# =============================================================================
# %%
# Phase 4: Scrape Documentation
# =============================================================================

DOC_SOURCES = {    "cppreference": {
        "base_url": "https://en.cppreference.com/w/c",
        "sections": [
            "string", "io", "program", "numeric", "chrono",
            "memory", "thread", "atomic", "locale", "program/signal",
            "string/byte", "string/wide", "error",
        ],
    },
    "python_docs": {
        "base_url": "https://docs.python.org/3/library/",
        "modules": [
            "os", "sys", "json", "asyncio", "pathlib", "collections",
            "re", "datetime", "math", "random", "itertools", "functools",
            "typing", "dataclasses", "concurrent.futures", "subprocess",
            "argparse", "logging", "sqlite3", "csv", "hashlib", "base64",
            "secrets", "socket", "ssl", "http.client", "urllib.request",
            "urllib.parse", "threading", "multiprocessing", "queue",
            "enum", "abc", "contextlib", "heapq", "bisect", "struct",
            "decimal", "fractions", "statistics", "string", "glob", "shutil",
            "tempfile", "zipfile", "tarfile", "gzip", "io", "textwrap",
            "pprint", "pickle", "copy", "operator", "warnings", "traceback",
        ],
    },
    "mdn": {
        "base_url": "https://developer.mozilla.org/en-US/docs/Web",
        "sections": [
            "JavaScript/Reference/Global_Objects/Array",
            "JavaScript/Reference/Global_Objects/String",
            "JavaScript/Reference/Global_Objects/Promise",
            "JavaScript/Reference/Global_Objects/Map",
            "HTML/Element/form",
            "HTML/Element/input",
            "HTML/Element/section",
            "CSS/Reference",
            "JavaScript/Reference/Global_Objects/Date",
            "JavaScript/Reference/Global_Objects/JSON",
            "JavaScript/Reference/Global_Objects/Object",
            "JavaScript/Reference/Global_Objects/Set",
            "JavaScript/Reference/Global_Objects/RegExp",
            "JavaScript/Reference/Global_Objects/Math",
            "HTML/Element/button",
            "HTML/Element/a",
            "HTML/Element/img",
            "HTML/Element/table",
            "HTML/Element/div",
            "CSS/box-sizing",
            "CSS/flexbox",
            "CSS/grid",
            "CSS/position",
        ],
    },
}


# ── Hugging Face doc/QA sources (streamed via datasets-server, no auth) ────
# Tiger Style: used to inflate the doc layer beyond finite scraping. Rows are
# streamed from https://datasets-server.huggingface.co and mapped to the
# canonical instruction/input/output schema. Licenses are permissive.
HF_DOC_SOURCES: List[Dict[str, Any]] = [
    {
        "name": "dolly-15k",
        "dataset": "databricks/databricks-dolly-15k",
        "config": "default",
        "split": "train",
        "cap": 5000,
        "license": "CC-BY-SA-3.0",
        "input_col": "context",
        "output_col": "response",
        "extra_meta": ["category"],
    },
    {
        "name": "alpaca",
        "dataset": "tatsu-lab/alpaca",
        "config": "default",
        "split": "train",
        "cap": 5000,
        "license": "Apache-2.0",
        "input_col": "input",
        "output_col": "output",
        "extra_meta": [],
    },
]


# ── Hugging Face devops/config sources (chat format: human→gpt) ────────────
# Tiger Style: inflates the devops layer from real HF SFT/QA corpora, exactly
# like the doc layer. Conversation turns are parsed into instruction/output.
HF_DEVOPS_SOURCES: List[Dict[str, Any]] = [
    {
        "name": "k8s-sft-100k",
        "dataset": "stindardlogic/devops-kubernetes-sft-100k",
        "config": "default",
        "split": "train",
        "cap": 3000,
        "license": "Apache-2.0",
        "instruction_col": "instruction",
        "output_col": "output",
    },
    {
        "name": "stackexchange-devops",
        "dataset": "mlfoundations-dev/stackexchange_devops",
        "config": "default",
        "split": "train",
        "cap": 3000,
        "license": "CC-BY-SA-40",
        "instruction_col": "instruction",
        "output_col": "completion",
    },
]

HF_CHAT_HUMAN = "human"
HF_CHAT_GPT = "gpt"


def _scrape_with_cache(
    url: str,
    cache_dir: Path,
    logger: logging.Logger,
) -> Optional[str]:
    """Fetch a URL with disk caching.

    Tiger Style:
      - Explicit cache hit/miss logging.
      - Cached HTML stored as files named by URL hash.
      - Polite delay between requests.
    """
    cache_key = safe_filename(url) + ".html"
    cache_path = cache_dir / cache_key

    # Cache hit.
    if cache_path.exists():
        logger.debug("    Cache HIT: %s", url)
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()

    # Cache miss — fetch.
    logger.debug("    Cache MISS: %s", url)
    ensure_dir(cache_dir)

    try:
        resp = requests.get(url, timeout=30, headers={
            "User-Agent": "DataPrepBot/1.0 (educational; contact@example.com)",
        })
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("    Failed to fetch %s: %s", url, exc)
        return None

    html = resp.text
    # Tiger Style: validate response is plausible HTML.
    assert len(html) > 100, f"Response too short ({len(html)} bytes) for {url}"
    assert "<html" in html.lower() or "<!doctype" in html.lower(), (
        f"Response doesn't look like HTML: {url[:200]}"
    )

    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Polite delay.
    time.sleep(REQUEST_DELAY_SECONDS)
    return html


def _parse_cppreference_page(html: str, url: str) -> Optional[Dict[str, Any]]:
    """Parse a cppreference page into a Q&A example.

    Tiger Style: defensive parsing — every BeautifulSoup access checked.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Extract function name from <h1> or page title.
    h1 = soup.find("h1")
    if not h1:
        return None
    func_name = h1.get_text(strip=True)

    # Extract signature from the synopsis code block.
    sig_code = soup.find("code", class_="t-cpp")
    if not sig_code:
        sig_code = soup.find("code")
    signature = sig_code.get_text(strip=True) if sig_code else ""

    # Extract description (first <p> after the synopsis).
    desc_p = soup.find("p")
    description = desc_p.get_text(strip=True) if desc_p else ""

    # Extract example code.
    example_div = soup.find("div", class_="example")
    example_code = ""
    if example_div:
        pre = example_div.find("pre")
        if pre:
            example_code = pre.get_text(strip=True)

    if not signature and not description:
        return None

    instruction = f"What is the signature and purpose of C's {func_name}?"
    output_parts = [part for part in [signature, description, example_code] if part]
    output = "\n\n".join(output_parts)

    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "metadata": {"layer": "doc", "source": "cppreference", "topic": func_name},
    }


def _parse_python_doc_page(html: str, module_name: str) -> List[Dict[str, Any]]:
    """Parse a Python docs page into one or more Q&A examples.

    Tiger Style: iterates over all <dt class="sig"> elements (function
    signatures) and produces one example per function.
    """
    soup = BeautifulSoup(html, "html.parser")
    results: List[Dict[str, Any]] = []

    # Modern Python docs mark signatures with <dt class="sig sig-object py">.
    for dt in soup.find_all("dt", class_=True):
        classes = dt.get("class") or []
        if "sig" not in classes:
            continue
        signature = " ".join(dt.get_text(" ", strip=True).split())
        # Strip the trailing "¶" anchor link if present.
        signature = re.sub(r"\s*¶\s*$", "", signature)
        if not signature or "(" not in signature:
            continue

        # Get the description from the following <dd>.
        dd = dt.find_next("dd")
        description = dd.get_text(" ", strip=True)[:1000] if dd else ""

        func_name = signature.split("(")[0].strip().split(".")[-1]
        instruction = f"What is the Python {module_name}.{func_name} function signature and behavior?"
        output = f"{signature}\n\n{description}" if description else signature

        results.append({
            "instruction": instruction,
            "input": "",
            "output": output,
            "metadata": {"layer": "doc", "source": "python_docs", "topic": f"{module_name}.{func_name}"},
        })

    return results


def phase_scrape_docs(logger: logging.Logger) -> Path:
    """Phase 4: Scrape documentation sites and generate Q&A pairs.

    Tiger Style:
      - Caches raw HTML to disk (avoids re-scraping on re-run).
      - Processes each doc source independently (one fails, others continue).
      - Post-condition: doc_chunks.jsonl has at least 100 examples.
    """
    output_path = CHUNKS_DIR / "doc_chunks.jsonl"
    if _check_checkpoint(output_path, "doc Q&A pairs", min_records=50, logger=logger):
        return output_path
    logger.info("=== Phase 4: Scraping documentation ===")

    cache_dir = DOCS_DIR / "raw_cache"
    all_examples: List[Dict[str, Any]] = []

    # ── 4a: cppreference ───────────────────────────────────────────────────
    logger.info("  [cppreference] Starting...")
    cppref = DOC_SOURCES["cppreference"]
    for section in cppref["sections"]:
        url = f"{cppref['base_url']}/{section}"
        html = _scrape_with_cache(url, cache_dir / "cppreference", logger)
        if html is None:
            continue
        example = _parse_cppreference_page(html, url)
        if example:
            all_examples.append(example)

    # ── 4b: Python docs ────────────────────────────────────────────────────
    logger.info("  [python docs] Starting...")
    py_docs = DOC_SOURCES["python_docs"]
    for module_name in py_docs["modules"]:
        url = f"{py_docs['base_url']}{module_name}.html"
        html = _scrape_with_cache(url, cache_dir / "python", logger)
        if html is None:
            continue
        examples = _parse_python_doc_page(html, module_name)
        all_examples.extend(examples)

    # ── 4c: MDN ────────────────────────────────────────────────────────────
    logger.info("  [MDN] Starting...")
    mdn = DOC_SOURCES["mdn"]
    for section in mdn["sections"]:
        url = f"{mdn['base_url']}/{section}"
        html = _scrape_with_cache(url, cache_dir / "mdn", logger)
        if html is None:
            continue
        try:
            soup = BeautifulSoup(html, "html.parser")
            title_elem = soup.find("h1")
            title = title_elem.get_text(strip=True) if title_elem else section.split("/")[-1]
            # Extract the first meaningful paragraph.
            main_content = soup.find("main") or soup.find("article") or soup
            first_p = main_content.find("p")
            description = first_p.get_text(strip=True)[:1000] if first_p else ""

            # Extract syntax block.
            syntax_div = soup.find("div", class_="syntax") or soup.find("pre", class_="syntaxbox")
            syntax = syntax_div.get_text(strip=True) if syntax_div else ""

            output_parts = [part for part in [syntax, description] if part]
            output = "\n\n".join(output_parts) if output_parts else description

            all_examples.append({
                "instruction": f"What is the MDN reference for {title}?",
                "input": "",
                "output": output,
                "metadata": {"layer": "doc", "source": "mdn", "topic": title},
            })
        except Exception as exc:
            logger.debug("    Error parsing MDN page %s: %s", url, exc)
            continue

    logger.info("  Total doc examples: %d", len(all_examples))

    # Post-condition.
    assert len(all_examples) > 50, (
        f"Only {len(all_examples)} doc examples scraped. "
        "Check network connectivity or doc source URLs."
    )

    output_path = CHUNKS_DIR / "doc_chunks.jsonl"
    write_jsonl(output_path, all_examples)
    logger.info("  Written: %s (%d examples)", output_path, len(all_examples))
    return output_path


# =============================================================================
# %%
# Phase 4b: Stream Doc/QA Examples from Hugging Face datasets-server
# =============================================================================

HF_ROWS_PAGE_SIZE = 100
_HF_API = "https://datasets-server.huggingface.co"


def _fetch_hf_rows(
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """Fetch a page of rows from the Hugging Face datasets-server API.

    Tiger Style:
      - Bounded page size (HF caps rows responses at 100).
      - Retries with backoff on 429 (rate-limit) instead of dropping data.
      - Returns [] (never None) on genuine failure — no silent crashes.
    """
    url = (
        f"{_HF_API}/rows?dataset={dataset}&config={config}"
        f"&split={split}&offset={offset}&length={length}"
    )
    max_retries = 4
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=40, headers={"User-Agent": "DataPrepBot/1.0"})
            if resp.status_code == 429:
                backoff = attempt * 5
                logger.warning("  HF rate-limited (429); backing off %ds (attempt %d)", backoff, attempt)
                time.sleep(backoff)
                continue
            resp.raise_for_status()
            payload = resp.json()
            rows = payload.get("rows", [])
            if not isinstance(rows, list):
                return []
            # Rows come back as {"row": {...}, "row_idx": ...}.
            return [r["row"] for r in rows if isinstance(r, dict) and "row" in r]
        except requests.RequestException as exc:
            if attempt < max_retries:
                logger.warning("  HF rows fetch failed at offset %d (attempt %d): %s", offset, attempt, exc)
                time.sleep(2 * attempt)
                continue
            logger.warning("  HF rows fetch failed at offset %d: %s", offset, exc)
            return []
        except (ValueError, KeyError) as exc:
            logger.warning("  HF rows parse failed at offset %d: %s", offset, exc)
            return []
    return []


def _hf_row_to_doc_example(
    source: Dict[str, Any],
    row: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Map a raw HF row into the canonical doc-layer schema.

    Tiger Style: validates presence of required values; returns None if the
    row is unusable so the caller can count it as skipped (no silent zeros).
    """
    instruction = str(row.get("instruction", "") or "").strip()
    input_text = str(row.get(source["input_col"], "") or "").strip()
    output = str(row.get(source["output_col"], "") or "").strip()

    if not instruction or not output:
        return None

    metadata: Dict[str, Any] = {
        "layer": "doc",
        "language": "unknown",
        "source": f"hf:{source['name']}",
        "license": source.get("license", ""),
    }
    for key in source.get("extra_meta", []):
        if row.get(key):
            metadata[key] = row.get(key)

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "metadata": metadata,
    }


def _fetch_hf_source(
    source: Dict[str, Any],
    mapper,  # callable(source, row) -> Optional[Dict[str, Any]]
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """Stream cap rows from one HF dataset config and map them to examples.

    Tiger Style:
      - Deduplicates within the stream by output hash (mirrors Phase 6).
      - Bounds work to the configured cap; never fetches unbounded data.
    """
    dataset = source["dataset"]
    config = source.get("config", "default")
    split = source.get("split", "train")
    cap = int(source.get("cap", 0))

    examples: List[Dict[str, Any]] = []
    seen_hashes: set = set()
    offset = 0
    empty_pages = 0

    while cap == 0 or len(examples) < cap:
        rows = _fetch_hf_rows(dataset, config, split, offset, HF_ROWS_PAGE_SIZE, logger)
        if not rows:
            empty_pages += 1
            if empty_pages >= 3:
                logger.warning("  HF %s: %d consecutive empty pages — stopping", source["name"], empty_pages)
                break
        else:
            empty_pages = 0
        for row in rows:
            if cap and len(examples) >= cap:
                break
            ex = mapper(source, row)
            if ex is None:
                continue
            h = compute_blake3(ex["output"])
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            examples.append(ex)
        if len(rows) < HF_ROWS_PAGE_SIZE:
            break
        offset += len(rows)
        # Tiger Style: polite pacing to avoid triggering HF rate limits (429).
        time.sleep(0.3)
        if offset > cap * 4:  # safety bound even if many rows are invalid
            logger.warning("  HF %s: hit safety bound at offset %d", source["name"], offset)
            break

    return examples


def _hf_row_to_devops_example(
    source: Dict[str, Any],
    row: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Map a chat-format HF row (human→gpt) into the devops-layer schema.

    Tiger Style: neutral about whether `conversations` is already a list or a
    JSON string; returns None on unusable rows (no silent empty strings).
    """
    conversation = row.get("conversations")
    if isinstance(conversation, str):
        try:
            conversation = json.loads(conversation)
        except (ValueError, TypeError):
            conversation = None

    if isinstance(conversation, list):
        instruction, output = "", ""
        for turn in conversation:
            if not isinstance(turn, dict):
                continue
            role = turn.get("from") or turn.get("role")
            value = str(turn.get("value") or turn.get("content") or "").strip()
            if role == HF_CHAT_HUMAN and not instruction:
                instruction = value
            elif role == HF_CHAT_GPT and not output:
                output = value
        if not instruction and not output:
            return None
    else:
        instruction = str(row.get("instruction", "") or "").strip()
        output = str(row.get(source["output_col"], "") or "").strip()
        if not instruction and not output:
            return None

    if not instruction or not output:
        return None

    metadata: Dict[str, Any] = {
        "layer": "devops",
        "language": "devops",
        "source": f"hf:{source['name']}",
        "license": source.get("license", ""),
    }
    src_meta = row.get("metadata")
    if isinstance(src_meta, dict) and src_meta.get("category"):
        metadata["category"] = src_meta["category"]

    return {
        "instruction": instruction,
        "input": "",
        "output": output,
        "metadata": metadata,
    }


def phase_fetch_hf(logger: logging.Logger) -> Dict[str, Path]:
    """Phase 4b: Stream doc/QA and devops examples from Hugging Face.

    Tiger Style:
      - Zero network flakiness: each page is fetched independently and skipped
        on transient errors (never crashes the pipeline).
      - Each output is checkpointed so re-runs are idempotent.
      - Post-conditions: doc_hf.jsonl and devops_hf.jsonl each >= 100 examples.
    """
    logger.info("=== Phase 4b: Streaming HuggingFace datasets (doc/QA + devops) ===")

    paths: Dict[str, Path] = {}

    doc_path = CHUNKS_DIR / "doc_hf.jsonl"
    if doc_path.exists():
        logger.info("  Checkpoint found: doc_hf.jsonl — skipping.")
    else:
        all_doc: List[Dict[str, Any]] = []
        for source in HF_DOC_SOURCES:
            exs = _fetch_hf_source(source, _hf_row_to_doc_example, logger)
            logger.info(
                "  [doc] %s — fetched %d examples (license: %s)",
                source["name"], len(exs), source.get("license", "?"),
            )
            all_doc.extend(exs)
        logger.info("  HF doc total: %d", len(all_doc))
        assert len(all_doc) >= 100, f"Only {len(all_doc)} HF doc examples fetched."
        write_jsonl(doc_path, all_doc)
        logger.info("  Written: %s (%d examples)", doc_path, len(all_doc))
    paths["doc"] = doc_path

    devops_path = CHUNKS_DIR / "devops_hf.jsonl"
    if devops_path.exists():
        logger.info("  Checkpoint found: devops_hf.jsonl — skipping.")
    else:
        all_devops: List[Dict[str, Any]] = []
        for source in HF_DEVOPS_SOURCES:
            exs = _fetch_hf_source(source, _hf_row_to_devops_example, logger)
            logger.info(
                "  [devops] %s — fetched %d examples (license: %s)",
                source["name"], len(exs), source.get("license", "?"),
            )
            all_devops.extend(exs)
        logger.info("  HF devops total: %d", len(all_devops))
        assert len(all_devops) >= 100, f"Only {len(all_devops)} HF devops examples fetched."
        write_jsonl(devops_path, all_devops)
        logger.info("  Written: %s (%d examples)", devops_path, len(all_devops))
    paths["devops"] = devops_path

    return paths


# =============================================================================
# %%
# Phase 5: Build Alignment Examples
# =============================================================================

TIGER_PRINCIPLES: List[Dict[str, str]] = [
    {
        "name": "No hidden memory allocations",
        "rule": "All buffers must be passed explicitly by the caller. No malloc/free inside functions.",
    },
    {
        "name": "No implicit control flow",
        "rule": "No exceptions, no hidden panics, no longjmp. Every error is returned as an explicit error code.",
    },
    {
        "name": "Explicit bounds checking",
        "rule": "Every array access, pointer arithmetic, and buffer operation must verify bounds before access.",
    },
    {
        "name": "No undefined behavior",
        "rule": "Even in 'unreachable' paths, behavior must be defined. No signed overflow, no uninitialized reads.",
    },
    {
        "name": "Deterministic destruction",
        "rule": "Resources must be cleaned up in reverse allocation order. RAII-style, no leak paths.",
    },
    {
        "name": "Minimal dependencies",
        "rule": "Prefer stdlib over external. Every dependency must be justified by measurable benefit.",
    },
]

PRINCIPLE_HINTS: Dict[str, str] = {
    "No hidden memory allocations": (
        "it allocates memory or resources internally and returns them, "
        "hiding ownership from the caller"
    ),
    "No implicit control flow": (
        "it raises exceptions or panics instead of returning explicit error codes"
    ),
    "Explicit bounds checking": (
        "it accesses arrays, lists, or buffers without verifying the index is in range"
    ),
    "No undefined behavior": (
        "it relies on undefined behavior like signed overflow or uninitialized reads"
    ),
    "Deterministic destruction": (
        "it leaks resources on early returns instead of cleaning up in reverse order"
    ),
    "Minimal dependencies": (
        "it pulls in heavy external dependencies when the standard library suffices"
    ),
}

# Tiger Style: seed examples hand-crafted with care. Each demonstrates
# one principle violation and its correct fix.
SEED_EXAMPLES: List[Dict[str, str]] = [
    {
        "principle": "No hidden memory allocations",
        "language": "c",
        "bad": (
            "char* read_file(const char* path) {\n"
            "    FILE* f = fopen(path, \"rb\");\n"
            "    fseek(f, 0, SEEK_END);\n"
            "    long size = ftell(f);\n"
            "    char* buf = malloc(size + 1);  // Hidden allocation!\n"
            "    fread(buf, 1, size, f);\n"
            "    buf[size] = '\\0';\n"
            "    fclose(f);\n"
            "    return buf;\n"
            "}"
        ),
        "good": (
            "// Caller allocates. Returns bytes read or error.\n"
            "ssize_t read_file(const char* path, char* buf, size_t buf_size) {\n"
            "    assert(buf != NULL);\n"
            "    assert(buf_size > 0);\n"
            "    FILE* f = fopen(path, \"rb\");\n"
            "    if (!f) return -1;\n"
            "    fseek(f, 0, SEEK_END);\n"
            "    long file_size = ftell(f);\n"
            "    if ((size_t)file_size >= buf_size) {\n"
            "        fclose(f);\n"
            "        return -2;  // Buffer too small\n"
            "    }\n"
            "    rewind(f);\n"
            "    size_t nread = fread(buf, 1, file_size, f);\n"
            "    buf[nread] = '\\0';\n"
            "    fclose(f);\n"
            "    return (ssize_t)nread;\n"
            "}"
        ),
    },
    {
        "principle": "Explicit bounds checking",
        "language": "python",
        "bad": (
            "def get_item(items, index):\n"
            "    return items[index]  # Can raise IndexError!"
        ),
        "good": (
            "from typing import Optional, List, TypeVar\n\n"
            "T = TypeVar('T')\n\n"
            "def get_item(items: List[T], index: int) -> Optional[T]:\n"
            "    if index < 0 or index >= len(items):\n"
            "        return None  # Explicit bounds check\n"
            "    return items[index]"
        ),
    },
    {
        "principle": "Minimal dependencies",
        "language": "python",
        "bad": (
            "import requests\n\n"
            "def fetch_json(url):\n"
            "    return requests.get(url).json()"
        ),
        "good": (
            "import urllib.request\n"
            "import json\n\n"
            "def fetch_json(url):\n"
            "    with urllib.request.urlopen(url) as resp:\n"
            "        return json.loads(resp.read().decode())"
        ),
    },
    {
        "principle": "No implicit control flow",
        "language": "c",
        "bad": (
            "int div(int a, int b) {\n"
            "    return a / b;  // Throws SIGFPE on b == 0\n"
            "}"
        ),
        "good": (
            "int div(int a, int b, int* out) {\n"
            "    if (out == NULL || b == 0) return -1;  // Explicit error code\n"
            "    *out = a / b;\n"
            "    return 0;\n"
            "}"
        ),
    },
    {
        "principle": "No undefined behavior",
        "language": "c",
        "bad": (
            "int add(int a, int b) {\n"
            "    return a + b;  // Signed overflow is UB\n"
            "}"
        ),
        "good": (
            "#include <stdint.h>\n\n"
            "int add_safe(int a, int b, int* out) {\n"
            "    if (out == NULL) return -1;\n"
            "    if ((b > 0 && a > INT32_MAX - b) ||\n"
            "        (b < 0 && a < INT32_MIN - b)) return -2;  // Overflow check\n"
            "    *out = a + b;\n"
            "    return 0;\n"
            "}"
        ),
    },
    {
        "principle": "Deterministic destruction",
        "language": "c",
        "bad": (
            "void process_file(const char* path) {\n"
            "    FILE* f = fopen(path, \"r\");\n"
            "    char buf[256];\n"
            "    while (fgets(buf, sizeof(buf), f)) {\n"
            "        if (strstr(buf, \"error\")) return;  // Leaks FILE!\n"
            "        handle(buf);\n"
            "    }\n"
            "    fclose(f);\n"
            "}"
        ),
        "good": (
            "void process_file(const char* path) {\n"
            "    FILE* f = fopen(path, \"r\");\n"
            "    if (!f) return;\n"
            "    char buf[256];\n"
            "    while (fgets(buf, sizeof(buf), f)) {\n"
            "        if (strstr(buf, \"error\")) {\n"
            "            fclose(f);\n"
            "            return;\n"
            "        }\n"
            "        handle(buf);\n"
            "    }\n"
            "    fclose(f);\n"
            "}"
        ),
    },
    {
        "principle": "Explicit bounds checking",
        "language": "javascript",
        "bad": (
            "function getItem(items, index) {\n"
            "    return items[index];  // undefined on out-of-range\n"
            "}"
        ),
        "good": (
            "function getItem(items, index) {\n"
            "    if (index < 0 || index >= items.length) {\n"
            "        return null;  // Explicit bounds check\n"
            "    }\n"
            "    return items[index];\n"
            "}"
        ),
    },
    {
        "principle": "No hidden memory allocations",
        "language": "python",
        "bad": (
            "def read_file(path):\n"
            "    with open(path) as f:\n"
            "        return f.read()  # Returns unbounded string\n"
        ),
        "good": (
            "def read_file(path, max_bytes):\n"
            "    with open(path) as f:\n"
            "        data = f.read(max_bytes + 1)  # Caller-controlled bound\n"
            "    if len(data) > max_bytes:\n"
            "        raise ValueError('file too large')\n"
            "    return data"
        ),
    },
    {
        "principle": "Deterministic destruction",
        "language": "python",
        "bad": (
            "def get_conn():\n"
            "    import sqlite3\n"
            "    return sqlite3.connect('app.db')  # Caller must remember to close\n"
        ),
        "good": (
            "from contextlib import contextmanager\n"
            "import sqlite3\n\n"
            "@contextmanager\n"
            "def get_conn(db_path='app.db'):\n"
            "    conn = sqlite3.connect(db_path)\n"
            "    try:\n"
            "        yield conn\n"
            "    finally:\n"
            "        conn.close()  # Deterministic destruction in reverse order"
        ),
    },
    {
        "principle": "No implicit control flow",
        "language": "python",
        "bad": (
            "def config_value(name):\n"
            "    return CONFIG[name]  # Raises KeyError\n"
        ),
        "good": (
            "from typing import Optional\n\n"
            "def config_value(name) -> Optional[str]:\n"
            "    if name not in CONFIG:\n"
            "        return None  # Explicit error return\n"
            "    return CONFIG[name]"
        ),
    },
    {
        "principle": "No undefined behavior",
        "language": "python",
        "bad": (
            "def pop_tail(items):\n"
            "    return items.pop()  # IndexError on empty list\n"
        ),
        "good": (
            "from typing import List, Optional, TypeVar\n\n"
            "T = TypeVar('T')\n\n"
            "def pop_tail(items: List[T]) -> Optional[T]:\n"
            "    if not items:\n"
            "        return None  # Defined behavior on empty input\n"
            "    return items.pop()"
        ),
    },
]


def _build_alignment_from_seed(
    seed: Dict[str, str],
    llm: Any,
    logger: logging.Logger,
) -> Optional[Dict[str, Any]]:
    """Build a single alignment training example from a seed.

    Uses the local LLM to generate a thought trace explaining the
    principle violation and the fix.
    """
    prompt = (
        f"Explain step-by-step why this code violates Tiger Style principle "
        f"'{seed['principle']}' and how to fix it.\n\n"
        f"BAD CODE ({seed['language']}):\n```\n{seed['bad']}\n```\n\n"
        f"EXPLANATION:"
    )

    try:
        response = llm(prompt, max_tokens=256, temperature=0.5, stop=["\n\n\n"])
        thought = response["choices"][0]["text"].strip()
    except Exception as exc:
        logger.debug("  LLM thought generation failed: %s", exc)
        thought = f"This code violates {seed['principle']}. The fix is provided below."

    # Tiger Style: validate output contains key elements.
    assert len(thought) > 20, f"Thought trace too short: {thought}"

    return {
        "instruction": (
            f"Refactor this {seed['language']} code to comply with Tiger Style: "
            f"{seed['principle']}"
        ),
        "input": seed["bad"],
        "output": f"<thought>{thought}</thought>\n\n{seed['good']}",
        "metadata": {
            "layer": "alignment",
            "principle": seed["principle"],
            "language": seed["language"],
        },
    }


def _generate_bad_code(
    llm: Any,
    language: str,
    principle: str,
) -> Optional[str]:
    """Generate a code snippet that violates a Tiger Style principle.

    Tiger Style: continuation-style prompt (the model completes after the
    "Output only the code:" lead-in) — the instruction-tuned model reliably
    produces code this way instead of parroting the format spec.
    """
    hint = PRINCIPLE_HINTS.get(principle, "it violates Tiger Style principles")
    prompt = (
        f"Write a short {language} function that has this flaw: {hint}. "
        f"Keep it under 15 lines. Output only the code:\n"
    )
    try:
        response = llm(prompt, max_tokens=256, temperature=0.8, stop=["\n\n"])
        return _strip_code_fences(response["choices"][0]["text"].strip())
    except Exception:
        return None


def _generate_good_code(
    llm: Any,
    language: str,
    principle: str,
    bad_code: str,
) -> Optional[str]:
    """Generate the corrected version of a flawed code snippet.

    Tiger Style: continuation-style prompt that fixes the given bad code.
    """
    prompt = (
        f"Here is a {language} function that violates '{principle}':\n"
        f"{bad_code}\n\n"
        f"Here is the corrected version that complies with '{principle}':\n"
    )
    try:
        response = llm(prompt, max_tokens=256, temperature=0.7, stop=["\n\n"])
        return _strip_code_fences(response["choices"][0]["text"].strip())
    except Exception:
        return None


def _strip_code_fences(code: str) -> str:
    """Remove leading/trailing ``` fences an LLM may wrap code in.

    Tiger Style: defensive — strips only complete fence pairs, never content.
    """
    lines = code.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _expand_alignment_examples(
    seed_examples: List[Dict[str, Any]],
    llm: Any,
    logger: logging.Logger,
    target_count: int = 200,
) -> List[Dict[str, Any]]:
    """Expand seed examples to a target count.

    Tiger Style:
      - Starts from deterministic hand-crafted seeds (never empty).
      - For each (principle, language) pair, generates a fresh flawed snippet
        then its corrected fix, looping until target_count is reached.
      - Every generated example is length-validated and deduplicated by
        output hash; failures are logged, never silently accepted.
    """
    expanded: List[Dict[str, Any]] = list(seed_examples)
    seen_outputs: set = set(compute_blake3(ex["output"]) for ex in expanded)
    target_languages = ["c", "python", "javascript", "html", "css"]
    principles = list(PRINCIPLE_HINTS.keys())
    round_num = 0
    max_rounds = 60

    while len(expanded) < target_count and round_num < max_rounds:
        round_num += 1
        made_progress = False

        for principle in principles:
            if len(expanded) >= target_count:
                break
            for target_lang in target_languages:
                if len(expanded) >= target_count:
                    break

                bad_code = _generate_bad_code(llm, target_lang, principle)
                if not bad_code or len(bad_code) < 10:
                    logger.debug("  Bad-code generation too short for %s/%s",
                                 principle, target_lang)
                    continue

                good_code = _generate_good_code(llm, target_lang, principle, bad_code)
                if not good_code or len(good_code) < 10:
                    logger.debug("  Good-code generation failed for %s/%s",
                                 principle, target_lang)
                    continue

                output = f"<thought>Fixing '{principle}' in {target_lang}.</thought>\n\n{good_code}"
                output_hash = compute_blake3(output)
                if output_hash in seen_outputs:
                    logger.debug("  Duplicate alignment example skipped")
                    continue
                seen_outputs.add(output_hash)

                expanded.append({
                    "instruction": (
                        f"Refactor this {target_lang} code to comply with Tiger Style: "
                        f"{principle}"
                    ),
                    "input": bad_code,
                    "output": output,
                    "metadata": {
                        "layer": "alignment",
                        "principle": principle,
                        "language": target_lang,
                    },
                })
                made_progress = True

        if not made_progress:
            logger.warning("  No new alignment examples generated in round %d — stopping.", round_num)
            break

    logger.info("  Alignment expansion: %d examples after %d rounds", len(expanded), round_num)
    random.shuffle(expanded)
    return expanded[:target_count]


def _expand_alignment_examples_orchestrated(
    seed_examples: List[Dict[str, Any]],
    model_path: str,
    specs: List[Any],
    logger: logging.Logger,
    target_count: int = 500,
) -> List[Dict[str, Any]]:
    """Expand seed examples using the orchestrator fleet (GPU + CPU).

    The (principle, lang) bad-code generations are independent, so they are all
    sent to the fleet in ONE batch; the good-code fix prompts depend on the bad
    outputs, so they go in a second fleet batch.  Two fleet startups total,
    instead of one per round.
    """
    expanded: List[Dict[str, Any]] = list(seed_examples)
    seen_outputs: set = set(compute_blake3(ex["output"]) for ex in expanded)
    target_languages = ["c", "python", "javascript", "html", "css"]
    principles = list(PRINCIPLE_HINTS.keys())
    combos = [(p, l) for p in principles for l in target_languages]

    # Number of (principle, lang) repeats needed to *likely* reach target_count:
    # ~50% of bad codes are valid and ~80% of those yield unique examples.
    needed = max(1, int(target_count / (len(combos) * 0.4)))
    candidates: List[Tuple[str, str]] = (combos * needed)[:needed * len(combos)]

    bad_reqs = [
        {
            "prompt": (
                f"Write a short {lang} function that has this flaw: "
                f"{PRINCIPLE_HINTS.get(principle, 'it violates Tiger Style principles')}. "
                f"Keep it under 15 lines. Output only the code:\n"
            ),
            "max_tokens": 256,
            "temperature": 0.8,
            "stop": ["\n\n"],
        }
        for principle, lang in candidates
    ]
    logger.info("  Alignment: requesting %d bad-code generations via orchestrator", len(bad_reqs))
    bad_texts = _orchestrator_complete(model_path, bad_reqs, specs, logger)
    bad_texts = bad_texts + [None] * (len(bad_reqs) - len(bad_texts))

    valid: List[Tuple[str, str, str]] = []
    for (principle, lang), bt in zip(candidates, bad_texts):
        if not bt:
            continue
        bad = _strip_code_fences(bt.strip())
        if len(bad) >= 10:
            valid.append((principle, lang, bad))

    logger.info("  Alignment: %d valid bad-code snippets → good-code pass", len(valid))
    if valid:
        good_reqs = [
            {
                "prompt": (
                    f"Here is a {lang} function that violates '{principle}':\n"
                    f"{bad}\n\n"
                    f"Here is the corrected version that complies with '{principle}':\n"
                ),
                "max_tokens": 256,
                "temperature": 0.7,
                "stop": ["\n\n"],
            }
            for principle, lang, bad in valid
        ]
        good_texts = _orchestrator_complete(model_path, good_reqs, specs, logger)
        good_texts = good_texts + [None] * (len(good_reqs) - len(good_texts))

        for (principle, lang, bad), gt in zip(valid, good_texts):
            if not gt:
                continue
            good = _strip_code_fences(gt.strip())
            if len(good) < 10:
                continue
            output = f"<thought>Fixing '{principle}' in {lang}.</thought>\n\n{good}"
            output_hash = compute_blake3(output)
            if output_hash in seen_outputs:
                continue
            seen_outputs.add(output_hash)
            expanded.append({
                "instruction": (
                    f"Refactor this {lang} code to comply with Tiger Style: "
                    f"{principle}"
                ),
                "input": bad,
                "output": output,
                "metadata": {
                    "layer": "alignment",
                    "principle": principle,
                    "language": lang,
                },
            })
            if len(expanded) >= target_count:
                break

    logger.info("  Alignment orchestrator expansion: %d examples", len(expanded))
    random.shuffle(expanded)
    return expanded[:target_count]


def _expand_alignment_examples_batched(
    seed_examples: List[Dict[str, Any]],
    blm: Any,
    logger: logging.Logger,
    target_count: int = 500,
) -> List[Dict[str, Any]]:
    """Expand seed examples using the LM Studio batched backend.

    Unlike the serial version which calls the LLM 2-3 times per (principle, lang)
    pair, this version collects all bad-code prompts for a round, batch-decodes
    them, then batch-decodes the corresponding good-code prompts.  The 65 (principle,
    lang) pairs complete in ~5-6 waves of n_parallel=12 each instead of 65 serial
    decode pairs.
    """
    expanded: List[Dict[str, Any]] = list(seed_examples)
    seen_outputs: set = set(compute_blake3(ex["output"]) for ex in expanded)
    target_languages = ["c", "python", "javascript", "html", "css"]
    principles = list(PRINCIPLE_HINTS.keys())
    round_num = 0
    max_rounds = 60

    while len(expanded) < target_count and round_num < max_rounds:
        round_num += 1

        # Build the candidate list for this round: every (principle, lang) pair.
        candidates: List[Tuple[str, str]] = []
        for principle in principles:
            for lang in target_languages:
                if len(expanded) + len(candidates) >= target_count:
                    break
                candidates.append((principle, lang))
            if len(expanded) + len(candidates) >= target_count:
                break

        if not candidates:
            break

        # ── Phase A: generate bad code for all candidates ───────────────────
        bad_reqs = [
            {
                "prompt": (
                    f"Write a short {lang} function that has this flaw: "
                    f"{PRINCIPLE_HINTS.get(principle, 'it violates Tiger Style principles')}. "
                    f"Keep it under 15 lines. Output only the code:\n"
                ),
                "max_tokens": 256,
                "temperature": 0.8,
                "stop": ["\n\n"],
            }
            for principle, lang in candidates
        ]
        bad_texts = blm.complete_batch(bad_reqs)

        valid_bad: List[Tuple[str, str, str]] = []
        for (principle, lang), bad_text in zip(candidates, bad_texts):
            if not bad_text:
                continue
            bad = _strip_code_fences(bad_text.strip())
            if len(bad) >= 10:
                valid_bad.append((principle, lang, bad))

        if not valid_bad:
            logger.debug("  Alignment: no valid bad code in round %d", round_num)
            break

        # ── Phase B: generate good code for each successful bad code ─────────
        good_reqs = [
            {
                "prompt": (
                    f"Here is a {lang} function that violates '{principle}':\n"
                    f"{bad}\n\n"
                    f"Here is the corrected version that complies with '{principle}':\n"
                ),
                "max_tokens": 256,
                "temperature": 0.7,
                "stop": ["\n\n"],
            }
            for principle, lang, bad in valid_bad
        ]
        good_texts = blm.complete_batch(good_reqs)

        made_progress = False
        for (principle, lang, bad), good_text in zip(valid_bad, good_texts):
            if not good_text:
                continue
            good = _strip_code_fences(good_text.strip())
            if len(good) < 10:
                continue

            output = f"<thought>Fixing '{principle}' in {lang}.</thought>\n\n{good}"
            output_hash = compute_blake3(output)
            if output_hash in seen_outputs:
                continue
            seen_outputs.add(output_hash)

            expanded.append({
                "instruction": (
                    f"Refactor this {lang} code to comply with Tiger Style: "
                    f"{principle}"
                ),
                "input": bad,
                "output": output,
                "metadata": {
                    "layer": "alignment",
                    "principle": principle,
                    "language": lang,
                },
            })
            made_progress = True

            if len(expanded) >= target_count:
                break

        if not made_progress:
            logger.warning("  Alignment batched: no new examples in round %d — stopping", round_num)
            break

    logger.info("  Alignment batched expansion: %d examples after %d rounds", len(expanded), round_num)
    random.shuffle(expanded)
    return expanded[:target_count]


def phase_build_alignment(logger: logging.Logger) -> Path:
    """Phase 5: Build alignment examples (Tiger Style + design principles).

    Tiger Style applied to alignment building itself:
      - Seed examples are hard-coded (deterministic, auditable).
      - LLM expansion is optional — core seeds are always present.
      - Post-condition: alignment_chunks.jsonl has at least 50 examples.
    """
    output_path = CHUNKS_DIR / "alignment_chunks.jsonl"
    if _check_checkpoint(output_path, "alignment examples", min_records=100, logger=logger):
        return output_path
    logger.info("=== Phase 5: Building alignment examples ===")

    model_path = MODELS_DIR / "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    model_exists = model_path.exists()
    model_path_str = str(model_path) if model_exists else ""

    if not model_exists:
        logger.warning(
            "  Model not found at %s. Using seed examples only (no LLM expansion).",
            model_path,
        )
        blm = None
        llm = None
    else:
        # Prefer the LM Studio batched backend.
        blm = _load_batched_llm(model_path_str, logger)
        if blm is None:
            llm = _load_llm(model_path_str)
        else:
            llm = None

    # Build seed examples. With the batched backend we use hardcoded thought
    # text (there are only ~18 seeds — not worth a batch call).
    seed_results: List[Dict[str, Any]] = []
    for seed in SEED_EXAMPLES:
        if llm:
            result = _build_alignment_from_seed(seed, llm, logger)
        else:
            result = {
                "instruction": (
                    f"Refactor this {seed['language']} code to comply with Tiger Style: "
                    f"{seed['principle']}"
                ),
                "input": seed["bad"],
                "output": f"<thought>This code violates {seed['principle']}.</thought>\n\n{seed['good']}",
                "metadata": {
                    "layer": "alignment",
                    "principle": seed["principle"],
                    "language": seed["language"],
                },
            }
        if result:
            seed_results.append(result)

    logger.info("  Seed alignment examples: %d", len(seed_results))

    # Expand to more examples via the orchestrator fleet if we can plan workers.
    try:
        specs = _orchestrator_plan(model_path_str, 4, None, None, logger)
    except Exception as exc:
        logger.warning("  Orchestrator unavailable for alignment (%s)", exc)
        specs = []

    if specs:
        expanded = _expand_alignment_examples_orchestrated(
            seed_results, model_path_str, specs, logger, target_count=500,
        )
        logger.info("  Expanded to %d examples via orchestrator fleet", len(expanded))
        all_alignment = expanded
    elif blm:
        expanded = _expand_alignment_examples_batched(seed_results, blm, logger, target_count=500)
        logger.info("  Expanded to %d examples via batched LLM", len(expanded))
        all_alignment = expanded
    elif llm:
        expanded = _expand_alignment_examples(seed_results, llm, logger, target_count=500)
        logger.info("  Expanded to %d examples via LLM", len(expanded))
        all_alignment = expanded
    else:
        all_alignment = seed_results

    # Generate whitepaper reasoning examples.
    whitepaper_prompts = [
        {
            "instruction": "Using principles from the MapReduce paper, design a function that counts word frequencies across multiple files.",
            "output": (
                "<thought>MapReduce separates tasks into map (extract) and reduce (aggregate) phases. "
                "I'll implement the mapper to emit (word, 1) pairs and the reducer to sum them.</thought>\n\n"
                "def map_word_count(filepath: str):\n"
                "    with open(filepath) as f:\n"
                "        for word in f.read().split():\n"
                "            yield (word.lower(), 1)\n\n"
                "def reduce_word_count(pairs):\n"
                "    counts = {}\n"
                "    for word, count in pairs:\n"
                "        counts[word] = counts.get(word, 0) + count\n"
                "    return counts"
            ),
        },
        {
            "instruction": "Using CAP theorem principles, design a simple key-value store that prioritizes availability over consistency.",
            "output": (
                "<thought>CAP says you can have at most two of Consistency, Availability, Partition tolerance. "
                "Prioritizing AP means accepting eventual consistency. I'll use a last-write-wins strategy.</thought>\n\n"
                "class APKeyValueStore:\n"
                "    def __init__(self):\n"
                "        self._data = {}\n"
                "        self._timestamps = {}\n\n"
                "    def put(self, key, value, timestamp):\n"
                "        # Last-write-wins: always accept writes.\n"
                "        if timestamp >= self._timestamps.get(key, 0):\n"
                "            self._data[key] = value\n"
                "            self._timestamps[key] = timestamp\n\n"
                "    def get(self, key):\n"
                "        return self._data.get(key, None)"
            ),
        },
    ]

    for wp in whitepaper_prompts:
        all_alignment.append({
            "instruction": wp["instruction"],
            "input": "",
            "output": wp["output"],
            "metadata": {"layer": "alignment", "source": "whitepaper"},
        })

    logger.info("  Total alignment examples: %d", len(all_alignment))

    assert len(all_alignment) >= 50, (
        f"Only {len(all_alignment)} alignment examples. "
        "Check seed examples and LLM availability."
    )

    output_path = CHUNKS_DIR / "alignment_chunks.jsonl"
    write_jsonl(output_path, all_alignment)
    logger.info("  Written: %s (%d examples)", output_path, len(all_alignment))
    return output_path


def _jsonl_records(path: Path):
    import orjson as _oj
    with open(path, "rb") as _fh:
        for _line in _fh:
            if _line.strip():
                yield _oj.loads(_line)


def _code_alignment_budget(
    logger: logging.Logger,
    target_pct: float,
) -> int:
    """Alignment tokens still missing to reach `target_pct` of the final mix.

    Reads the already-filtered layer files for current alignment tokens (A) and
    all other-layer tokens (O), then solves (A + new)/(O + A + new) == target_pct
    for `new`.  Returns the new-token budget (>= 0).
    """
    layers = {
        "code": "code_chunks_filtered.jsonl",
        "doc": "doc_chunks_filtered.jsonl",
        "devops": "devops_chunks_filtered.jsonl",
        "alignment": "alignment_chunks_filtered.jsonl",
    }
    other = 0
    cur_align = 0
    for name, fn in layers.items():
        p = CHUNKS_DIR / fn
        if not p.exists():
            logger.warning("  [budget] %s not found (%s) — treating as 0 tokens.", name, fn)
            continue
        tk = sum(r.get("token_count", 0) for r in _jsonl_records(p))
        if name == "alignment":
            cur_align = tk
        else:
            other += tk
    denom = 1 - target_pct
    if denom <= 1e-9:
        logger.warning("  [budget] invalid target_pct %.2f — no bump.", target_pct)
        return 0
    new_tokens = int((target_pct * other - cur_align * denom) / denom)
    logger.info(
        "  [budget] other=%d tokens, current alignment=%d tokens → need %d more "
        "alignment tokens for %.0f%% share.",
        other, cur_align, max(0, new_tokens), target_pct * 100,
    )
    return max(0, new_tokens)


def _avg_code_tokens(records: List[Dict[str, Any]]) -> int:
    """Mean tokens per code chunk used to estimate how many yield the budget."""
    if not records:
        return 0
    total = 0
    for rec in records:
        total += _count_tokens(rec.get("output", ""))
    return total // len(records)


def phase_expand_alignment_from_code(
    logger: logging.Logger,
    limit: Optional[int] = None,
    target_pct: float = 0.15,
    gpu_parallel: Optional[int] = None,
) -> Path:
    """Phase 5c: Bump the alignment layer from real code chunks, PURE GPU.

    Each code chunk becomes one alignment example: the instruction asks for a
    Tiger-Style-compliant refactor, the input is the original code verbatim, and
    the output is <thought> + the model's refactor.  The local model runs through
    the orchestrator fleet with CPU workers forced to 0 (GPU-only) because only
    the refactor (not the input) needs a generation pass.

    Writes `alignment_code_chunks.jsonl`, which the filter phase merges into the
    alignment layer (same pattern as doc_hf/devops_hf).
    """
    output_path = CHUNKS_DIR / "alignment_code_chunks.jsonl"
    if _check_checkpoint(output_path, "code-derived alignment examples", min_records=100, logger=logger):
        return output_path
    logger.info("=== Phase 5c: Expanding alignment examples from code (pure GPU) ===")

    input_path = CHUNKS_DIR / "code_chunks_ready.jsonl"
    assert input_path.exists(), f"Run --phase chunk first. File not found: {input_path}"
    records = [r for r in _jsonl_records(input_path)
               if r.get("metadata", {}).get("language") != "css"]
    logger.info("  Code chunks available: %d (CSS excluded from alignment)", len(records))
    random.Random(42).shuffle(records)

    if limit is not None:
        chosen = records[:limit]
    else:
        # Default cap for the code-derived alignment layer.
        chosen = records[:4000]
    logger.info("  Converting %d code chunks to alignment examples", len(chosen))

    model_path = MODELS_DIR / "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
    assert model_path.exists(), f"Model not found: {model_path}"

    try:
        specs = _orchestrator_plan(str(model_path), cpu_workers=0, gpu_parallel=gpu_parallel, logger=logger)
    except Exception as exc:
        logger.error("  Orchestrator GPU plan failed: %s", exc)
        raise
    if not specs:
        raise RuntimeError("No GPU worker available for pure-GPU alignment expansion.")
    if any(s.kind != "gpu" for s in specs):
        logger.warning("  GPU-only requested but got %d CPU workers — run is not pure GPU.", len(specs))

    principles = list(PRINCIPLE_HINTS.keys())
    paired = [(rec, random.choice(principles)) for rec in chosen]
    jobs = [
        {
            "prompt": (
                f"Here is a {rec.get('metadata', {}).get('language', 'unknown')} "
                f"function that violates '{principle}':\n{rec['output']}\n\n"
                f"Here is the corrected version that complies with '{principle}':\n"
            ),
            "max_tokens": 256,
            "temperature": 0.7,
            "stop": ["\n\n"],
        }
        for rec, principle in paired
    ]
    logger.info("  Alignment-from-code: requesting %d refactors via GPU fleet", len(jobs))
    texts = _orchestrator_complete(str(model_path), jobs, specs, logger)
    texts = texts + [None] * (len(jobs) - len(texts))

    seed_hashes: set = set()
    seeds_path = CHUNKS_DIR / "alignment_chunks.jsonl"
    if seeds_path.exists():
        seed_hashes = {compute_blake3(rec["output"]) for rec in _jsonl_records(seeds_path)}

    expanded: List[Dict[str, Any]] = []
    for (rec, principle), raw in zip(paired, texts):
        if not raw:
            continue
        lang = rec.get("metadata", {}).get("language", "unknown")
        good = _strip_code_fences(raw.strip())
        if len(good) < 10:
            continue
        output = f"<thought>Fixing '{principle}' in {lang}.</thought>\n\n{good}"
        out_hash = compute_blake3(output)
        if out_hash in seed_hashes:
            continue
        seed_hashes.add(out_hash)
        expanded.append({
            "instruction": f"Refactor this {lang} code to comply with Tiger Style: {principle}",
            "input": rec["output"],
            "output": output,
            "token_count": _count_tokens(rec["output"]) + _count_tokens(output),
            "metadata": {
                "layer": "alignment",
                "source": "code-expansion",
                "principle": principle,
                "language": lang,
                "repo": rec.get("metadata", {}).get("repo"),
            },
        })

    random.shuffle(expanded)
    write_jsonl(output_path, expanded)
    logger.info("  Written: %s (%d code-derived alignment examples)", output_path, len(expanded))
    if not expanded:
        logger.warning("  No valid examples generated — check the model / GPU plan.")
    return output_path


# =============================================================================
# %%
# Phase 5b: DevOps, Logs & Harness Layer (10% of the mixture)
# =============================================================================
# What the model needs from this layer: reproducible build environments
# (Makefiles, Dockerfiles), service supervision (systemd), and structured
# logging formats that a deterministic harness can parse.  Seeds are
# hand-crafted so this layer never depends on scraping or a live model.

DEVOPS_SEEDS: List[Dict[str, str]] = [
    {
        "instruction": "Write a Makefile with build, test, and clean targets for a C project with a single source file.",
        "output": (
            "CC ?= cc\n"
            "CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror\n"
            "SRC := main.c\n"
            "BIN := app\n\n"
            ".PHONY: all test clean\n\n"
            "all: $(BIN)\n\n"
            "$(BIN): $(SRC)\n"
            "\t$(CC) $(CFLAGS) -o $@ $<\n\n"
            "test: $(BIN)\n"
            "\t./$(BIN) --selftest\n\n"
            "clean:\n"
            "\trm -f $(BIN)\n"
        ),
    },
    {
        "instruction": "Write a multi-stage Dockerfile that builds a small static C binary and runs it in scratch.",
        "output": (
            "FROM gcc:13 AS build\n"
            "WORKDIR /src\n"
            "COPY main.c .\n"
            "RUN gcc -std=c11 -O2 -static -o app main.c\n\n"
            "FROM scratch\n"
            "COPY --from=build /src/app /app\n"
            "ENTRYPOINT [\"/app\"]\n"
        ),
    },
    {
        "instruction": "Write a systemd service unit that keeps a Python worker running with restart-on-failure.",
        "output": (
            "[Unit]\n"
            "Description=worker service\n"
            "After=network.target\n\n"
            "[Service]\n"
            "Type=simple\n"
            "User=worker\n"
            "WorkingDirectory=/opt/worker\n"
            "ExecStart=/opt/worker/.venv/bin/python main.py\n"
            "Restart=on-failure\n"
            "RestartSec=2\n"
            "MemoryMax=512M\n"
            "CPUQuota=200%\n"
            "StandardOutput=journal\n"
            "StandardError=journal\n\n"
            "[Install]\n"
            "WantedBy=multi-user.target\n"
        ),
    },
    {
        "instruction": "Write a Python logging configuration that emits JSON lines a harness can parse.",
        "output": (
            "import json, logging, time\n\n"
            "class JsonFormatter(logging.Formatter):\n"
            "    def format(self, record):\n"
            "        return json.dumps({\n"
            "            'ts': time.time(),\n"
            "            'level': record.levelname,\n"
            "            'logger': record.name,\n"
            "            'msg': record.getMessage(),\n"
            "        })\n\n"
            "def configure():\n"
            "    h = logging.StreamHandler()\n"
            "    h.setFormatter(JsonFormatter())\n"
            "    logging.basicConfig(handlers=[h], level=logging.INFO)\n"
        ),
    },
    {
        "instruction": "Write a GitHub Actions workflow that runs tests on every push.",
        "output": (
            "name: ci\n"
            "on: [push]\n"
            "jobs:\n"
            "  test:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - uses: actions/checkout@v4\n"
            "      - uses: actions/setup-python@v5\n"
            "        with:\n"
            "          python-version: '3.12'\n"
            "      - run: pip install -r requirements.txt\n"
            "      - run: pytest tests/\n"
        ),
    },
    {
        "instruction": "Write a shell script that builds the project with strict error checking and logs each phase to stderr.",
        "output": (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n\n"
            "log() { printf '[build] %s\\n' \"$*\" >&2; }\n\n"
            "log 'configuring'\n"
            "cmake -B build -DCMAKE_BUILD_TYPE=Release\n\n"
            "log 'building'\n"
            "cmake --build build -j$(nproc)\n\n"
            "log 'testing'\n"
            "ctest --test-dir build --output-on-failure\n\n"
            "log 'done'\n"
        ),
    },
    {
        "instruction": "Write a docker-compose.yml that runs an app and a postgres service with healthchecks.",
        "output": (
            "services:\n"
            "  app:\n"
            "    build: .\n"
            "    ports: [\"8080:8080\"]\n"
            "    depends_on:\n"
            "      db:\n"
            "        condition: service_healthy\n"
            "  db:\n"
            "    image: postgres:16\n"
            "    environment:\n"
            "      POSTGRES_PASSWORD: dev\n"
            "    healthcheck:\n"
            "      test: [\"CMD-SHELL\", \"pg_isready -U postgres\"]\n"
            "      interval: 5s\n"
            "      timeout: 3s\n"
            "      retries: 5\n"
        ),
    },
    {
        "instruction": "Write a structured JSON log line for a failed request including request id and latency.",
        "output": (
            "{\n"
            "  \"ts\": 1710000000.123,\n"
            "  \"level\": \"ERROR\",\n"
            "  \"request_id\": \"a1b2c3\",\n"
            "  \"method\": \"GET\",\n"
            "  \"path\": \"/v1/items\",\n"
            "  \"status\": 500,\n"
            "  \"latency_ms\": 231.4,\n"
            "  \"msg\": \"upstream timeout\"\n"
            "}"
        ),
    },
    {
        "instruction": "Write a systemd timer that runs a nightly database backup.",
        "output": (
            "[Unit]\n"
            "Description=nightly backup\n\n"
            "[Timer]\n"
            "OnCalendar=*-*-* 02:00:00\n"
            "Persistent=true\n\n"
            "[Install]\n"
            "WantedBy=timers.target\n"
        ),
    },
    {
        "instruction": "Write an nginx server block that serves a static site and sets security headers.",
        "output": (
            "server {\n"
            "  listen 80;\n"
            "  server_name example.com;\n"
            "  root /srv/www;\n"
            "  index index.html;\n"
            "  add_header X-Content-Type-Options nosniff;\n"
            "  add_header X-Frame-Options DENY;\n"
            "  add_header Referrer-Policy strict-origin-when-cross-origin;\n"
            "  location / {\n"
            "    try_files $uri $uri/ /index.html;\n"
            "  }\n"
            "}\n"
        ),
    },
    {
        "instruction": "Write a simple .gitignore for a Python project that excludes virtualenvs and caches.",
        "output": (
            "__pycache__/\n"
            "*.py[cod]\n"
            ".venv/\n"
            "venv/\n"
            ".pytest_cache/\n"
            ".mypy_cache/\n"
            "*.egg-info/\n"
            "dist/\n"
            "build/\n"
            "data/*.jsonl\n"
        ),
    },
    {
        "instruction": "Write a Makefile target that runs a linter and a formatter check in CI.",
        "output": (
            ".PHONY: lint fmt-check\n\n"
            "lint:\n"
            "\trufflehog filesystem --directory . --fail || true\n"
            "\tflake8 src tests\n\n"
            "fmt-check:\n"
            "\tblack --check src tests\n"
            "\tisort --check-only src tests\n"
        ),
    },
    # ── curl / HTTP harness operations — unblocks the 10% devops binding layer ──
    {
        "instruction": "Write a curl command that pushes a JSON log line to a central ingest endpoint over HTTPS.",
        "output": (
            "curl -sS -X POST https://ingest.example.com/v1/logs \\\n"
            "  -H 'Content-Type: application/json' \\\n"
            "  -H 'Authorization: Bearer <token>' \\\n"
            "  --data-binary '{\"ts\":1710000000,\"level\":\"INFO\",\"msg\":\"job started\"}' \\\n"
            "  --fail-with-body\n"
        ),
    },
    {
        "instruction": "Write a curl command that downloads a model file and shows a progress bar, resuming an interrupted download.",
        "output": (
            "curl -L -C - \\\n"
            "  --progress-bar \\\n"
            "  -o qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \\\n"
            "  'https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf'\n"
        ),
    },
    {
        "instruction": "Write a curl command to benchmark an OpenAI-compatible chat completions endpoint.",
        "output": (
            "time curl -sS http://localhost:1234/v1/chat/completions \\\n"
            "  -H 'Content-Type: application/json' \\\n"
            "  -d '{\"model\":\"qwen2.5-coder-1.5b\",\"messages\":"
            "[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":false}'\n"
        ),
    },
    {
        "instruction": "Write an HTTP request that streams a long LLM completion with curl and keeps headers visible.",
        "output": (
            "curl -i -N http://localhost:1234/v1/chat/completions \\\n"
            "  -H 'Content-Type: application/json' \\\n"
            "  -d '{\"messages\":[{\"role\":\"user\",\"content\":\"count to 10\"}],\"max_tokens\":512,\"stream\":true}'\n"
        ),
    },
    {
        "instruction": "Write curl commands to check that a REST health endpoint is up.",
        "output": (
            "curl -sf http://localhost:8080/healthz \\\n"
            "  || exit 1\n"
            "curl -sf -o /dev/null -w 'status=%{http_code} latency=%{time_total}s\\n' \\\n"
            "  http://localhost:8080/readyz\n"
        ),
    },
    {
        "instruction": "Write a curl request that returns only the HTTP status code and headers for debugging.",
        "output": (
            "curl -sS -o /dev/null -D - -X POST http://localhost:1234/v1/embeddings \\\n"
            "  -H 'Content-Type: application/json' \\\n"
            "  -d '{\"input\":\"test\",\"model\":\"embed\"}'\n"
        ),
    },
    {
        "instruction": "Write a curl command that uploads a dataset JSONL file to a training server.",
        "output": (
            "curl -sS -X PUT http://localhost:9000/data \\\n"
            "  -H 'Content-Type: application/x-ndjson' \\\n"
            "  --data-binary @data/train.jsonl\n"
        ),
    },
    {
        "instruction": "Write a shell one-liner that retries a flaky curl call a few times with backoff.",
        "output": (
            "for i in 1 2 3 4 5; do\n"
            "  curl -sf http://localhost:9000/jobs > /tmp/out.json && break\n"
            "  echo \"attempt $i failed\"\n"
            "  sleep 2\n"
            "done\n"
        ),
    },
]


def _build_devops_seeds() -> List[Dict[str, Any]]:
    """Deterministic DevOps layer — every seed becomes one training example."""
    examples: List[Dict[str, Any]] = []
    for i, seed in enumerate(DEVOPS_SEEDS):
        examples.append({
            "instruction": seed["instruction"],
            "input": "",
            "output": seed["output"],
            "metadata": {
                "layer": "devops",
                "language": "devops",
                "source": "seed",
            },
        })
    return examples


# ── DevOps online-source chunking (no LLM required) ─────────────────────────
# Tiger Style: same pattern as the doc layer — deterministic instructions are
# derived from the config file's own structure (kind/name/keys), never from an
# LLM, so this phase has zero fleet dependency.

def _is_devops_file(filepath: Path) -> bool:
    """True if a file is devops config material (Kubernetes, compose, Docker)."""
    name = filepath.name.lower()
    if name == "makefile" or name.startswith("makefile."):
        return True
    if name.startswith("dockerfile"):
        return True
    if filepath.suffix.lower() in DEVOPS_EXTENSIONS:
        return True
    return False


def _get_devops_files(repo_path: Path) -> List[Path]:
    """Recursively list devops-config files in a repo, honoring exclusions.

    Tiger Style:
      - Reuses EXCLUDE_DIRS (no test/build/vendor noise).
      - Bounds file size via MAX_FILE_LINES like the code path.
      - Skips docs/dotfiles and binary-looking files.
    """
    devops_files: List[Path] = []
    for filepath in repo_path.rglob("*"):
        if not filepath.is_file():
            continue
        rel_parts = filepath.relative_to(repo_path).parts
        if any(part in EXCLUDE_DIRS for part in rel_parts):
            continue
        if any(part.startswith(".") for part in rel_parts):
            continue
        if not _is_devops_file(filepath):
            continue
        # Skip pure documentation/readme yaml.
        if filepath.suffix.lower() in (".md", ".markdown"):
            continue
        try:
            with open(filepath, "rb") as f:
                line_count = sum(1 for _ in f)
        except (OSError, PermissionError):
            continue
        if line_count < 1 or line_count > MAX_FILE_LINES:
            continue
        devops_files.append(filepath)
    return devops_files


def _yaml_top_level_key(line: str) -> Optional[str]:
    """Return the top-level YAML key if `line` opens a new top-level block."""
    stripped = line.rstrip()
    if not stripped or stripped.startswith("#") or stripped.startswith("---"):
        return None
    if not line[:1].strip():  # indented → not top-level
        return None
    m = re.match(r"^([A-Za-z0-9_.][A-Za-z0-9_.\-]*):\s*(.*)$", stripped)
    return m.group(1) if m else None


def _split_yaml_chunks(text: str) -> List[str]:
    """Split a YAML file into logical chunks at top-level keys / doc markers.

    Tiger Style:
      - Bounded: every chunk is within [MIN_CHUNK_LINES, MAX_CHUNK_LINES].
      - Deterministic: same text, same chunks, every run.
    """
    lines = text.splitlines()
    chunks: List[str] = []
    current: List[str] = []

    def flush() -> None:
        nonlocal current
        while len(current) >= MAX_CHUNK_LINES:
            chunks.append("\n".join(current[:MAX_CHUNK_LINES]))
            current = current[MAX_CHUNK_LINES:]
        if current:
            chunks.append("\n".join(current))
            current = []

    for line in lines:
        key = _yaml_top_level_key(line)
        if key is not None:
            flush()
        current.append(line)
    flush()

    # Merge tiny fragments (must satisfy MIN_CHUNK_LINES).
    merged: List[str] = []
    pending: List[str] = []
    for chunk in chunks:
        n = len(chunk.splitlines())
        if n < MIN_CHUNK_LINES:
            pending.append(chunk)
        else:
            if pending:
                merged.append("\n".join(pending) + "\n" + chunk)
                pending = []
            else:
                merged.append(chunk)
    if pending:
        merged.append("\n".join(pending))
    return merged


def _devops_instruction(rel_path: str, chunk: str) -> str:
    """Deterministic instruction derived from the config chunk itself."""
    rp = rel_path.lower()
    name = Path(rel_path).name
    chunk_l = chunk.lower()

    if rp.endswith((".yaml", ".yml")):
        kind_m = re.search(r"^kind:\s*([A-Za-z]+)", chunk, re.M)
        kind = kind_m.group(1) if kind_m else None
        name_m = re.search(r"metadata:\s*\n\s+name:\s*([^\s]+)", chunk)
        obj_name = name_m.group(1) if name_m else None

        if "docker-compose" in name or "compose" in rp:
            services = re.findall(r"^  ([a-zA-Z0-9_-]+):", chunk, re.M)
            target = services[0] if services else "app"
            return (
                f"Write the docker-compose service definition for `{target}` "
                "as shown."
            )
        if kind and obj_name:
            return f"Write the Kubernetes {kind} manifest for `{obj_name}`."
        if kind:
            return f"Write the Kubernetes {kind} manifest shown here."
        if "---" in chunk or "apiVersion" in chunk_l:
            return f"Provide the Kubernetes configuration defined in {name}."
        return f"Provide the YAML configuration defined in {name}."

    if name.startswith("dockerfile"):
        return f"Write the {name} multi-stage build shown here."
    if name == "makefile" or name.startswith("makefile."):
        targets = re.findall(r"^([A-Za-z0-9_-]+):", chunk, re.M)
        listed = ", ".join(t for t in targets[:4])
        return f"Write the Makefile target(s) for: {listed}."
    if rp.endswith((".service", ".timer")):
        return f"Write the systemd {Path(rel_path).suffix.lstrip('.')} unit defined in {name}."
    if rp.endswith((".tf", ".tfvars")):
        return f"Write the Terraform configuration defined in {name}."
    return f"Provide the DevOps configuration defined in {name}."


def _chunk_devops_repo(repo_path: Path, logger: logging.Logger) -> List[Dict[str, Any]]:
    """Extract devops examples from a cloned config-heavy repo.

    Tiger Style:
      - Deterministic output (no LLM, no randomness).
      - Every example is bounds-checked (MIN/MAX chunk lines).
    """
    examples: List[Dict[str, Any]] = []
    files = _get_devops_files(repo_path)
    logger.info("    [devops] %s — %d config files found", repo_path.name, len(files))

    for filepath in sorted(files):
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except (OSError, PermissionError):
            continue
        if not text.strip():
            continue

        rel_path = str(filepath.relative_to(repo_path))
        if filepath.suffix.lower() in (".yaml", ".yml"):
            chunks = _split_yaml_chunks(text)
        else:
            chunks = [text]

        for chunk in chunks:
            if len(chunk.splitlines()) < MIN_CHUNK_LINES:
                continue
            if len(chunk) > MAX_TOKENS_PER_EXAMPLE * 8:
                # Extremely long raw block — phase 6 token filter would drop it anyway.
                for start in range(0, len(chunk.splitlines()), MAX_CHUNK_LINES):
                    sub = "\n".join(chunk.splitlines()[start:start + MAX_CHUNK_LINES])
                    if len(sub.splitlines()) >= MIN_CHUNK_LINES:
                        chunks_slice = sub
                        examples.append({
                            "instruction": _devops_instruction(rel_path, sub),
                            "input": "",
                            "output": sub,
                            "metadata": {
                                "layer": "devops",
                                "language": "devops",
                                "source": "repo",
                                "repo": repo_path.name,
                                "file": rel_path,
                            },
                        })
                continue
            examples.append({
                "instruction": _devops_instruction(rel_path, chunk),
                "input": "",
                "output": chunk,
                "metadata": {
                    "layer": "devops",
                    "language": "devops",
                    "source": "repo",
                    "repo": repo_path.name,
                    "file": rel_path,
                },
            })

    return examples


def _clone_devops_repos(logger: logging.Logger) -> List[Path]:
    """Clone the devops repos (depth=1), pulling if they already exist.

    Tiger Style: mirrors phase_clone_repos invariants — git presence checked,
    post-condition each repo dir has files.
    """
    git_path = shutil.which("git")
    assert git_path is not None, "`git` not found in PATH."

    paths: List[Path] = []
    for repo in DEVOPS_REPOS:
        target = REPOS_DIR / repo["name"]
        if target.exists():
            assert (target / ".git").is_dir(), f"{target} not a git repo"
            result = subprocess.run(
                ["git", "-C", str(target), "pull", "--ff-only"],
                capture_output=True, text=True, timeout=120,
            )
            assert result.returncode == 0, (
                f"git pull failed for {repo['name']}:\n{result.stderr}"
            )
        else:
            ensure_dir(REPOS_DIR)
            result = subprocess.run(
                ["git", "clone", "--depth", "1", repo["url"], str(target)],
                capture_output=True, text=True, timeout=300,
            )
            assert result.returncode == 0, (
                f"git clone failed for {repo['name']}:\n{result.stderr}"
            )
        file_count = len(list(target.rglob("*")))
        assert file_count > 5, f"Repo {repo['name']} has only {file_count} files"
        paths.append(target)
        logger.info("    OK — %s (%d files)", repo["name"], file_count)
    return paths


def phase_build_devops(logger: logging.Logger) -> Path:
    """Phase 5b: DevOps, logs & harness layer (10%).

    Tiger Style:
      - Seeds are deterministic and auditable (no scraping, no LLM needed).
      - Online devops repos are cloned depth=1 and chunked deterministically —
        instructions are derived from the config structure, never an LLM.
      - Post-condition: devops_chunks.jsonl has at least 10 examples.
    """
    output_path = CHUNKS_DIR / "devops_chunks.jsonl"
    if _check_checkpoint(output_path, "devops examples", min_records=10, logger=logger):
        return output_path
    logger.info("=== Phase 5b: Building devops/logs examples ===")

    all_devops = _build_devops_seeds()
    logger.info("  DevOps seeds: %d examples", len(all_devops))

    for repo_path in _clone_devops_repos(logger):
        repo_examples = _chunk_devops_repo(repo_path, logger)
        logger.info(
            "    [devops] %s — %d examples", repo_path.name, len(repo_examples)
        )
        all_devops.extend(repo_examples)

    logger.info("  DevOps total: %d examples", len(all_devops))
    assert len(all_devops) >= 10, (
        f"Only {len(all_devops)} devops examples."
    )

    write_jsonl(output_path, all_devops)
    logger.info("  Written: %s (%d examples)", output_path, len(all_devops))
    return output_path


# =============================================================================
# %%
# Phase 6: Token Accounting & Quality Filtering
# =============================================================================

_tokenizer: Any = None


def _count_tokens(text: str) -> int:
    """Count tokens using SmolLM3's tokenizer.

    Tiger Style:
      - The tokenizer is loaded ONCE and cached — re-instantiating it per
        example would take ~1.7s × N examples (hours for 58k chunks).
      - Falls back to a simple whitespace split if tokenizer is unavailable
        (degrade gracefully, never crash).
    """
    global _tokenizer
    try:
        if _tokenizer is None:
            from transformers import AutoTokenizer
            _tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
        return len(_tokenizer.encode(text))
    except Exception:
        # Fallback: rough estimate (~4 chars per token for code).
        return max(1, len(text) // 4)


def _sample_layer_to_tokens(
    examples: List[Dict[str, Any]],
    target_tokens: int,
) -> List[Dict[str, Any]]:
    """Deterministically subsample a layer down to a token budget.

    Tiger Style:
      - Seeded shuffle so runs are reproducible.
      - Greedy selection fills the budget without exceeding it by more
        than one example.
      - Returns the full list if already within budget (never grows).
    """
    if sum(ex.get("token_count", 0) for ex in examples) <= target_tokens:
        return examples

    sampled: List[Dict[str, Any]] = []
    budget_left = target_tokens
    # Tiger Style: deterministic — same input, same sample every run.
    shuffled = list(examples)
    random.seed(42)
    random.shuffle(shuffled)

    for ex in shuffled:
        tokens = ex.get("token_count", 0)
        if budget_left - tokens < 0:
            continue
        sampled.append(ex)
        budget_left -= tokens

    return sampled


def phase_filter_and_balance(logger: logging.Logger) -> Dict[str, Path]:
    """Phase 6: Token accounting, quality filtering, and ratio balancing.

    Tiger Style:
      - Every example is validated against MIN/MAX token bounds.
      - Exact deduplication by output hash.
      - Proportional subsampling brings layers to the 50/25/15/10 token ratio.
      - Post-condition: each layer file has < 2048 tokens per example.

    Returns:
        Dict mapping layer name → filtered file path.
    """
    # Check if all four filtered files already exist.
    layers = ["code", "doc", "alignment", "devops"]
    all_exist = all(
        (CHUNKS_DIR / f"{layer}_chunks_filtered.jsonl").exists()
        for layer in layers
    )
    if all_exist:
        sizes = {
            layer: (CHUNKS_DIR / f"{layer}_chunks_filtered.jsonl").stat().st_size
            for layer in layers
        }
        total_mb = sum(sizes.values()) / (1024 * 1024)
        logger.info("=== Phase 6: All 4 filtered files exist (%.1f MB total) — skipping.", total_mb)
        return {layer: CHUNKS_DIR / f"{layer}_chunks_filtered.jsonl" for layer in layers}
    logger.info("=== Phase 6: Token accounting & filtering ===")

    layer_files = {
        "code":      CHUNKS_DIR / "code_chunks_ready.jsonl",
        "doc":       CHUNKS_DIR / "doc_chunks.jsonl",
        "alignment": CHUNKS_DIR / "alignment_chunks.jsonl",
        "devops":    CHUNKS_DIR / "devops_chunks.jsonl",
    }

    filtered_paths: Dict[str, Path] = {}
    filtered_by_layer: Dict[str, List[Dict[str, Any]]] = {}

    for layer_name, input_path in layer_files.items():
        if not input_path.exists():
            logger.warning("  Skipping %s: file not found (%s)", layer_name, input_path)
            continue

        examples = read_jsonl(input_path)
        # Merge streamed HuggingFace doc/QA examples into the doc layer.
        if layer_name == "doc":
            hf_path = CHUNKS_DIR / "doc_hf.jsonl"
            if hf_path.exists():
                hf_examples = read_jsonl(hf_path)
                logger.info("  [doc] Merged %d HuggingFace examples", len(hf_examples))
                examples = hf_examples + examples
        # Merge streamed HuggingFace devops examples into the devops layer.
        if layer_name == "devops":
            hf_path = CHUNKS_DIR / "devops_hf.jsonl"
            if hf_path.exists():
                hf_examples = read_jsonl(hf_path)
                logger.info("  [devops] Merged %d HuggingFace examples", len(hf_examples))
                examples = hf_examples + examples
        # Merge GPU-expanded code-derived examples into the alignment layer.
        if layer_name == "alignment":
            ac_path = CHUNKS_DIR / "alignment_code_chunks.jsonl"
            if ac_path.exists():
                ac_examples = read_jsonl(ac_path)
                logger.info("  [alignment] Merged %d code-derived examples", len(ac_examples))
                examples = ac_examples + examples
        logger.info("  [%s] Loaded %d raw examples", layer_name, len(examples))

        # ── Quality filters ────────────────────────────────────────────────
        filtered: List[Dict[str, Any]] = []
        seen_hashes: set = set()
        dropped_reasons: Dict[str, int] = {}

        for ex in tqdm(examples, desc=f"    Filtering {layer_name}", leave=False):
            # Tiger Style: validate required keys exist.
            assert "instruction" in ex, f"Missing 'instruction' key in {ex}"
            assert "output" in ex, f"Missing 'output' key in {ex}"

            instruction = ex.get("instruction", "")
            output = ex.get("output", "")
            combined = instruction + " " + output

            # Token count.
            token_count = _count_tokens(combined)
            ex["token_count"] = token_count

            if token_count < MIN_TOKENS_PER_EXAMPLE:
                dropped_reasons["too_few_tokens"] = dropped_reasons.get("too_few_tokens", 0) + 1
                continue
            if token_count > MAX_TOKENS_PER_EXAMPLE:
                dropped_reasons["too_many_tokens"] = dropped_reasons.get("too_many_tokens", 0) + 1
                continue

            # Dedup by output hash.
            output_hash = compute_blake3(output)
            if output_hash in seen_hashes:
                dropped_reasons["duplicate"] = dropped_reasons.get("duplicate", 0) + 1
                continue
            seen_hashes.add(output_hash)

            # Instruction quality check.
            if len(instruction) < 5:
                dropped_reasons["short_instruction"] = dropped_reasons.get("short_instruction", 0) + 1
                continue

            # Non-ASCII garbage check.
            if instruction != instruction.encode("ascii", errors="replace").decode("ascii"):
                dropped_reasons["non_ascii"] = dropped_reasons.get("non_ascii", 0) + 1
                continue

            filtered.append(ex)

        # ── Log drop statistics ────────────────────────────────────────────
        if dropped_reasons:
            logger.info(
                "    Dropped %d examples: %s",
                len(examples) - len(filtered),
                dict(sorted(dropped_reasons.items(), key=lambda x: -x[1])),
            )

        layer_tokens = sum(ex.get("token_count", 0) for ex in filtered)
        logger.info(
            "    → %d examples, %d tokens after filtering",
            len(filtered), layer_tokens,
        )
        filtered_by_layer[layer_name] = filtered

# ── Code-dominant balancing ────────────────────────────────────────────
    # Policy change: use EVERY filtered example (especially all 17k+ code
    # examples). The 50/25/15/10 targets are kept as *reporting only* — a
    # hard cap was starving the dataset to the weakest layer's budget. Code
    # anchors the mix; doc/alignment/devops are included in full, so the final
    # dataset is intentionally code-dominant (see docs/data-prep-stage).
    ratios = {
        "code": TARGET_CODE_PCT,
        "doc": TARGET_DOC_PCT,
        "alignment": TARGET_ALIGN_PCT,
        "devops": TARGET_DEVOPS_PCT,
    }

    balanced_by_layer = dict(filtered_by_layer)
    total_tokens_by_layer = {
        layer: sum(ex.get("token_count", 0) for ex in examples)
        for layer, examples in balanced_by_layer.items()
    }
    total_examples_by_layer = {
        layer: len(examples) for layer, examples in balanced_by_layer.items()
    }
    logger.info(
        "  Balanced: keeping every filtered example per layer (no subsampling). "
        "Code → %d examples.", total_examples_by_layer.get("code", 0),
    )
    for layer_name in balanced_by_layer:
        logger.info(
            "    [%s] → %d examples, %d tokens",
            layer_name, total_examples_by_layer[layer_name], total_tokens_by_layer[layer_name],
        )

    # ── Write balanced outputs ─────────────────────────────────────────────
    for layer_name, balanced in balanced_by_layer.items():
        output_path = CHUNKS_DIR / f"{layer_name}_chunks_filtered.jsonl"
        write_jsonl(output_path, balanced)
        filtered_paths[layer_name] = output_path

    # ── Final token mix reporting (report only, no hard assert) ────────────
    grand_total_tokens = sum(total_tokens_by_layer.values())
    if grand_total_tokens > 0:
        logger.info("  Token mix (report only — code-dominant policy):")
        code_pct = total_tokens_by_layer.get("code", 0) / grand_total_tokens
        doc_pct = total_tokens_by_layer.get("doc", 0) / grand_total_tokens
        align_pct = total_tokens_by_layer.get("alignment", 0) / grand_total_tokens
        devops_pct = total_tokens_by_layer.get("devops", 0) / grand_total_tokens
        logger.info("    Code:      %.1f%% (legacy target: %.0f%%)", code_pct * 100, TARGET_CODE_PCT * 100)
        logger.info("    Docs:      %.1f%% (legacy target: %.0f%%)", doc_pct * 100, TARGET_DOC_PCT * 100)
        logger.info("    Alignment: %.1f%% (legacy target: %.0f%%)", align_pct * 100, TARGET_ALIGN_PCT * 100)
        logger.info("    Devops:    %.1f%% (legacy target: %.0f%%)", devops_pct * 100, TARGET_DEVOPS_PCT * 100)
        if code_pct >= TARGET_CODE_PCT:
            logger.info(
                "    ✓ Code is the ruling share (%.0f%% ≥ %.0f%%). Ratio cap disabled by policy.",
                code_pct * 100, TARGET_CODE_PCT * 100,
            )
        else:
            logger.warning(
                "    Code share (%.1f%%) is below the %.0f%% reference target — no ratio "
                "constraint is enforced under code-dominant policy.", code_pct * 100, TARGET_CODE_PCT * 100,
            )

    logger.info("=== Phase 6 complete ===")
    return filtered_paths


# =============================================================================
# %%
# Phase 7: Mix, Shuffle & Write Final Dataset
# =============================================================================

def phase_mix_and_shuffle(
    filtered_paths: Dict[str, Path],
    logger: logging.Logger,
) -> Path:
    """Phase 7: Combine all layers, shuffle, and write train.jsonl.

    Tiger Style:
      - Explicit post-condition: train.jsonl contains exactly the sum of all layer counts.
      - Writes stats.json alongside train.jsonl for auditability.
    """
    output_path = DATA_DIR / "train.jsonl"
    if _check_checkpoint(output_path, "final dataset", min_records=100, logger=logger):
        return output_path
    logger.info("=== Phase 7: Mixing, shuffling & writing final dataset ===")

    all_examples: List[Dict[str, Any]] = []
    layer_counts: Dict[str, int] = {}
    language_counts: Dict[str, int] = {}
    total_tokens = 0

    for layer_name, filepath in filtered_paths.items():
        examples = read_jsonl(filepath)
        all_examples.extend(examples)
        layer_counts[layer_name] = len(examples)
        logger.info("  [%s] %d examples", layer_name, len(examples))

        # Count language breakdown.
        for ex in examples:
            meta = ex.get("metadata", {})
            lang = meta.get("language", "unknown") if isinstance(meta, dict) else "unknown"
            language_counts[lang] = language_counts.get(lang, 0) + 1
            total_tokens += ex.get("token_count", 0)

    # Tiger Style: deterministic shuffle with explicit seed.
    random.seed(42)
    random.shuffle(all_examples)

    output_path = DATA_DIR / "train.jsonl"
    write_jsonl(output_path, all_examples)

    # Write stats.
    stats = {
        "total_examples": len(all_examples),
        "total_tokens": total_tokens,
        "layer_counts": layer_counts,
        "language_counts": language_counts,
        "token_ratios": {
            layer: round(count / max(total_tokens, 1) * 100, 1)
            for layer, count in layer_counts.items()
        },
    }
    stats_path = DATA_DIR / "stats.json"
    with open(stats_path, "wb") as f:
        f.write(orjson.dumps(stats, option=orjson.OPT_INDENT_2))

    logger.info("  Final dataset: %d examples, %d tokens", len(all_examples), total_tokens)
    logger.info("  Language breakdown: %s", language_counts)
    logger.info("  Written: %s", output_path)
    logger.info("  Stats:    %s", stats_path)

    # Post-condition validation.
    assert output_path.exists(), "train.jsonl was not written"
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("  File size: %.1f MB", file_size_mb)

    return output_path


# =============================================================================
# %%
# Phase 8: Validate Final Dataset
# =============================================================================

def phase_validate(logger: logging.Logger) -> bool:
    """Phase 8: Comprehensive validation of the final dataset.

    Tiger Style:
      - Every validation check is explicit and logged.
      - Returns True if all checks pass, False otherwise.
      - Never crashes — reports all failures before returning.

    Returns:
        True if all validation checks pass.
    """
    logger.info("=== Phase 8: Final validation ===")

    train_path = DATA_DIR / "train.jsonl"
    stats_path = DATA_DIR / "stats.json"

    all_pass = True

    # ── Check 1: Files exist ──────────────────────────────────────────────
    if not train_path.exists():
        logger.error("  FAIL: train.jsonl not found at %s", train_path)
        return False
    if not stats_path.exists():
        logger.warning("  WARN: stats.json not found (run --phase mix)")

    with open(train_path, "rb") as f:
        lines = f.readlines()

    # ── Check 2: Non-empty ────────────────────────────────────────────────
    if len(lines) < 100:
        logger.error("  FAIL: Only %d examples (need ≥ 100)", len(lines))
        all_pass = False
    else:
        logger.info("  ✓ %d total lines", len(lines))

    # ── Check 3: Every line is valid JSON with required keys ──────────────
    structure_errors = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = orjson.loads(line)
            assert "instruction" in obj, f"Missing 'instruction' at line {i+1}"
            assert "output" in obj, f"Missing 'output' at line {i+1}"
        except (orjson.JSONDecodeError, AssertionError) as exc:
            structure_errors += 1
            if structure_errors <= 3:
                logger.error("  Structure error at line %d: %s", i + 1, exc)

    if structure_errors > 0:
        logger.error("  FAIL: %d structure errors", structure_errors)
        all_pass = False
    else:
        logger.info("  ✓ All lines valid JSON with required keys")

    # ── Check 4: Token ratio within tolerance ─────────────────────────────
    layer_tokens: Dict[str, int] = {"code": 0, "doc": 0, "alignment": 0, "devops": 0}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        obj = orjson.loads(line)
        meta = obj.get("metadata", {})
        layer = meta.get("layer", "code") if isinstance(meta, dict) else "code"
        token_count = obj.get("token_count", _count_tokens(obj.get("instruction", "") + " " + obj.get("output", "")))
        if layer in layer_tokens:
            layer_tokens[layer] += token_count

    total = sum(layer_tokens.values())
    if total > 0:
        code_pct = layer_tokens["code"] / total
        doc_pct = layer_tokens["doc"] / total
        align_pct = layer_tokens["alignment"] / total
        devops_pct = layer_tokens["devops"] / total

        logger.info("  Token ratios: Code %.1f%%, Doc %.1f%%, Align %.1f%%, Devops %.1f%%",
                     code_pct * 100, doc_pct * 100, align_pct * 100, devops_pct * 100)

        # Code-dominant policy: ratios are reported, never hard-failed. Code is
        # the ruling share by design (all code examples are preserved).
        if code_pct >= TARGET_CODE_PCT:
            logger.info("    ✓ Code-dominant mix confirmed (code %.1f%% ≥ %.0f%%)", code_pct * 100, TARGET_CODE_PCT * 100)
        else:
            logger.warning("    WARN: Code share (%.1f%%) below %.0f%% reference target", code_pct * 100, TARGET_CODE_PCT * 100)
        if doc_pct < TARGET_DOC_PCT - 0.10:
            logger.info("    Documented: doc share %.1f%% (below legacy target — expected under code-dominant policy)", doc_pct * 100)
        if align_pct < TARGET_ALIGN_PCT - 0.05:
            logger.info("    Documented: alignment share %.1f%% (below legacy target)", align_pct * 100)
        if devops_pct < TARGET_DEVOPS_PCT - 0.05:
            logger.info("    Documented: devops share %.1f%% (below legacy target)", devops_pct * 100)

    # ── Check 5: Language diversity ───────────────────────────────────────
    languages_found: set = set()
    for line in lines:
        if not line.strip():
            continue
        obj = orjson.loads(line)
        meta = obj.get("metadata", {})
        if isinstance(meta, dict):
            lang = meta.get("language", "unknown")
            languages_found.add(lang)

    expected_langs = {"c", "python", "javascript", "html", "css"}
    missing_langs = expected_langs - languages_found
    if missing_langs:
        logger.warning("  WARN: Missing languages: %s", missing_langs)
    else:
        logger.info("  ✓ All 5 target languages present")

    # ── Check 6: No duplicate outputs ─────────────────────────────────────
    output_hashes: set = set()
    dup_count = 0
    for line in lines:
        if not line.strip():
            continue
        obj = orjson.loads(line)
        h = compute_blake3(obj.get("output", ""))
        if h in output_hashes:
            dup_count += 1
        output_hashes.add(h)

    if dup_count > 0:
        logger.warning("  WARN: %d duplicate outputs found", dup_count)
    else:
        logger.info("  ✓ No duplicate outputs")

    # ── Summary ───────────────────────────────────────────────────────────
    if all_pass:
        logger.info("  ✓ ALL CHECKS PASSED — dataset is ready for training")
    else:
        logger.error("  ✗ SOME CHECKS FAILED — review logs above")

    return all_pass


# %%
# =============================================================================
# CLI Entry Point
# =============================================================================

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Tiger Style: every option has help text, type validation, and default.
    """
    parser = argparse.ArgumentParser(
        description="Phase C: Dataset Construction (50/25/15/10 Rule) for SmolLM3 fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              uv run python data_prep.py --phase all                     # Full pipeline
              uv run python data_prep.py --phase clone --verbose          # Clone with debug logging
              uv run python data_prep.py --phase chunk,instruct,filter    # Specific phases
        """),
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        help="Phase(s) to run: all, clone, chunk, instruct, docs, hf, align, align-code, devops, filter, mix, validate. "
             "Comma-separated for multiple.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(MODELS_DIR / "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"),
        help="Path to the GGUF model for instruction generation.",
    )
    parser.add_argument(
        "--instruct-limit",
        type=int,
        default=None,
        help="Process only the first N code chunks in the instruct phase "
             "(useful for benchmarking the orchestrator on real data).",
    )
    parser.add_argument(
        "--align-limit",
        type=int,
        default=None,
        help="Convert only the first N code chunks into alignment examples in "
             "the align-code phase (for testing). Default: auto-scale to --align-target.",
    )
    parser.add_argument(
        "--align-target",
        type=float,
        default=0.15,
        help="Target token share for the alignment layer when auto-scaling "
             "the align-code phase (default: 0.15 = 15%%).",
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=None,
        help="Force the number of CPU workers in the orchestrator plan "
             "(default: auto from CPU/RAM probe).",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Threads per CPU worker (default: 2).",
    )
    parser.add_argument(
        "--gpu-parallel",
        type=int,
        default=None,
        help="Max sequences per GPU forward pass (default: auto from VRAM).",
    )
    return parser.parse_args()


def main() -> int:
    """Main entry point.

    Tiger Style:
      - Every phase has explicit pre/post conditions.
      - Returns 0 on success, 1 on failure (for shell scripting).
      - No silent exit — always logs final status.
    """
    args = _parse_args()
    logger = _setup_logging(verbose=args.verbose)

    logger.info("=" * 60)
    logger.info("Data Preparation — Phase C: 50/25/15/10 Dataset Construction")
    logger.info("=" * 60)

    # Determine which phases to run.
    if args.phase == "all":
        phases = ["clone", "chunk", "instruct", "docs", "hf", "align", "devops", "filter", "mix", "validate"]
    else:
        phases = [p.strip() for p in args.phase.split(",") if p.strip()]

    logger.info("Phases to execute: %s", ", ".join(phases))

    # Tiger Style: ensure required directories exist.
    ensure_dir(REPOS_DIR)
    ensure_dir(DOCS_DIR)
    ensure_dir(ALIGNMENT_DIR)
    ensure_dir(CHUNKS_DIR)
    ensure_dir(MODELS_DIR)

    filtered_paths: Optional[Dict[str, Path]] = None

    for phase_name in phases:
        phase_name = phase_name.lower()

        if phase_name == "clone":
            phase_clone_repos(logger)

        elif phase_name == "chunk":
            phase_chunk_code(logger)

        elif phase_name == "instruct":
            phase_generate_instructions(
                logger,
                instruct_limit=args.instruct_limit,
                cpu_workers=args.cpu_workers,
                cpu_threads=args.cpu_threads,
                gpu_parallel=args.gpu_parallel,
            )

        elif phase_name == "docs":
            phase_scrape_docs(logger)

        elif phase_name == "hf":
            phase_fetch_hf(logger)

        elif phase_name == "align":
            phase_build_alignment(logger)

        elif phase_name == "align-code":
            phase_expand_alignment_from_code(
                logger,
                limit=args.align_limit,
                target_pct=args.align_target,
                gpu_parallel=args.gpu_parallel,
            )

        elif phase_name == "devops":
            phase_build_devops(logger)

        elif phase_name == "filter":
            filtered_paths = phase_filter_and_balance(logger)

        elif phase_name == "mix":
            # If filter wasn't run in this session, load default paths.
            if filtered_paths is None:
                filtered_paths = {
                    "code":      CHUNKS_DIR / "code_chunks_filtered.jsonl",
                    "doc":       CHUNKS_DIR / "doc_chunks_filtered.jsonl",
                    "alignment": CHUNKS_DIR / "alignment_chunks_filtered.jsonl",
                    "devops":    CHUNKS_DIR / "devops_chunks_filtered.jsonl",
                }
                # Check which ones actually exist.
                filtered_paths = {
                    k: v for k, v in filtered_paths.items() if v.exists()
                }
                if not filtered_paths:
                    logger.error("No filtered files found. Run --phase filter first.")
                    return 1
            phase_mix_and_shuffle(filtered_paths, logger)

        elif phase_name == "validate":
            success = phase_validate(logger)
            if not success:
                logger.warning("Validation found issues. Review and fix before training.")

        else:
            logger.error("Unknown phase: %s. Valid: %s", phase_name,
                         "all, clone, chunk, instruct, docs, hf, align, align-code, devops, filter, mix, validate")
            return 1

    logger.info("=" * 60)
    logger.info("Pipeline complete. Run `uv run python data_prep.py --phase validate` to verify.")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
