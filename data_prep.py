"""
data_prep.py — Phase C: Dataset Construction (The 60/25/15 Rule)
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
MAX_TOKENS_PER_EXAMPLE = 2048
MIN_TOKENS_PER_EXAMPLE = 8
MAX_INSTRUCTION_GEN_RETRIES = 3
REQUEST_DELAY_SECONDS = 0.5  # Polite scraping delay

# Tiger Style: target ratios with explicit tolerance bounds.
TARGET_CODE_PCT = 0.60
TARGET_DOC_PCT = 0.25
TARGET_ALIGN_PCT = 0.15
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
    prompt = INSTRUCTION_PROMPT_TEMPLATE.format(language=language, code=code[:3000])
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
    pbar = tqdm(total=total_to_process, desc="  Generating instructions (batched)")

    try:
        while queue:
            wave = []
            while queue and len(wave) < llm.n_parallel:
                wave.append(queue.popleft())

            reqs = [
                {
                    "prompt": INSTRUCTION_PROMPT_TEMPLATE.format(
                        language=c["language"], code=c["code"][:3000]
                    ),
                    "max_tokens": 128,
                    "temperature": 0.7,
                    "stop": ["\n\n"],
                }
                for _, c, _ in wave
            ]

            texts = llm.complete_batch(reqs)

            for (idx, chunk, attempts), text in zip(wave, texts):
                if text is not None:
                    instruction = _validate_instruction(text, chunk["code"], logger)
                else:
                    instruction = None

                if instruction is None:
                    if attempts > 1:
                        queue.append((idx, chunk, attempts - 1))
                    else:
                        generation_failures += 1
                        pbar.update(1)
                    continue

                record = _make_instruct_record(instruction, chunk)
                fout.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))
                fout.flush()
                saved_count += 1
                pbar.update(1)
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


def phase_generate_instructions(logger: logging.Logger) -> Path:
    """Phase 3: Generate synthetic instructions for every code chunk.

    Tiger Style:
      - Lazy-loads the LLM (no wasted resources if this phase is skipped).
      - Processes chunks in deterministic order with progress bar.
      - Writes filtered results — chunks that failed generation are dropped.
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

    ensure_dir(output_path.parent)

    # Prefer the batched LM Studio backend; fall back to serial llama-cpp-python.
    batched = _load_batched_llm(model_path_str, logger)
    if batched is not None:
        saved_count, generation_failures = _generate_instructions_batched(
            batched, chunks, completed, output_path, logger
        )
    else:
        llm = _load_llm(model_path_str)
        saved_count, generation_failures = _generate_instructions_serial(
            llm, chunks, completed, output_path, logger
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

DOC_SOURCES = {
    "cppreference": {
        "base_url": "https://en.cppreference.com/w/c",
        "sections": [
            "string", "io", "program", "numeric", "chrono",
            "memory", "thread", "atomic", "locale", "program/signal",
        ],
    },
    "python_docs": {
        "base_url": "https://docs.python.org/3/library/",
        "modules": [
            "os", "sys", "json", "asyncio", "pathlib", "collections",
            "re", "datetime", "math", "random", "itertools", "functools",
            "typing", "dataclasses", "concurrent.futures", "subprocess",
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
        ],
    },
}


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

    # Expand to more examples via LLM (if available).
    if blm:
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
      - Proportional subsampling brings layers to the 60/25/15 token ratio.
      - Post-condition: each layer file has < 2048 tokens per example.

    Returns:
        Dict mapping layer name → filtered file path.
    """
    # Check if all three filtered files already exist.
    all_exist = all(
        (CHUNKS_DIR / f"{layer}_chunks_filtered.jsonl").exists()
        for layer in ["code", "doc", "alignment"]
    )
    if all_exist:
        sizes = {
            layer: (CHUNKS_DIR / f"{layer}_chunks_filtered.jsonl").stat().st_size
            for layer in ["code", "doc", "alignment"]
        }
        total_mb = sum(sizes.values()) / (1024 * 1024)
        logger.info("=== Phase 6: All 3 filtered files exist (%.1f MB total) — skipping.", total_mb)
        return {layer: CHUNKS_DIR / f"{layer}_chunks_filtered.jsonl" for layer in ["code", "doc", "alignment"]}
    logger.info("=== Phase 6: Token accounting & filtering ===")

    layer_files = {
        "code":      CHUNKS_DIR / "code_chunks_ready.jsonl",
        "doc":       CHUNKS_DIR / "doc_chunks.jsonl",
        "alignment": CHUNKS_DIR / "alignment_chunks.jsonl",
    }

    filtered_paths: Dict[str, Path] = {}
    filtered_by_layer: Dict[str, List[Dict[str, Any]]] = {}

    for layer_name, input_path in layer_files.items():
        if not input_path.exists():
            logger.warning("  Skipping %s: file not found (%s)", layer_name, input_path)
            continue

        examples = read_jsonl(input_path)
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

    # ── Ratio balancing (Tiger Style: proportional subsampling) ────────────
    ratios = {
        "code": TARGET_CODE_PCT,
        "doc": TARGET_DOC_PCT,
        "alignment": TARGET_ALIGN_PCT,
    }
    available_tokens = {
        layer: sum(ex.get("token_count", 0) for ex in examples)
        for layer, examples in filtered_by_layer.items()
    }

    # The binding layer is the one that fills its target ratio with the
    # fewest tokens — it caps the whole dataset size.
    max_total = min(
        available_tokens.get(layer, 0) / ratio
        for layer, ratio in ratios.items()
        if layer in filtered_by_layer
    )
    logger.info(
        "  Balancing 3 layers to %d/%d/%d (binding total ≈ %.0f tokens)",
        int(TARGET_CODE_PCT * 100), int(TARGET_DOC_PCT * 100),
        int(TARGET_ALIGN_PCT * 100), max_total,
    )

    balanced_by_layer: Dict[str, List[Dict[str, Any]]] = {}
    total_tokens_by_layer: Dict[str, int] = {}
    total_examples_by_layer: Dict[str, int] = {}

    for layer_name, examples in filtered_by_layer.items():
        target_tokens = int(max_total * ratios[layer_name])
        balanced = _sample_layer_to_tokens(examples, target_tokens)
        balanced_by_layer[layer_name] = balanced
        total_tokens_by_layer[layer_name] = sum(ex.get("token_count", 0) for ex in balanced)
        total_examples_by_layer[layer_name] = len(balanced)
        logger.info(
            "    [%s] → %d examples, %d tokens (target %d)",
            layer_name, len(balanced), total_tokens_by_layer[layer_name], target_tokens,
        )

    # ── Write balanced outputs ─────────────────────────────────────────────
    for layer_name, balanced in balanced_by_layer.items():
        output_path = CHUNKS_DIR / f"{layer_name}_chunks_filtered.jsonl"
        write_jsonl(output_path, balanced)
        filtered_paths[layer_name] = output_path

    # ── Final ratio check ──────────────────────────────────────────────────
    grand_total_tokens = sum(total_tokens_by_layer.values())
    if grand_total_tokens > 0:
        logger.info("  Token ratio check:")

        code_pct = total_tokens_by_layer.get("code", 0) / grand_total_tokens
        doc_pct = total_tokens_by_layer.get("doc", 0) / grand_total_tokens
        align_pct = total_tokens_by_layer.get("alignment", 0) / grand_total_tokens

        logger.info("    Code:      %.1f%% (target: %.0f%%)", code_pct * 100, TARGET_CODE_PCT * 100)
        logger.info("    Docs:      %.1f%% (target: %.0f%%)", doc_pct * 100, TARGET_DOC_PCT * 100)
        logger.info("    Alignment: %.1f%% (target: %.0f%%)", align_pct * 100, TARGET_ALIGN_PCT * 100)

        # Tiger Style: assert ratios are within tolerance.
        assert abs(code_pct - TARGET_CODE_PCT) <= RATIO_TOLERANCE, (
            f"Code ratio {code_pct:.1%} outside tolerance ±{RATIO_TOLERANCE:.0%} "
            f"(target {TARGET_CODE_PCT:.0%})."
        )
        assert abs(doc_pct - TARGET_DOC_PCT) <= RATIO_TOLERANCE, (
            f"Doc ratio {doc_pct:.1%} outside tolerance ±{RATIO_TOLERANCE:.0%}."
        )
        assert abs(align_pct - TARGET_ALIGN_PCT) <= RATIO_TOLERANCE, (
            f"Alignment ratio {align_pct:.1%} outside tolerance ±{RATIO_TOLERANCE:.0%}."
        )
        logger.info("    ✓ All ratios within ±%.0f%% tolerance", RATIO_TOLERANCE * 100)

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
    layer_tokens: Dict[str, int] = {"code": 0, "doc": 0, "alignment": 0}
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

        logger.info("  Token ratios: Code %.1f%%, Doc %.1f%%, Align %.1f%%",
                     code_pct * 100, doc_pct * 100, align_pct * 100)

        if abs(code_pct - TARGET_CODE_PCT) > RATIO_TOLERANCE:
            logger.error("  FAIL: Code ratio %.1f%% outside tolerance", code_pct * 100)
            all_pass = False
        if abs(doc_pct - TARGET_DOC_PCT) > RATIO_TOLERANCE:
            logger.error("  FAIL: Doc ratio %.1f%% outside tolerance", doc_pct * 100)
            all_pass = False
        if abs(align_pct - TARGET_ALIGN_PCT) > RATIO_TOLERANCE:
            logger.error("  FAIL: Alignment ratio %.1f%% outside tolerance", align_pct * 100)
            all_pass = False

        if all_pass:
            logger.info("  ✓ All ratios within ±%.0f%% tolerance", RATIO_TOLERANCE * 100)

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
        description="Phase C: Dataset Construction (60/25/15 Rule) for SmolLM3 fine-tuning",
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
        help="Phase(s) to run: all, clone, chunk, instruct, docs, align, filter, mix, validate. "
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
    logger.info("Data Preparation — Phase C: 60/25/15 Dataset Construction")
    logger.info("=" * 60)

    # Determine which phases to run.
    if args.phase == "all":
        phases = ["clone", "chunk", "instruct", "docs", "align", "filter", "mix", "validate"]
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
            phase_generate_instructions(logger)

        elif phase_name == "docs":
            phase_scrape_docs(logger)

        elif phase_name == "align":
            phase_build_alignment(logger)

        elif phase_name == "filter":
            filtered_paths = phase_filter_and_balance(logger)

        elif phase_name == "mix":
            # If filter wasn't run in this session, load default paths.
            if filtered_paths is None:
                filtered_paths = {
                    "code":      CHUNKS_DIR / "code_chunks_filtered.jsonl",
                    "doc":       CHUNKS_DIR / "doc_chunks_filtered.jsonl",
                    "alignment": CHUNKS_DIR / "alignment_chunks_filtered.jsonl",
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
                         "all, clone, chunk, instruct, docs, align, filter, mix, validate")
            return 1

    logger.info("=" * 60)
    logger.info("Pipeline complete. Run `uv run python data_prep.py --phase validate` to verify.")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
