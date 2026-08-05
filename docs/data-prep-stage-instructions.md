# Data Preparation Stage — Phase C: Dataset Construction (The 50/25/15/10 Rule)

## Purpose

Construct a single `train.jsonl` file at `data/train.jsonl` where every line is:
```json
{"instruction": "...", "input": "(optional)", "output": "..."}
```

The final mixture must satisfy:

| Layer | Ratio | Source | Purpose |
|---|---|---|---|
| Code | 50% | SQLite, Redis, Starlette, Rich, HTTPX, Alpine.js, htmx, Pico.css | Ground model in concrete, executable syntax |
| Docs | 25% | cppreference.com, docs.python.org, MDN | Teach exact API signatures and usage facts |
| Alignment | 15% | Tiger Style guide, low-level design principles, whitepapers | Instill disciplined, safety-first coding habits |
| Devops/Logs | 10% | Config formats, shell/CI snippets, structured logs, CLI usage | Teach operational tooling, YAML/TOML, log formats, deployment patterns |

**Target size:** 10,000–25,000 total examples (sufficient for LoRA fine-tuning a 3B model over 2–3 epochs without overfitting).

---

## How to Run

The full pipeline is driven by two files in the project root:

| File | Purpose |
|------|---------|
| `data_prep.py` | Python pipeline orchestrator (all phases) |
| `run_data_prep.sh` | Bash runner — pre-flight checks, venv, model download, invokes `data_prep.py` |

**Recommended (bash runner):**
```bash
# Full pipeline (clone → chunk → instruct → docs → align → filter → mix → validate):
./run_data_prep.sh

# Single phase (resumes from checkpoint):
./run_data_prep.sh --phase instruct

# Multiple comma-separated phases:
./run_data_prep.sh --phase chunk,instruct

# Smoke test (quick LM Studio backend validation):
./run_data_prep.sh --smoke-test

# Skip model auto-download (use an existing model file):
./run_data_prep.sh --skip-model-download

# Verbose debug logging:
./run_data_prep.sh --verbose

# Benchmark the orchestrator on the first 100 real chunks:
./run_data_prep.sh --phase instruct --instruct-limit 100

# Force a specific worker mix (GPU + N CPU workers / GPU batch size):
./run_data_prep.sh --phase instruct --cpu-workers 2 --gpu-parallel 8
```

**Direct Python (without bash wrapper):**
```bash
# Set up environment first:
uv venv
uv pip install -r requirements.txt

# Download model:
huggingface-cli download Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF \
    qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --local-dir models/

# Run individual phases:
uv run python data_prep.py --phase all
uv run python data_prep.py --phase instruct   # batched if LM Studio detected
uv run python data_prep.py --phase align      # batched if LM Studio detected
```

**Checkpoints:** Each phase writes to a checkpoint file in `data/chunks/`. If a
phase is re-run, it skips already-processed records and resumes from where it left
off. For a fresh run, delete the corresponding checkpoint file.

**Quick validation (LM Studio backend):**
```bash
uv run python tests/smoke_batched_llm.py
```

---

## Local Model for Synthetic Instruction Generation

All instruction generation is done **locally** using a small GGUF model. The
instruct phase runs through the **resource-aware orchestrator** (`orchestrator.py`),
which probes the machine and auto-splits the job across GPU and CPU workers (see
[Orchestrator](#resource-aware-orchestrator) below). If LM Studio is installed its
CUDA backend is auto-detected (`~/.lmstudio/extensions/backends/llama.cpp-*-cuda-*`)
and used for **dynamic batched inference** on the GPU worker.

**Recommended model:** `Qwen2.5-Coder-1.5B-Instruct` (GGUF Q4_K_M)
- **Why:** Specifically trained for code understanding and instruction following. 1.5B params runs comfortably on CPU (~1 GB RAM, ~15 tok/s). Generates high-quality natural-language instructions from raw code.
- **Fallback:** `SmolLM2-1.7B-Instruct` (same model family as our target).

---

## Directory Layout

```
input_data/
├── repos/                    # Cloned git repositories
│   ├── sqlite/               #   C — SQLite amalgamation + ext
│   ├── redis/                #   C — Redis server core
│   ├── starlette/            #   Python — ASGI framework
│   ├── rich/                 #   Python — terminal UI lib
│   ├── httpx/                #   Python — HTTP client
│   ├── alpine/               #   JS — reactive framework
│   ├── htmx/                 #   HTML — hypermedia framework
│   └── picocss/              #   CSS — minimal stylesheet
├── docs/                     # Scraped documentation
│   ├── cppreference/         #   C stdlib docs
│   ├── python/               #   Python 3 stdlib docs
│   └── mdn/                  #   JS/HTML/CSS reference
└── alignment/                # Tiger Style + design resources
    ├── tiger-style/          #   Tiger Style guide content
    ├── low-level-design/     #   Design principle documents
    └── whitepapers/          #   Jordi Villar whitepaper summaries

data/
├── chunks/                   # Intermediate files (one per layer)
│   ├── code_chunks.jsonl
│   ├── doc_chunks.jsonl
│   └── alignment_chunks.jsonl
├── train.jsonl               # Final shuffled, deduplicated dataset
└── stats.json                # Token counts, language breakdown
```

---

## Detailed Steps

### Step 0: Environment Setup

**Action:**
```bash
pip install tree-sitter tree-sitter-c tree-sitter-python tree-sitter-javascript \
            tree-sitter-css tree-sitter-html \
            llama-cpp-python requests beautifulsoup4 gitpython orjson \
            huggingface-hub tqdm
```

**Why:** `tree-sitter-{lang}` packages bundle pre-compiled grammars so we don't need to compile `.so` files manually. `orjson` is faster than stdlib `json` for bulk JSONL writing. `llama-cpp-python` gives us a Python API over GGUF models.

**Download generator model:**
```bash
huggingface-cli download Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF \
    qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
    --local-dir models/
```

---

### Step 1: Clone Target Repositories (50% Code Layer)

**What:** Clone the 8 approved repositories into `input_data/repos/`.

| Repo | Language | Why chosen |
|---|---|---|
| `https://github.com/sqlite/sqlite.git` | C | Pure C, battle-tested, teaches memory discipline |
| `https://github.com/redis/redis.git` | C | Production C, networking, data structures |
| `https://github.com/encode/starlette.git` | Python | Modern async Python, clean code |
| `https://github.com/Textualize/rich.git` | Python | Expressive Python, lots of stdlib usage |
| `https://github.com/encode/httpx.git` | Python | Async HTTP, real-world API design |
| `https://github.com/alpinejs/alpine.git` | JS | Reactive JS without framework bloat |
| `https://github.com/bigskysoftware/htmx.git` | HTML | Hypermedia-driven HTML attributes, semantic patterns |
| `https://github.com/picocss/pico.git` | CSS | Minimal, semantic CSS |

**Action:**
```python
for repo in REPOS:
    target = f"input_data/repos/{repo['name']}"
    if not os.path.exists(target):
        git.Repo.clone_from(repo["url"], target, depth=1)
    else:
        # Pull latest
        git.Repo(target).remotes.origin.pull()
```

**Why `depth=1`:** We only need the current state of the code, not history. Saves disk space and clone time.

**Why clone fresh vs. Hugging Face datasets:** The repos are the exact codebases approved in the research. Pre-built HF datasets may contain different versions, have license issues, or include test files we don't want. Cloning gives us full control.

**Filtering rules** (applied in Step 2):
- Exclude `test/`, `tests/`, `testing/` directories
- Exclude `build/`, `vendor/`, `third_party/` directories
- Exclude files > 5000 lines (likely generated or amalgamated)
- C: only `.c` and `.h` files
- Python: only `.py` files (exclude `__pycache__`, `.pyc`)
- JS: only `.js` files (exclude bundled/minified `.min.js`)
- HTML: only `.html` and `.htm` files (exclude generated/template partials)
- CSS: only `.css` files

---

### Step 2: Chunk Code with tree-sitter (50% Code Layer)

**What:** Parse every source file with tree-sitter and extract individual function and class definitions as discrete chunks.

**Why tree-sitter over regex:**
- Regex cannot correctly identify function boundaries in C (nested functions, preprocessor noise) or Python (decorators, nested classes)
- tree-sitter produces a concrete syntax tree with exact start/end byte ranges
- Supports all 5 target languages (C, Python, JS, HTML, CSS) with one uniform API, JS, HTML, CSS, TS)

**Action for each file:**

```
For each approved source file:
  1. Read file content into string
  2. Create tree-sitter Parser for the file's language
  3. Parse → get CST root
  4. Query for:
     - C:     function_definition, declaration
     - Python: function_definition, class_definition, decorated_definition
     - JS:    function_declaration, class_declaration, arrow_function, method_definition
- CSS:   rule_set, @media_statement
   - HTML:  element (selecting top-level semantic elements: <main>, <section>, <article>, <form>, <template>, <custom-element>)
  5. For each matched node:
     a. Extract line range, byte range
     b. Skip if < 3 lines or > 200 lines (too small = trivial, too large = unfocused)
     c. Build chunk record:
        {
          "repo": "sqlite",
          "language": "c",
          "file_path": "src/main.c",
          "chunk_type": "function",
          "name": "sqlite3_exec",
          "start_line": 1200,
          "end_line": 1450,
          "code": "<exact source text>",
          "docstring": "<any preceding comment/docstring>",
          "signature": "<first line of function>"
        }
  6. Append to data/chunks/code_chunks_raw.jsonl
```

**Why these chunk size limits:**
- < 3 lines is usually a trivial getter/setter that adds no training signal
- > 200 lines is usually a monolithic function that's hard for a 3B model to learn from
- Both extremes dilute dataset quality

**Expected output volume:**

| Language | Repos | Est. functions/classes |
|---|---|---|
| C | SQLite + Redis | 1,500–3,000 |
| Python | Starlette + Rich + HTTPX | 600–1,200 |
| JS | Alpine | 150–300 |
| HTML | htmx | 100–250 |
| CSS | Pico | 50–100 |
| **Total** | **8 repos** | **2,400–5,150 chunks** |2,300–4,600 chunks** |

---

### Step 3: Generate Synthetic Instructions for Code Chunks (50% Code Layer)

**What:** For every code chunk, use the local GGUF model to generate a natural-language instruction that a human would give to elicit exactly that code as output.

**Why:** Raw code without instructions is useless for instruct fine-tuning. The model needs to learn "If a user asks X, produce Y." We use a local model so no data leaves our machine and no API costs.

**Prompt template:**
```
You are an expert programmer. Given the following {language} code, write a clear,
concise instruction that a human would write to ask for exactly this code.
The instruction should be 1-3 sentences, be specific, and mention the language if relevant.
Output ONLY the instruction, nothing else.

CODE:
```{language}
{code_chunk}
```

INSTRUCTION:
```

**Action:**
```python
llm = Llama(model_path="models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
            n_ctx=2048, n_threads=4)

chunks = read_jsonl("data/chunks/code_chunks_raw.jsonl")
results = []
for chunk in tqdm(chunks):
    prompt = build_prompt(chunk["language"], chunk["code"])
    response = llm(prompt, max_tokens=128, temperature=0.7, stop=["\n\n"])
    instruction = response["choices"][0]["text"].strip()
    results.append({
        "instruction": instruction,
        "input": "",
        "output": chunk["code"],
        "metadata": {
            "layer": "code",
            "language": chunk["language"],
            "repo": chunk["repo"],
            "chunk_name": chunk["name"]
        }
    })

write_jsonl("data/chunks/code_chunks_with_instructions.jsonl", results)
```

**Quality filters:**
- Drop if instruction < 5 chars (model failed)
- Drop if instruction is an exact copy of the code (hallucination)
- Drop if instruction contains "I cannot" or "I'm unable" (refusal)
- Deduplicate by exact output match (same function may appear in multiple files)

**Why temperature 0.7:** High enough for diversity, low enough to avoid gibberish.

**Why n_ctx 2048:** Code chunks are max 200 lines; 2048 context is more than enough and keeps inference fast.

---

### Step 4: Scrape Documentation (25% Documentation Layer)

**What:** Scrape official docs and convert each API entry into a direct Q&A pair.

**Why:** A 3B model has limited capacity. It cannot reliably infer API signatures from code alone. Explicit Q&A pairs teach exact function signatures, parameter meanings, and usage patterns.

#### 4a. Scrape cppreference.com (C Standard Library)

**Source:** https://en.cppreference.com/w/c

**Action:**
```
For each page under /w/c/ (string, stdio, stdlib, math, time, etc.):
  1. Fetch HTML with requests
  2. Parse with BeautifulSoup
  3. Extract:
     - Function name (from <h1> or page title)
     - Signature (from <code> in synopsis section)
     - Description (first paragraph after synopsis)
     - Example code (from <div class="example">)
  4. Generate Q&A:
     instruction: "What is the signature and purpose of C's {function}?"
     output: "{signature}\n\n{description}\n\nExample:\n{example}"
  5. Also create reverse Q&A:
     instruction: "Write a C function that {description in 5 words}"
     output: "{example code}"
```

**Expected volume:** ~400–600 C stdlib functions.

#### 4b. Scrape docs.python.org (Python 3 Standard Library)

**Source:** https://docs.python.org/3/library/

**Action:**
```
For each stdlib module page (os, sys, json, asyncio, pathlib, collections, etc.):
  1. Fetch HTML
  2. Parse with BeautifulSoup — Python docs have consistent structure
  3. For each function/class in the module:
     - Extract: signature (from <code> or <dt>), description, example
  4. Generate Q&A (same pattern as 4a)
```

**Expected volume:** ~800–1,200 Python stdlib entries.

#### 4c. Scrape MDN (JS, HTML, CSS)

**Source:** https://developer.mozilla.org/en-US/docs/Web

**Action:**
```
For each reference section (JavaScript, HTML, CSS):
  JS:  Array, String, Promise, fetch, etc.
  HTML: <div>, <form>, semantic elements, attributes
  CSS:  flexbox, grid, display, positioning, etc.

  For each API entry:
    1. Fetch HTML
    2. Extract syntax, description, example
    3. Generate Q&A
```

**Expected volume:** ~1,000–2,000 MDN entries across all three.

#### Why scrape fresh?

Offline/prebuilt datasets for these docs exist (e.g., `wikitext`, `the-stack`), but:
- They may be stale (Python 3.13+ features, new CSS specs)
- They mix prose with API references
- They lack structured `{instruction, output}` format

Scraping gives us exact, structured, up-to-date Q&A pairs.

**Caching:** Store raw HTML in `input_data/docs/{source}/raw/` so re-scraping isn't needed every run. Only re-scrape if `--refresh` flag is passed.

**Delay:** Add 0.5s delay between requests to respect robots.txt and avoid rate limiting.

---

### Step 5: Build Alignment Examples (15% Alignment Layer)

**What:** Create "refactoring" and "code review" examples that teach Tiger Style principles, low-level design thinking, and architecture reasoning.

**Why without examples from these principles, the model writes code that "works" but is unsafe — hidden allocations, no bounds checking, implicit assumptions. The alignment layer explicitly trains the model to think defensively.**

#### 5a. Ingest Tiger Style Guide

**Source:** https://github.com/tigerbeetle/tigerbeetle/blob/main/docs/TIGER_STYLE.md

**Key principles to encode (paraphrased):**
1. No hidden memory allocations — all buffers passed explicitly by caller
2. No implicit control flow — no exceptions, no hidden panics
3. Explicit bounds checking on every array access
4. No undefined behavior — even in "unreachable" paths
5. Deterministic destruction — RAII-style cleanup
6. Minimal dependencies — prefer stdlib over external

**Action:**
```
For each principle:
  1. Write a "BAD" code snippet that violates the principle (in C, Python, or HTML/JS)
  2. Write a "GOOD" code snippet that complies
  3. Use local LLM to generate a thought trace:
     prompt: "Explain step-by-step why this code violates Tiger Style
              principle '{principle}' and how to fix it."
  4. Build training example/JS)
  2. Write a "GOOD" code snippet that complies
  3. Use local the principle (in C or Python)
  2. Write a "GOOD" code snippet that complies
  3. Use local LLM to generate a thought trace:
     prompt: "Explain step-by-step why this code violates Tiger Style
              principle '{principle}' and how to fix it."
  4. Build training example:
     instruction: "Refactor this {lang} code to comply with Tiger Style:
                   {principle_name}"
     input: "{bad_code}"
     output: "<thought>{thought_trace}</thought>\n\n{good_code}"
```

#### 5b. Low-Level Design Principles

**Principles to encode:**
- Prefer composition over inheritance
- Single responsibility per function
- Fail fast and loudly
- No silent data corruption
- Prefer immutable data where possible
- Cache invalidation and consistency patterns

**Action:** Same as 5a — bad code → thought trace → refactored code pairs.

#### 5c. Whitepaper Reasoning Traces

**Source:** Jordi Villar's paper list (https://jordivillar.com/notes/papers)

**Action:**
```
For each key paper (MapReduce, Bigtable, Chubby, Dynamo, CAP theorem, etc.):
  1. Write a short summary of the paper's key insight
  2. Create a prompt that asks the model to apply the principle:
     instruction: "Using principles from {paper_name}, design a {component}
                   that handles {constraint}."
     output: "<thought>Applying the paper's principle of {X}...</thought>\n\n{design}"
```

**Why this is valuable:** Teaches the model to apply abstract system design concepts to concrete coding tasks — bridging the gap between whitepaper theory and implementation.

#### 5d. Expand with the Local LLM

The seed examples (~50–100 hand-crafted) are too few. Use the local model to generate variants:

```
For each seed example:
  1. Vary: language (C → Python → JS), variable names, code structure
  2. Vary: principle combinations (e.g., "no hidden allocs + bounds checking")
  3. Filter: drop any where the generated "fixed" code is lower quality than the original
```

**Expected volume:** ~1,500–3,000 alignment examples after expansion.

---

### Step 6: Token Accounting & Quality Filtering

**Why:** The 50/25/15/10 ratio must be measured in *tokens seen during training*, not just example counts. A 10-line C function and a 3-page design document have very different token counts.

**Action:**
```python
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")

for layer in ["code", "doc", "alignment", "devops"]:
    examples = read_jsonl(f"data/chunks/{layer}_chunks.jsonl")
    for ex in examples:
        total_tokens = len(tokenizer.encode(ex["instruction"] + ex["output"]))
        ex["token_count"] = total_tokens

    total_layer_tokens = sum(ex["token_count"] for ex in examples)
    print(f"{layer}: {len(examples)} examples, {total_layer_tokens} tokens")
```

**Filtering rules (applied per layer):**
- Drop if `token_count < 8` (too short to be useful)
- Drop if `token_count > 2048` (exceeds model's effective context for learning)
- Drop if output is identical to another output (exact dedup)
- Drop if instruction contains non-ASCII garbage characters

**Balance adjustment:**
After token counting, if a layer is over/under-represented in token-space:
- Under: pad with additional examples (re-generate from unused chunks)
- Over: subsample the longest examples first (they contribute disproportionately)

**Target token ratios:** 50% (±3%) code, 25% (±3%) docs, 15% (±3%) alignment, 10% (±3%) devops.

---

### Step 7: Mix, Shuffle & Write Final Dataset

**Action:**
```python
all_examples = []
for layer in ["code", "doc", "alignment", "devops"]:
    examples = read_jsonl(f"data/chunks/{layer}_chunks_filtered.jsonl")
    all_examples.extend(examples)

random.shuffle(all_examples)

write_jsonl("data/train.jsonl", all_examples)
```

**Also write stats:**
```json
{
  "total_examples": 15000,
  "total_tokens": 12500000,
  "code_layer": {"examples": 7500, "tokens": 6250000, "pct": 50.0},
  "doc_layer": {"examples": 3750, "tokens": 3125000, "pct": 25.0},
  "alignment_layer": {"examples": 2250, "tokens": 1875000, "pct": 15.0},
  "devops_layer": {"examples": 1500, "tokens": 1250000, "pct": 10.0},
  "languages": {"c": {"examples": 4000, "tokens": 5000000}, ...}
}
```

---

### Step 8: Validate Final Dataset

**Checklist:**
1. `data/train.jsonl` is valid JSONL (every line parses as valid JSON)
2. Every line has `instruction`, `input`, `output` keys
3. Total examples between 10,000 and 25,000
4. Token ratio is within ±3% of 50/25/15/10
5. Language breakdown includes C, Python, JS, CSS, HTML
6. No duplicate outputs across the entire file
7. Sample 50 random examples and manually verify quality:
   - Instructions are coherent and match their outputs
   - Code examples are syntactically valid
   - Alignment examples contain `<thought>` traces
   - Doc examples contain accurate API information

**Validation script:**
```bash
python -c "
import json, sys
with open('data/train.jsonl') as f:
    lines = f.readlines()
print(f'Total examples: {len(lines)}')
# Check structure
for i, line in enumerate(lines[:5]):
    obj = json.loads(line)
    assert 'instruction' in obj, f'Missing instruction in line {i}'
    assert 'output' in obj, f'Missing output in line {i}'
print('Structure valid')
"
```

---

## Summary: End-to-End Pipeline

```
Step 0: pip install + download GGUF model
    │
Step 1: git clone 7 repos → input_data/repos/
    │
Step 2: tree-sitter parse → extract functions/classes → code_chunks_raw.jsonl
    │
Step 3: Local LLM (Qwen2.5-Coder-1.5B) generates instructions → code_chunks_ready.jsonl
    │
Step 4: Scrape cppreference + python docs + MDN → doc_chunks.jsonl
    │
Step 5: Build seed alignment examples → LLM expands → alignment_chunks.jsonl
    │
Step 6: Token counting → filter → balance → deduplicate
    │
Step 7: Shuffle + write data/train.jsonl + data/stats.json
    │
Step 8: Validate structure, ratios, quality
```

## Resource-Aware Orchestrator

The instruct phase (and every future LLM-heavy stage) runs through
`orchestrator.py` — a barebones, reusable execution runtime. It owns three things:

1. **Hardware probing** (`probe()`) — live CPU cores, RAM, disk, GPU (via
   `nvidia-smi`), and the LM Studio llama.cpp backend path.
2. **Resource planning** (`plan_llm_resources()`) — turns the probe into a worker
   set: one batched GPU worker (LM Studio `libllama.so`, `n_parallel` sized to fit
   free VRAM) plus as many CPU workers (llama-cpp-python, pure-CPU) as free cores
   and RAM allow.
3. **Distributed execution** (`run_llm_completions()`) — spawn-isolates each worker
   (the LM Studio libllama and llama-cpp-python can never share one process), feeds
   the job queue, collects results in input order, and logs per-run system stats.

**Why spawn-isolated workers:** the GPU path loads LM Studio's bundled `libllama.so`
(ctypes, `RTLD_GLOBAL`) and the CPU path loads llama-cpp-python, which bundles its
*own* llama.cpp. They export the same symbols, so they cannot coexist in one process.
Separate processes let both memory buses run at once.

**CPU workers are zero-VRAM:** llama-cpp-python is usually built with CUDA, so left
alone it allocates VRAM even at `n_gpu_layers=0`. Each CPU worker sets
`CUDA_VISIBLE_DEVICES=""` in its own process, making it pure-CPU so it never OOMs the
GPU worker.

**Per-run logs:** every run writes to `data/logs/<run_id>/`:
- `run.log` — worker plan, progress, `[sys]` samples every 10 s
- `system_stats.csv` — 1 s CPU%/RAM/GPU-util/VRAM/power/temp samples
- `worker_<name>.log` — per-worker logs
- `plan.json` — worker specs, per-worker throughput, final stats

**Auto-calibration:** with no flags, `_orchestrator_plan` runs `calibrate_plan()`,
which races gpu-only / +1cpu / +2cpu on a 60-job sample of the *real* workload and
keeps whichever wins on the machine. On this laptop (RTX 3050 4 GB, 1.5B Q4 model)
the measured result was:

| Mix | Throughput |
|---|---|
| gpu-only | 1.25 jobs/s |
| gpu + 1 cpu worker | 1.08 jobs/s |
| gpu + 2 cpu workers | 0.96 jobs/s |

GPU-only wins here: the batched GPU worker is VRAM-bandwidth bound and the CPU
workers steal CPU time it needs for scheduling while adding far less than they take.
CPU workers are still supported and win for other job types (embeddings, larger
models, no-GPU machines) — override the choice with `--cpu-workers N`.

**CLI:**
```bash
uv run python orchestrator.py                          # print probe + plan
uv run python data_prep.py --phase instruct            # calibrated plan
uv run python data_prep.py --phase instruct --cpu-workers 2 --cpu-threads 2 --gpu-parallel 8
uv run python data_prep.py --phase instruct --instruct-limit 100   # benchmark on real chunks
```

## Batched LLM Inference via LM Studio

The default serial loop processes one code chunk per `llama_decode` call, keeping
the GPU memory-bandwidth-bound and idle most of the time. On an RTX 3050 Laptop
(4 GB, 192 GB/s) the 58k chunk instruction generation ran for >30 hours in serial
mode.

When LM Studio is installed its bundled `libllama.so` (llama.cpp b8733) is loaded
directly — no compilation, no server process, no network round-trip. The low-level
batch API (`llama_batch_init`, `llama_decode` with multiple sequences) decodes all
active sequences in one forward pass, sharing the weight read across all N sequences:

- **Prefill phase:** all prompts are tokenised and decoded in one call.
- **Decode loop:** one `llama_decode` per step produces N tokens (one per sequence).
- **Pipeline:** Each step reads the 934 MiB model weights once and reuses them for
  all N sequences, giving ~N× the single-sequence decode throughput.

On a 4 GB RTX 3050 the pipeline runs at `n_parallel=12` with 2048 tokens per
sequence, consuming ~2.8 GB VRAM. Measured throughput for 128-token completions:

| Mode              | 12 completions | Per-request rate | Wall-clock (58k chunks) |
|-------------------|----------------|------------------|-------------------------|
| Serial (old)      | ~24 s          | 0.5 req/s        | >30 h                   |
| Batched (LM Studio)| ~12 s          | 1.0 req/s         | ~15 h (estimated)       |

**Auto-detection:** `data_prep.py` calls `find_lmstudio_backend()` at runtime.
If a compatible CUDA backend is found (`~/.lmstudio/extensions/backends/llama.cpp-*-cuda-*`)
the batched path is used automatically. Otherwise it falls back to the serial
`llama-cpp-python` path — no configuration needed.

**Warm-up workaround:** This particular LM Studio llama.cpp build (b8733,
commit d6f3030) requires a warm-up sequence of one CPU-only model load followed
by a deliberately-failing GPU no-mmap load before the real GPU mmap load succeeds.
The `BatchedLlama` class handles this transparently at ~5 s startup cost.
Future LM Studio backend versions may not need this workaround.

**When to skip LM Studio:**
- The model file doesn't exist → no LLM expansion at all, seeds only.
- `find_lmstudio_backend()` returns `None` → serial `llama-cpp-python` path.
- Any exception during `BatchedLlama.start()` → logged as warning, serial fallback.

## File Layout

The pipeline lives in the project root:

```
├── data_prep.py              # CLI entry point: orchestrates all phases (Steps 1–8)
├── llm_backend.py            # LM Studio batched LLM backend (ctypes bindings to libllama.so)
├── orchestrator.py           # Resource-aware orchestrator: probe → plan → distributed run
├── run_data_prep.sh          # Bash runner — pre-flight checks, venv, model, invokes data_prep.py
├── tests/
│   └── smoke_batched_llm.py  # Quick smoke test for the LM Studio backend
├── data/
│   ├── logs/                 # Per-run orchestrator logs (run.log, system_stats.csv, plan.json)
│   └── chunks/               # Checkpoint files (code_chunks_raw, _ready, doc_, alignment_, devops_)
├── models/                   # GGUF model files (qwen2.5-coder-1.5b-instruct-q4_k_m.gguf)
├── input_data/               # Cloned repos, scraped docs, alignment seeds
└── docs/
    └── data-prep-stage-instructions.md  # This document
```

Key design properties:
- **Single Python file:** `data_prep.py` contains all phases — no module hierarchy to
  navigate. Each phase is a self-contained function with an `assert`-enforced contract.
- **Batched inference lives in `llm_backend.py`:** a self-contained ctypes wrapper
  around LM Studio's `libllama.so`. The two modules share no state except through the
  global `_batched_llm_instance` / `_llm_instance` pointers (exactly one of which is
  set at any time).
- **Resumable:** Checkpoint files in `data/chunks/` let each phase skip already-done
  work. Delete the checkpoint to force a re-run.
- **Independently runnable phases:** The `--phase` CLI flag accepts a single phase or
  comma-separated list (e.g., `--phase chunk,instruct`).