## Constraints, Reccomendation and Questionnaire

### Hardware Constraints

**The Constraint:** ~4 CPU Cores, 4GB RAM.
**The Problem:** Large Language Models (LLMs) require memory for two things: **Weights** (the model itself) and the **KV Cache** (the memory of the current conversation/context).

Let's do the math:
*   **Total RAM:** 4096 MB.
*   **OS & Python Runtime overhead:** ~512 MB.
*   **Available Inference RAM:** ~3584 MB.

If we run a model in standard 16-bit precision (FP16):
*   A 1-Billion parameter (1B) model takes 2000 MB just for weights. This leaves only ~1500 MB for the KV cache, meaning you can only process a few hundred tokens before crashing.
*   A 3B model takes 6000 MB. **It will not run.**

**The Solution (Derived):**
1.  **Quantization is Mandatory:** You must use 4-bit quantization (specifically `GGUF` format via `llama.cpp`). At 4-bit, a 1.5B model takes ~900 MB. A 3B model takes ~1900 MB. This leaves ample room for the KV cache.
2.  **Model Size Limit:** You are strictly limited to the **1.5B to 3B parameter range**.
3.  **Orchestrator Must Be Deterministic:** You cannot afford to run an LLM as the top-level orchestrator. Running a 3rd model just to "route" queries will thrash your 4GB RAM and destroy CPU performance. The orchestrator *must* be a lightweight, rule-based Python state machine (e.g., using `LangGraph` or simple Finite State Machines). The LLMs are just "tools" or "nodes" that the Python script calls.

---

### Part 2: Model Recommendations

Based on the math above, here are the exact models that fit your constraints while maximizing reasoning capabilities.

#### 1. The Research / Encyclopedia Model
*   **Recommendation:** **Qwen2.5-3B-Instruct** (or Qwen2.5-1.5B-Instruct if you need to run both models in RAM simultaneously).
*   **Why:** Qwen2.5 currently dominates the sub-7B parameter space in reasoning, math, and factual recall. 
*   **Regarding your Wikipedia idea:** Fine-tuning on `wikimedia/structured-wikipedia` is interesting, but from first principles, *memorizing* a dynamic encyclopedia in model weights leads to severe hallucinations and outdated facts.
    *   *Better Approach:* Use **Local RAG (Retrieval-Augmented Generation)**. Embed a Wikipedia dump locally using a tiny embedding model (like `BAAI/bge-small-en-v1.5`, which is ~130MB). When the user asks a question, do a vector search, inject the exact Wikipedia paragraphs into the Qwen2.5 prompt, and ask it to synthesize the answer. This guarantees zero hallucination on factual data.

#### 2. The Coding Model (with TDD & Sandbox)
*   **Recommendation:** **Qwen2.5-Coder-1.5B-Instruct** (or 3B-Instruct).
*   **Why:** Qwen2.5-Coder is specifically trained on GitHub repositories and understands multiple languages (C, Python, JS/HTML/CSS). The 1.5B version is exceptionally good at writing boilerplate and standard logic, which is perfect for your TDD loop.
*   **Sandboxing:** Use `Docker` (if resources allow) or `nsjail`/`firejail` to execute the generated Python/C code safely to verify test cases.

#### 3. The Top-Level Orchestrator
*   **Recommendation:** **Do not use an LLM.** Use a Python-based **Router/State Machine**.
*   **Why:** An orchestrator needs to "filter, format, apply policy, and decide parameters." These are deterministic logic gates. If Query contains "code" -> route to Coding Model. If Query contains "history" -> route to Research Model. Doing this via an LLM wastes precious CPU cycles.

---

### Part 3: What Looks Vague? (Questions for You)

Your architecture has several conceptual gaps that need to be defined before we can finalize the prompt engineering and harness loops. Please clarify the following:

**1. The "Validation & Auto-Fix" Loop (The Hallucination Problem)**
*   *Vagueness:* You state: *"If they come out to be wrong (hallucinates), then fix the prompt and run again automatically."*
*   *First Principles Question:* How does the harness *objectively know* the model hallucinated?
    *   For the **Coding Model**, this is easy: The code is run in the sandbox, the tests fail, the error log is fed back to the model to fix it.
    *   For the **Research Model**, how do you validate facts? Do you have access to an external Search API (like Brave Search) to cross-reference? Or are you relying on a separate "Judge" model? (Running a Judge model on 4GB RAM will severely bottleneck the system).

**2. Model Memory Management (Concurrency vs. Hot-Swapping)**
*   *Vagueness:* Do the Research Model and Coding Model run at the same time?
*   *First Principles Question:* If you load Qwen-3B-Research and Qwen-1.5B-Coder into RAM simultaneously, you will use ~2.8GB of RAM just for weights, leaving very little for context, and causing massive CPU context-switching overhead.
    *   Do you want to **Hot-Swap** (Load Research -> Generate -> Unload -> Load Coding -> Generate -> Unload)? This takes 2-4 seconds per swap on a 4-core CPU.
    *   Or do you want to use **One Unified Model** (e.g., Qwen2.5-3B) and just change its system prompt to act as a coder vs. a researcher? (Saves memory, but slightly drops coding performance).

**3. "System Design Planning" on a 1.5B Model**
*   *Vagueness:* You mention the Coding Model should be capable of *"Planning based on system design"*.
*   *First Principles Question:* System design (e.g., "Design a scalable microservices architecture for an e-commerce app") requires massive world-knowledge and deep reasoning. A 1.5B or 3B model *will* hallucinate wildly at this task.
    *   Do you mean "System Design" at the macro level (AWS, databases, load balancers), or at the micro level (Class diagrams, file structures, function signatures for the current script)? We must restrict the scope of the prompt, or the auto-fix loop will run forever.

**4. What is the "Bifurcation" mechanism?**
*   *Vagueness:* *"Keep bifurcating query top down to save compute resources based on compute resource availability."*
*   *First Principles Question:* How does the system measure "compute resource availability" in real-time? Are you monitoring CPU load averages/RAM usage, and if CPU > 80%, you force the model to output a shorter, less reasoned response? Or do you mean the Orchestrator breaks a complex question into 3 smaller sub-questions to prevent the LLM from running out of context window?

---



## STEP 1 Answers:

### 1. First Principles of Fact Validation (The Judge)

**The Physics of Validation:** A hallucination is a statement lacking external grounding. Running a 1.5B model to "judge" a 1.5B model is mathematically redundant; they share the same statistical biases and will likely agree with each other's hallucinations.

**The Solution:** Separate the *claim* from the *text*.
1.  **Extraction:** Use a lightweight NLP tool (e.g., `spaCy` or `GLiNER`, <200MB RAM) to extract Subject-Verb-Object (SVO) triples from the output.
2.  **Grounding:** Pass these atomic claims to your **SearXNG API**.
3.  **Scoring:** Use a lightweight Cross-Encoder (like `bge-reranker-base`, ~300MB RAM) to score the semantic overlap between the claim and the top 3 SearXNG snippets.

If you *must* use an LLM for validation, use **Qwen2.5-0.5B-Instruct**. It takes ~400MB RAM in 4-bit quantization and is fast enough to act as a lightweight judge.

### 2. The Physics of Hot-Swapping & KV Cache

**The Math:** A 1.5B model with a 2048 token context length has a KV Cache of exactly **448 MB**.
*   **Offloading:** Writing 448 MB to an NVMe SSD takes ~0.15s; a SATA SSD takes ~1s. Yes, you can offload the KV Cache to disk using tools like `llama.cpp` with the `--offload-kqv` flag.
*   **The Hard Limit:** You **cannot share KV Caches** between the Research Model and the Coding Model. The KV Cache is the result of multiplying input tokens by the specific weight matrices ($W_k, W_v$) of Model A. Passing Model A's KV Cache to Model B is mathematically equivalent to passing the intermediate state of a SHA-256 hash into a SHA-512 algorithm—it yields garbage.
*   **Architecture:** You only pass the *generated text* between models. You only offload the KV cache to disk if you plan to *resume the exact same model* later.

### 3. Tiger Style & Extreme Bifurcation

**The Derivation:** Tiger Style mandates explicit state, no hidden control flow, and zero panics. Extreme bifurcation means reducing a problem to a Directed Acyclic Graph (DAG) of atomic, independently testable functions.

**Execution:** Do not ask the Coding Model to "write the program." Force it to output a strict JSON DAG of tasks, then write the TDD test for Node 1, then the implementation for Node 1. The Sandbox runs the test; if it fails, the loop repeats.
*   **Model Choice:** **Qwen2.5-Coder-1.5B-Instruct**. It is heavily optimized for strict, logical code generation and fits perfectly in your RAM budget.

### 4. Resource Monitoring & The Orchestrator State Machine

**The Physics:** If CPU > 80%, inference latency spikes exponentially due to OS context switching.
*   **Monitoring:** A lightweight background thread using `psutil` to check CPU/RAM.
*   **State Machine on Disk:** The Orchestrator doesn't just "save context"; it maintains a **Persistent State Machine** (e.g., using a local SQLite database).
    *   Task 1 (`read_file`): Status `Pending`.
    *   Task 2 (`count_words`): Status `Blocked (Waiting on Task 1)`.
    *   When Task 1 finishes, the output string is written to SQLite, Task 2 becomes `Ready`, and the Coding Model is hot-swapped in.

### 5. The Hardware-Optimized Model Stack (Total RAM Budget: ~5GB)

*   **Orchestrator (Always Resident):** **Qwen2.5-0.5B-Instruct** (~400MB RAM in Q4_K_M). Stays in RAM to route traffic.
*   **Research Model (Hot-Swapped):** **Qwen2.5-1.5B-Instruct** (~1.1 GB RAM in Q4_K_M).
*   **Coding Model (Hot-Swapped):** **Qwen2.5-Coder-1.5B-Instruct** (~1.1 GB RAM in Q4_K_M).
*   **Judge/Validator:** **SearXNG + `bge-reranker-base`** (~300MB RAM).

*Total peak RAM (Orchestrator + 1 Active Model + Judge) = ~1.8GB. This leaves **3.2GB for the OS, KV Cache, and Sandbox**, which is highly stable.*

### 6. Vagueness Check & Clarifying Questions

*   **Vagueness 1 (The Orchestrator's Logic):** How does the 0.5B Orchestrator *know* how to bifurcate "Count words in a document" into `read_file` and `count_words`? A 0.5B model is statistically too weak to do complex algorithmic decomposition on the fly; it will hallucinate the breakdown.
    *   *Question:* Should the Orchestrator use deterministic parsing (regex/code) for known patterns, or do you want a slightly larger model (1.5B) specifically for the routing logic? ( Can we use the 1.5B as orchestrator?? maybe less quantized 3bit or 2bit ). If the RAM exceeds the requirements, let's first bifurcate, then dump output DAG to disk, and then signal the other models to pick it up.
*   **Vagueness 2 (Interaction Latency):** You mentioned the models will "continuously interact." If the Coding model needs to ask the Research model a question, you have to hot-swap *back*, load the Research model, process the question, and hot-swap *again*. This 3-way swap could take 6+ seconds per interaction loop.
    *   *Question:* Is a 6-second latency per interaction loop acceptable, or should we pre-compute a "Knowledge Context Vector" (RAG) that the Coding Model can access without hot-swapping? Let's pre compute a Knowledge Context Vector which will keep getting updated. If required, we can hot swap for specific cases.
*   **Vagueness 3 (Sandboxing):** You mentioned a "proper sandboxed environment." True containerization (Docker/Podman) eats ~200-300MB of RAM just for the daemon and namespaces.
    *   *Question:* Are you planning to use lightweight OS-level sandboxing (like `nsjail`, `bubblewrap`, or `firejail`), or just a restricted Python `subprocess`? use lightweight OS-level sandboxing (like `nsjail`, `bubblewrap`, or `firejail`)

---

## STEP 2 ANSWERS:

### 1. The 1.5B Unified Core Strategy (The Physics of Swapping)

**First Principle:** To minimize hot-swap latency, memory fragmentation, and allocation overhead, the Planner and the Coder must share the exact same base architecture (identical layer counts, hidden dimensions, and attention heads). 

**Derivation:** We will use the **Qwen2.5-1.5B** family. Because the Planner and Coder share the exact same tensor shapes, the underlying inference engine (like `llama.cpp` or `vLLM`) can predict memory allocation perfectly, drastically reducing swap times.

*   **The Planner (Orchestrator):** `Qwen2.5-1.5B-Instruct` quantized to **Q5_K_M** (~1.3 GB RAM). We use 5-bit quantization here because bifurcation and DAG generation require maximum logical reasoning capabilities.
*   **The Coder:** `Qwen2.5-Coder-1.5B-Instruct` quantized to **Q4_K_M** (~1.1 GB RAM). We use 4-bit quantization here because code generation relies more heavily on syntax memorization, leaving an extra 200MB of RAM strictly for the Sandbox.

**Execution Flow:**
1. Planner loads (1.3 GB) $\rightarrow$ Generates DAG + Knowledge Brief $\rightarrow$ Writes to SQLite $\rightarrow$ Unloads.
2. Coder loads (1.1 GB) $\rightarrow$ Reads DAG from SQLite $\rightarrow$ Executes Node 1.

### 2. The "Knowledge Context Vector" (First Principles of RAG)

**First Principle:** You mentioned a "Knowledge Context Vector." In neural networks, a vector is an array of floats (e.g., 768 dimensions). An LLM cannot directly read an array of floats into its attention mechanism; it only reads text (tokens). 

**Derivation:** If you use dense mathematical embeddings for RAG, you must run a separate neural network to process them, violating your compute constraints. Therefore, the "Vector" must be implemented as a **Dynamic Textual Context Window**—a highly compressed, continually updated Markdown file on disk.

**Execution Flow:**
1. During Phase 1, the Planner queries SearXNG.
2. Instead of feeding the raw HTML to the LLM, the Orchestrator uses a deterministic statistical filter (like **TextRank** or **TF-IDF** implemented in raw Python, consuming 0MB RAM) to extract the top 5 most mathematically relevant sentences from the search results.
3. These 5 sentences are appended to a `context.md` file on disk.
4. When the Coder loads, it reads `context.md` into its context window alongside the DAG. It never needs to hot-swap back to the Planner.

### 3. Sandboxing: The Physics of `bubblewrap`

**First Principle:** Containerization (Docker/Podman) requires a background daemon that manages namespaces, networking, and storage drivers. This daemon consumes ~200MB of RAM just sitting idle. OS-level sandboxing relies directly on the Linux kernel's native `clone()` syscall with namespace flags.

**Derivation:** **`bubblewrap` (bwrap)** is the optimal choice. It is a tiny C binary that sets up Linux namespaces, executes the script, and dies. It has zero RAM overhead when idle.

**Implementation for TDD:**
The Orchestrator creates a temporary directory, writes the test script, and invokes `bwrap`:
```bash
bwrap \
  --ro-bind /usr /usr \
  --ro-bind /lib /lib \
  --bind /tmp/workspace /workspace \
  --unshare-pid \
  --unshare-net \
  --unshare-ipc \
  python3 /workspace/test.py
```
*   `--unshare-net`: The script cannot make outbound API calls (preventing the Coder from bypassing the sandbox to fetch data).
*   `--unshare-pid`: The script cannot see or kill other processes.
*   `--bind`: The script can only write to the temporary workspace.

### 4. Vagueness Check & Clarifying Questions

Before finalizing the architecture, there are three physical limits we must define to prevent the system from crashing or looping infinitely.

**Vagueness 1: The DAG Schema**
A 1.5B model will hallucinate the structure of the DAG if not strictly constrained by a JSON Schema. 
*   **Question:** Should we strictly define the JSON Schema for the DAG right now? (e.g., Each node must explicitly declare `id`, `dependencies`, `input_context_path`, `expected_output_format`, and `language`).

**Vagueness 2: The TDD Loop Limit**
If the Coder writes a failing test and tries to fix the code, how many times do we let it loop before aborting? Without a hard limit, a logic error will cause an infinite loop, destroying your SSD and maxing out the CPU.
*   **Question:** What is the absolute maximum number of retry loops per node before the Orchestrator hard-stops, unloads the Coder, and flags the node as `FAILED` for human review? (Standard engineering practice is exactly 3 attempts).

**Vagueness 3: Context Window Overflow**
As the Coder executes Node 5, it must read the DAG and the outputs of Nodes 1 through 4. This accumulated history will eventually exceed the 2048 or 4096 token context window of the 1.5B model, causing an out-of-memory (OOM) crash or truncation.
*   **Question:** How do we compress the output of completed nodes? Do we just extract the final `stdout` string and discard the generated code to save tokens, or do we use a sliding window that drops Node 1 once Node 4 is finished?

---

# Step 3 Answer
Yes. Your answers remove the major ambiguity. The system should be built as:

1. **Persistent Orchestrator**: deterministic Python/C controller, not an always-resident LLM.
2. **LLM Planner / Research Model**: hot-swapped 1.5B model.
3. **Coding Model**: hot-swapped 1.5B coding model.
4. **DAG**: strict JSON schema written to disk.
5. **Runtime state**: separate SQLite state machine.
6. **Context overflow control**: sliding window plus dependency contracts.
7. **TDD retry limit**: exactly 3 attempts per node.
8. **Sandbox**: lightweight Linux namespace sandbox, preferably `bubblewrap`.

Below is the concrete architecture.

---

# 1. Important Quantization Correction

You said:

> Can we use the 1.5B as orchestrator? maybe less quantized 3bit or 2bit

Small but important correction:

- **2-bit / 3-bit is more quantized**, not less.
- For a small 1.5B model doing planning, decomposition, and JSON schema adherence, **do not use 2-bit**.
- 3-bit is risky.
- Use **Q4_K_M minimum**.
- If RAM allows, use **Q5_K_M** for the planner.

Reason from first principles:

A 1.5B model has very little redundancy. When you quantize below 4-bit, the model loses precision in the weights responsible for:

- maintaining JSON structure,
- tracking dependencies,
- avoiding repeated loops,
- following strict schemas,
- preserving multi-step logic.

For planning, Q4_K_M is the safe lower bound.

---

# 2. Final Model Selection

For your constraints:

- 4-core CPU,
- 4GB RAM baseline,
- optionally 5GB RAM with judge,
- CPU-first,
- hot-swapping,
- low TPS acceptable,
- strong need for small footprint,

the best current practical choices are:

| Role | Model | Quantization | Approx RAM | Notes |
|---|---:|---:|---:|---|
| Persistent Orchestrator | Deterministic Python controller | N/A | 150–300MB | Should not be an LLM |
| Planner / Research / Bifurcator | `Qwen2.5-1.5B-Instruct` | Q4_K_M or Q5_K_M | 1.1–1.4GB | Hot-swapped |
| Coding Model | `Qwen2.5-Coder-1.5B-Instruct` | Q4_K_M | ~1.1–1.3GB | Hot-swapped |
| Optional Judge / Policy | `Qwen2.5-0.5B-Instruct` | Q4_K_M | ~400–600MB | Only if deterministic validation is insufficient |
| Fact Validation / Search Ranking | `BAAI/bge-reranker-base` or small cross-encoder | FP32/FP16 | ~300–500MB | Optional but useful |
| Intent / Policy micro-classifier | 100M–150M tiny model | FP16/INT8 | <200MB | Optional, replaces your 135M example |

### About the 135M example

A 135M model is not suitable for:

- planning,
- DAG generation,
- first-principles decomposition,
- system design,
- code planning.

It can be used only for very narrow tasks such as:

- intent classification,
- language detection,
- toxicity / policy triage,
- query routing,
- short text normalization.

For your architecture, the 135M model is optional, not central.

---

# 3. Critical Architecture Rule

## The persistent orchestrator should not be the LLM.

The persistent orchestrator should be a deterministic state machine.

The LLM is only called when the system needs:

- planning,
- re-planning,
- research summarization,
- code generation,
- code repair,
- validation summarization.

This gives you:

- predictable RAM,
- no always-loaded 1.5B model,
- safer hot-swapping,
- lower CPU usage,
- simpler observability.

So the architecture is:

```text
User Query
    |
    v
Deterministic Orchestrator
    |
    |-- Policy Filter
    |-- Resource Monitor
    |-- Context Compiler
    |-- Model Loader / Unloader
    |-- Sandbox Runner
    |-- SQLite State Machine
    |-- Observability Logger
    |
    |-- Hot-swap Planner Model
    |       |
    |       --> outputs DAG + Knowledge Brief
    |
    |-- Hot-swap Coding Model
            |
            --> executes DAG nodes via TDD loop
```

The LLM does not own the control loop. The control loop owns the LLM.

This is important for low-resource devices.

---

# 4. DAG Schema

You answered yes to strict schema.

The planner must output only JSON.

No markdown.  
No commentary.  
No code fences.  
No prose.

The output must validate against a strict schema before the system accepts it.

## 4.1. Canonical DAG Schema

Use this as the normative structure.

```json
{
  "schema_version": "1.0",
  "query_id": "string",
  "objective": "string",
  "resource_profile": {
    "cpu_cores": 4,
    "ram_mb": 5120,
    "context_window_tokens": 3072,
    "ssd_class": "nvme | sata | unknown",
    "network_available": true
  },
  "policy": {
    "allow_network": false,
    "allow_file_write": true,
    "allowed_write_paths": ["workspace"],
    "max_node_timeout_ms": 30000,
    "max_total_nodes": 24
  },
  "knowledge_artifacts": [
    {
      "id": "A01",
      "type": "knowledge_brief | search_result | encyclopedia | literature | code_contract | stdout_summary",
      "path": "artifacts/A01.md",
      "hash": "sha256:...",
      "summary": "string",
      "source_urls": ["https://..."]
    }
  ],
  "nodes": [
    {
      "id": "N01",
      "title": "Read file",
      "kind": "research | code | test | integrate | validate | plan",
      "language": "python | c | javascript | html | css | shell | none",
      "dependencies": [],
      "inputs": [
        {
          "name": "file_path",
          "type": "string | number | boolean | path | json",
          "source": "static | user | artifact",
          "value": "input.txt"
        }
      ],
      "outputs": [
        {
          "name": "file_text",
          "type": "string",
          "artifact_id": "A02"
        }
      ],
      "acceptance_criteria": [
        "Program exits with code 0",
        "File content is returned as UTF-8 text",
        "Missing file produces structured error"
      ],
      "context_refs": [
        "artifact:A01",
        "brief:N01"
      ],
      "sandbox": {
        "network": false,
        "filesystem_write_paths": ["workspace/N01"],
        "timeout_ms": 20000,
        "max_output_bytes": 65536
      },
      "retry_policy": {
        "max_attempts": 3,
        "backoff_ms": 500
      }
    }
  ]
}
```

---

## 4.2. Required Node Fields

Every node must contain:

```text
id
title
kind
language
dependencies
inputs
outputs
acceptance_criteria
context_refs
sandbox
retry_policy
```

Do not allow optional critical fields.

If a field is missing, reject the plan.

---

## 4.3. Node ID Rules

Use strict IDs:

```text
N01
N02
N03
...
N99
```

Regex:

```regex
^N[0-9]{2}$
```

Dependencies must refer only to previous valid node IDs.

Forbidden:

```json
"dependencies": ["N99"]
```

if `N99` does not exist.

Forbidden:

```json
"id": "N02",
"dependencies": ["N03"]
```

because forward dependencies create cycles.

The deterministic orchestrator must validate:

- no cycles,
- no missing dependencies,
- no duplicate IDs,
- no self-dependencies,
- no excessive node count.

---

## 4.4. Example Minimal DAG

For:

> Build a program to count words in a document.

The planner should output something like:

```json
{
  "schema_version": "1.0",
  "query_id": "q_2026_06_16_001",
  "objective": "Count words in a document",
  "resource_profile": {
    "cpu_cores": 4,
    "ram_mb": 5120,
    "context_window_tokens": 3072,
    "ssd_class": "unknown",
    "network_available": true
  },
  "policy": {
    "allow_network": false,
    "allow_file_write": true,
    "allowed_write_paths": ["workspace"],
    "max_node_timeout_ms": 30000,
    "max_total_nodes": 24
  },
  "knowledge_artifacts": [
    {
      "id": "A01",
      "type": "knowledge_brief",
      "path": "artifacts/A01.md",
      "hash": "sha256:pending",
      "summary": "Word counting requires reading text, normalizing whitespace, and counting tokens.",
      "source_urls": []
    }
  ],
  "nodes": [
    {
      "id": "N01",
      "title": "Read input file",
      "kind": "code",
      "language": "python",
      "dependencies": [],
      "inputs": [
        {
          "name": "file_path",
          "type": "path",
          "source": "user",
          "value": "input.txt"
        }
      ],
      "outputs": [
        {
          "name": "file_text",
          "type": "string",
          "artifact_id": "A02"
        }
      ],
      "acceptance_criteria": [
        "Function returns file content as string",
        "Function raises structured error if file is missing",
        "Function handles UTF-8 text"
      ],
      "context_refs": [
        "artifact:A01"
      ],
      "sandbox": {
        "network": false,
        "filesystem_write_paths": ["workspace/N01"],
        "timeout_ms": 20000,
        "max_output_bytes": 65536
      },
      "retry_policy": {
        "max_attempts": 3,
        "backoff_ms": 500
      }
    },
    {
      "id": "N02",
      "title": "Count words in string",
      "kind": "code",
      "language": "python",
      "dependencies": ["N01"],
      "inputs": [
        {
          "name": "text",
          "type": "string",
          "source": "artifact",
          "value": "A02"
        }
      ],
      "outputs": [
        {
          "name": "word_count",
          "type": "number",
          "artifact_id": "A03"
        }
      ],
      "acceptance_criteria": [
        "Empty string returns 0",
        "Whitespace-only string returns 0",
        "Multiple spaces are treated as one separator",
        "Function returns integer"
      ],
      "context_refs": [
        "artifact:A01"
      ],
      "sandbox": {
        "network": false,
        "filesystem_write_paths": ["workspace/N02"],
        "timeout_ms": 20000,
        "max_output_bytes": 65536
      },
      "retry_policy": {
        "max_attempts": 3,
        "backoff_ms": 500
      }
    },
    {
      "id": "N03",
      "title": "Integrate reader and counter",
      "kind": "integrate",
      "language": "python",
      "dependencies": ["N01", "N02"],
      "inputs": [
        {
          "name": "file_path",
          "type": "path",
          "source": "user",
          "value": "input.txt"
        }
      ],
      "outputs": [
        {
          "name": "final_word_count",
          "type": "number",
          "artifact_id": "A04"
        }
      ],
      "acceptance_criteria": [
        "CLI exits with code 0",
        "stdout contains final integer word count",
        "stderr contains structured error if input is invalid"
      ],
      "context_refs": [
        "artifact:A02",
        "artifact:A03"
      ],
      "sandbox": {
        "network": false,
        "filesystem_write_paths": ["workspace/N03"],
        "timeout_ms": 30000,
        "max_output_bytes": 65536
      },
      "retry_policy": {
        "max_attempts": 3,
        "backoff_ms": 500
      }
    }
  ]
}
```

---

# 5. Separate Plan From Runtime State

Do not let the LLM update runtime state inside the DAG.

The plan is immutable.

Runtime state lives separately.

Use SQLite.

## 5.1. SQLite Tables

### `plans`

```sql
CREATE TABLE plans (
    query_id TEXT PRIMARY KEY,
    objective TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

### `node_state`

```sql
CREATE TABLE node_state (
    query_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    status TEXT NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 0,
    started_at TEXT,
    ended_at TEXT,
    exit_code INTEGER,
    stdout_path TEXT,
    stderr_path TEXT,
    error_summary TEXT,
    PRIMARY KEY (query_id, node_id)
);
```

### `artifacts`

```sql
CREATE TABLE artifacts (
    artifact_id TEXT PRIMARY KEY,
    query_id TEXT NOT NULL,
    node_id TEXT,
    type TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    token_estimate INTEGER,
    created_at TEXT NOT NULL
);
```

### `events`

```sql
CREATE TABLE events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id TEXT,
    node_id TEXT,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

Valid node statuses:

```text
PENDING
READY
RUNNING
TEST_FAILED
CODE_FAILED
SANDBOX_VIOLATION
TIMEOUT
FAILED
DONE
```

The LLM never writes directly to these tables.

The deterministic orchestrator writes to them.

---

# 6. Sliding Window Design

You chose sliding window.

This is correct for low RAM.

But it must be dependency-aware, not just time-aware.

---

## 6.1. First Principle

The model context window is finite.

For Qwen2.5-1.5B:

- KV cache is approximately **0.22 MB per token** in FP16.
- 2048 tokens ≈ 450MB KV.
- 4096 tokens ≈ 900MB KV.
- 8192 tokens ≈ 1.8GB KV.

On a 4–5GB machine, you should not casually use large context.

Recommended operational context:

| Total RAM | Safe Context |
|---:|---:|
| 4GB | 2048 tokens |
| 5GB | 3072 tokens |
| 6GB+ | 4096 tokens |

For your system, default to **3072 tokens** if you have 5GB available.

If not, use **2048 tokens**.

---

## 6.2. Context Budget

Use a fixed prompt budget.

Example for 3072 tokens:

| Section | Token Budget |
|---|---:|
| System / policy / style | 180 |
| Current node spec | 350 |
| DAG topology summary | 250 |
| Knowledge brief | 350 |
| Dependency contracts | 600 |
| Sliding recent summaries | 650 |
| Error tail from previous attempt | 250 |
| Reserved for model output | 442 |

Total:

```text
3072 tokens
```

If using 4096 tokens, increase:

- dependency contracts,
- sliding summaries,
- output reserve.

Do not let the prompt compiler exceed the budget.

---

## 6.3. Sliding Window Rules

For each coding prompt, include only:

1. **Current node definition**
2. **Direct dependency contracts**
3. **Last 2 completed node summaries**
4. **Knowledge brief relevant to current node**
5. **Previous failed attempt error tail**, if retrying
6. **Tiger Style coding constraints**
7. **Output format instructions**

Do not include:

- full code from all previous nodes,
- full stdout from all previous nodes,
- full search results,
- entire DAG JSON,
- entire conversation history.

---

## 6.4. Dependency Contracts

When Node N05 depends on Node N02, do not include all of N02’s code.

Include only the contract.

For Python:

```python
def read_file(file_path: str) -> str:
    """
    Returns UTF-8 file text.
    Raises FileNotFoundError if missing.
    """
```

For C:

```c
int read_file(const char *path, char **out_text, size_t *out_len);
```

For JavaScript:

```js
export function readFile(path) { ... }
```

For HTML/CSS:

```text
Artifact: workspace/N02/index.html
Important IDs: #app, #result
Important classes: .card, .error
Expected behavior: static page renders result container
```

This is crucial.

The coding model usually does not need previous implementations.

It needs interfaces.

---

## 6.5. Eviction Policy

When the prompt exceeds budget:

Evict in this order:

1. Oldest non-dependency node summary.
2. Older knowledge brief snippets.
3. Middle portion of long stdout/stderr.
4. Full dependency code, replaced by contract.
5. Old DAG descriptions, replaced by minimal topology.

Never evict:

- current node spec,
- direct dependency contracts,
- output format,
- sandbox constraints,
- previous error tail during retry.

---

## 6.6. Artifact Compression

After each node completes, the orchestrator should store:

```text
/state/{query_id}/nodes/{node_id}/code.ext
/state/{query_id}/nodes/{node_id}/test.ext
/state/{query_id}/nodes/{node_id}/stdout.txt
/state/{query_id}/nodes/{node_id}/stderr.txt
/state/{query_id}/nodes/{node_id}/result.json
/state/{query_id}/nodes/{node_id}/summary.md
/state/{query_id}/nodes/{node_id}/contract.txt
```

The model prompt uses only:

```text
summary.md
contract.txt
result.json
error tail
```

Full files remain on disk.

---

# 7. TDD Loop With 3 Attempts

You chose 3 attempts.

Make it exact.

## 7.1. Definition of One Attempt

One attempt means:

```text
Generate or repair test
Generate or repair implementation
Run static validation
Run sandboxed test
Evaluate result
```

If any of these fail, it consumes one attempt.

This includes:

- syntax error,
- compile error,
- test crash,
- runtime timeout,
- failed assertion,
- invalid output format.

Sandbox violation should be treated specially:

```text
SANDBOX_VIOLATION = immediate FAILED
```

Do not retry sandbox violations by default.

---

## 7.2. TDD State Machine

For each node:

```text
attempt = 1

while attempt <= 3:
    load coding model
    build prompt from sliding window
    generate test and implementation
    write files to workspace
    run static validation
    run sandboxed test

    if success:
        mark node DONE
        store artifacts
        compress context
        break

    else:
        store stdout/stderr
        create error tail
        attempt += 1

if attempt > 3:
    mark node FAILED
    unload coding model
    trigger human review or replan
```

---

## 7.3. Language-Specific Validation

### Python

Use:

```bash
python3 -m py_compile code.py
python3 -m py_compile test_code.py
python3 test_code.py
```

Optional:

```bash
python3 -m pytest test_code.py
```

But `pytest` may not be installed. Plain `unittest` is safer.

### C

Use:

```bash
gcc -Wall -Wextra -Werror -O0 -o test_bin test_code.c code.c
./test_bin
```

Or if compiling separately:

```bash
gcc -Wall -Wextra -Werror -c code.c
gcc -Wall -Wextra -Werror -c test_code.c
gcc -o test_bin code.o test_code.o
./test_bin
```

### JavaScript

Use:

```bash
node --check code.js
node --check test_code.js
node test_code.js
```

For browser JS + HTML + CSS, avoid assuming DOM exists unless you create a minimal DOM stub.

### HTML

Do not run HTML as code.

Validate:

- well-formedness,
- required IDs,
- required classes,
- expected tag structure.

Use lightweight parsers:

- Python `html.parser`,
- `tidy` if installed,
- custom DOM assertion script.

### CSS

Validate:

- syntax,
- expected selectors,
- no invalid property blocks.

Use lightweight linting if available.

Do not use a headless browser on 4GB RAM unless absolutely necessary.

It is too heavy.

---

# 8. Research Model and Knowledge Context

You mentioned:

> pre compute a Knowledge Context Vector which will keep getting updated

This is correct in spirit, but the implementation should not be a mathematical vector.

The coding model cannot read a raw embedding vector.

It reads tokens.

So implement it as a:

```text
Knowledge Context Packet
```

or:

```text
Knowledge Brief
```

stored on disk.

---

## 8.1. Knowledge Brief Structure

Use Markdown.

Example:

```markdown
# Knowledge Brief

## Query
Count words in a document.

## Facts
1. Word counting requires reading file content as text.
2. Whitespace should be normalized before counting.
3. Empty files should return zero.

## Sources
- local encyclopedia chunk: wiki/text_processing.md
- searxng result: https://example.com/word-count

## Confidence
high

## Relevance To Nodes
- N01: file reading
- N02: tokenization and whitespace handling
```

The research model updates this file.

The coding model reads only the relevant portion.

---

## 8.2. Research Validation

For factual validation, use this priority:

1. Deterministic extraction.
2. Search retrieval.
3. Statistical reranking.
4. Optional tiny judge model.

Do not make the main research model judge itself.

Better flow:

```text
Research model produces claim
    |
    v
Extract atomic claims
    |
    v
Query SearXNG / local Wikipedia index
    |
    v
Rank snippets with small reranker
    |
    v
Accept / reject / mark low confidence
```

If a claim is low confidence:

- mark it in the knowledge brief,
- do not use it for code planning,
- or trigger a targeted research node.

---

## 8.3. Local Offline Option

If internet is unavailable, use:

- Wikipedia dump chunks,
- SQLite FTS5,
- BM25 ranking.

Do not use dense vector search as the primary method on this hardware.

BM25 / FTS5 is:

- smaller,
- faster,
- easier to debug,
- CPU-friendly,
- explainable.

Dense retrieval can be added later, but it is not the first-principles choice for 4-core / 4–5GB RAM.

---

# 9. Resource Monitoring and Dynamic Bifurcation

You want:

> if resource is less, divide the query further, store rest to disk, load later

Correct.

The orchestrator should monitor:

```python
cpu_percent
available_ram
swap_usage
disk_io
queue_depth
model_load_time
sandbox_runtime
```

Use `psutil`.

---

## 9.1. Resource Thresholds

Example thresholds:

```python
CPU_HIGH = 80
RAM_LOW_MB = 800
NODE_TIMEOUT_MS = 30000
```

If CPU > 80 for more than 5 seconds:

- reduce context window,
- reduce sliding window,
- reduce max output tokens,
- split current node.

If available RAM < 800MB:

- do not load extra judge model,
- reduce context window,
- use smaller knowledge brief,
- split node,
- avoid concurrent artifact processing.

---

## 9.2. Node Splitting

When a node is too large, do not re-plan the entire DAG.

Re-plan only the current node.

Use a narrow prompt:

```text
You are splitting one node into smaller nodes.

Current node:
{current_node_json}

Reason:
Resource constraint / repeated failure / excessive scope.

Rules:
- Output strict DAG JSON.
- Split into 2 or 3 smaller nodes.
- Each node must be independently testable.
- Do not change unrelated nodes.
- Preserve original node objective.
```

This prevents global re-planning blowups.

---

## 9.3. Prevent Infinite Splitting

You need hard limits.

Recommended defaults:

```text
max_total_nodes = 24
max_split_depth = 3
max_replans_per_query = 5
max_replans_per_node = 2
```

If exceeded:

```text
status = FAILED
reason = PLAN_EXPLOSION
```

Do not let the system recursively split forever.

---

# 10. Sandbox Design

You chose lightweight OS-level sandboxing.

The best option is:

```text
bubblewrap
```

Alternative:

```text
nsjail
```

Avoid:

```text
Docker
Podman
```

unless RAM is much larger.

---

## 10.1. Why `bubblewrap`

First principles:

- Docker requires daemon + container runtime + extra namespaces + storage overhead.
- `bubblewrap` uses Linux namespaces directly.
- `bubblewrap` has near-zero idle RAM overhead.
- `bubblewrap` is suitable for ephemeral code execution.

---

## 10.2. Example Sandbox Command

```bash
bwrap \
  --ro-bind /usr /usr \
  --ro-bind /bin /bin \
  --ro-bind /lib /lib \
  --ro-bind /lib64 /lib64 \
  --ro-bind /etc/alternatives /etc/alternatives \
  --proc /proc \
  --dev /dev \
  --tmpfs /tmp \
  --bind /state/query_123/workspace/N01 /workspace \
  --unshare-pid \
  --unshare-net \
  --unshare-ipc \
  --die-with-parent \
  python3 /workspace/test_code.py
```

For C:

```bash
bwrap \
  --ro-bind /usr /usr \
  --ro-bind /bin /bin \
  --ro-bind /lib /lib \
  --ro-bind /lib64 /lib64 \
  --proc /proc \
  --dev /dev \
  --tmpfs /tmp \
  --bind /state/query_123/workspace/N01 /workspace \
  --unshare-pid \
  --unshare-net \
  --unshare-ipc \
  --die-with-parent \
  /bin/sh -c "gcc -Wall -Wextra -Werror -o /workspace/test_bin /workspace/test_code.c /workspace/code.c && /workspace/test_bin"
```

For Node:

```bash
bwrap \
  --ro-bind /usr /usr \
  --ro-bind /bin /bin \
  --ro-bind /lib /lib \
  --ro-bind /lib64 /lib64 \
  --proc /proc \
  --dev /dev \
  --tmpfs /tmp \
  --bind /state/query_123/workspace/N01 /workspace \
  --unshare-pid \
  --unshare-net \
  --unshare-ipc \
  --die-with-parent \
  node /workspace/test_code.js
```

---

## 10.3. Sandbox Rules

Default sandbox policy:

```json
{
  "network": false,
  "write_paths": ["workspace"],
  "read_paths": ["workspace", "artifacts"],
  "timeout_ms": 30000,
  "max_output_bytes": 65536,
  "max_file_write_bytes": 1048576,
  "allow_subprocess": false,
  "allow_env_access": false
}
```

If a node requires network, it must explicitly declare:

```json
"network": true
```

But default should be false.

---

# 11. Observability

You need observability from day one.

Do not add it later.

## 11.1. Structured Events

Log JSONL events:

```json
{
  "ts": "2026-06-16T00:00:00Z",
  "query_id": "q123",
  "node_id": "N02",
  "event": "node.attempt.start",
  "attempt": 1,
  "model": "qwen2.5-coder-1.5b-instruct-q4_k_m",
  "context_tokens": 2810,
  "cpu_percent": 61.0,
  "ram_available_mb": 2048
}
```

Important events:

```text
query.received
policy.accepted
policy.rejected
planner.load.start
planner.load.end
plan.generated
plan.schema.invalid
plan.accepted
node.ready
node.attempt.start
sandbox.start
sandbox.exit
sandbox.timeout
sandbox.violation
node.test.failed
node.done
node.failed
model.unload.start
model.unload.end
resource.threshold.crossed
node.split
knowledge.brief.updated
```

---

## 11.2. Metrics to Track

Track:

```text
model_load_time_ms
model_unload_time_ms
tokens_in
tokens_out
tokens_per_second
prompt_build_time_ms
sandbox_runtime_ms
node_attempt_count
node_failure_count
schema_invalid_count
context_truncation_count
ram_available_mb
cpu_percent
kv_cache_estimate_mb
hot_swap_count
```

This gives you debuggability.

Without this, the system will be impossible to tune.

---

# 12. Hot-Swap Flow

The exact sequence should be:

```text
1. Orchestrator receives query.
2. Policy filter runs.
3. Resource monitor checks available RAM/CPU.
4. Orchestrator loads Planner Model.
5. Planner generates:
   - DAG JSON
   - Knowledge Brief
   - Context packet
6. Orchestrator validates DAG schema.
7. Orchestrator saves DAG to SQLite/disk.
8. Orchestrator unloads Planner Model.
9. Orchestrator selects first READY node.
10. Orchestrator loads Coding Model.
11. Prompt compiler builds sliding-window prompt.
12. Coding model generates test/code.
13. Sandbox runs test.
14. Orchestrator stores artifacts.
15. If success:
      mark node DONE
      compress outputs
      update dependencies
   If failure:
      increment attempt
      retry up to 3
16. When node completes, unload or keep coding model depending on next node.
17. If next node is research:
      unload coding model
      load planner/research model
18. If next node is coding:
      keep coding model loaded if RAM permits
      otherwise hot-swap
19. Repeat until terminal node.
20. Format final output.
```

---

# 13. Recommended Inference Runtime

For CPU-first low-RAM devices, use:

```text
llama.cpp
```

or a compatible GGUF runtime.

Reasons:

- excellent CPU inference,
- GGUF quantization,
- low overhead,
- controllable context size,
- supports constrained output via grammar,
- can run local models without heavy Python GPU stack.

Important settings:

```text
--threads 4
--ctx-size 2048 or 3072
--batch-size small
--mlock optional
--no-gpu if CPU-only
```

If RAM is tight, consider quantized KV cache if your runtime supports it.

But test carefully because KV quantization can reduce code reliability.

---

# 14. What Still Looks Vague

There are still several points that need precise answers before implementation.

---

## Question 1: Target operating system?

`bubblewrap` requires Linux.

Are the lightweight devices running:

- Linux?
- Raspberry Pi OS?
- Ubuntu?
- Debian?
- Yocto?
- Android/Termux?
- Windows?
- macOS?

If not Linux, the sandbox design changes.

---

## Question 2: Context window target?

Choose one:

```text
A. 2048 tokens  safest for 4GB RAM
B. 3072 tokens  balanced for 5GB RAM
C. 4096 tokens  better but higher KV cache
```

My recommendation:

```text
Default: 3072 tokens if 5GB RAM is guaranteed.
Fallback: 2048 tokens if RAM pressure is detected.
```

---

## Question 3: Are language runtimes installed?

For C, JS, Python, HTML, CSS, we need to know what exists on device.

Are these installed?

```text
python3
gcc
clang
node
npm
tidy
css validator
headless browser
```

If not installed, the sandbox and TDD design must adapt.

---

## Question 4: Is internet available?

You mentioned SearXNG.

Does the device have reliable internet?

If yes:

```text
Use SearXNG + optional Wikipedia API.
```

If no:

```text
Need local Wikipedia / literature index.
Use SQLite FTS5 + BM25.
```

This changes storage and RAM planning.

---

## Question 5: Do you actually require a knowledge graph now?

You mentioned knowledge graphs.

A real knowledge graph requires:

- entity extraction,
- relation extraction,
- triple store,
- graph queries,
- graph validation.

That is heavy for 4GB RAM.

My recommendation:

```text
MVP:
Use atomic claims + search snippets + text knowledge brief.

Later:
Add SQLite-based triple store if needed.
```

Do you want knowledge graph in the first version, or can it be deferred?

---

## Question 6: Does the 3-attempt limit include test generation failures?

You said 3 attempts.

Need exact definition.

Option A:

```text
3 total attempts for everything:
test generation, code generation, syntax errors, runtime failures.
```

Option B:

```text
3 implementation attempts after a valid test is generated.
Test repair has a separate small budget.
```

Option A is simpler and safer.

Option B is more flexible but more complex.

Which do you want?

---

## Question 7: Maximum DAG size?

Need hard limits.

Recommended:

```text
max_total_nodes = 24
max_split_depth = 3
max_replans_per_query = 5
max_replans_per_node = 2
```

Do you accept these defaults?

---

## Question 8: How should HTML/CSS be validated?

HTML/CSS cannot be tested like Python or C unless you use a browser engine.

A headless browser is too heavy for 4GB RAM.

Options:

```text
A. Static linting + DOM structure assertions.
B. Static linting only.
C. Headless browser, but heavy.
```

My recommendation:

```text
A for HTML: parse DOM, check IDs/classes/structure.
B for CSS: syntax and selector checks.
No headless browser in MVP.
```

Do you accept that?

---

## Question 9: What storage format do you want for state?

Recommended:

```text
SQLite for state
Files for artifacts
```

Alternative:

```text
Pure JSON files
```

SQLite is better because it gives:

- atomic updates,
- queryable state,
- crash recovery,
- simpler observability.

Do you accept SQLite?

---

## Question 10: What policies must be enforced?

You mentioned policy.

Need exact policy classes.

Examples:

```text
PII redaction
secret detection
malicious code blocking
network blocking
file path restriction
copyright filtering
unsafe command blocking
prompt injection filtering
```

Which policies are mandatory for the first version?

---

# 15. My Current Recommended Defaults

If you want the cleanest low-resource design, use these defaults:

```text
Persistent orchestrator: deterministic Python controller
Planner model: Qwen2.5-1.5B-Instruct Q5_K_M or Q4_K_M
Coding model: Qwen2.5-Coder-1.5B-Instruct Q4_K_M
Judge model: optional Qwen2.5-0.5B-Instruct Q4_K_M
Fact validation: SearXNG + small reranker + atomic claims
Knowledge graph: deferred
State: SQLite
Artifacts: filesystem
Context window: 3072 tokens, fallback to 2048
Sliding window: dependency-aware
TDD attempts: 3 total attempts per node
Sandbox: bubblewrap
HTML/CSS validation: static/DOM assertions, no headless browser
Max nodes: 24
Max split depth: 3
Max replans per node: 2
Max replans per query: 5
```

---

# 16. The Next Concrete Build Step

The next thing to define is not the model prompt.

It is the **Orchestrator Contract**.

That contract should define:

```text
1. Input format
2. Policy decision format
3. Planner invocation format
4. DAG validation rules
5. Node scheduling rules
6. Sandbox invocation rules
7. Artifact storage layout
8. Retry policy
9. Sliding window prompt compiler
10. Observability event schema
```

Once that is fixed, the models become replaceable components.

That is the correct first-principles order.
