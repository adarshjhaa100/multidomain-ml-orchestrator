# MULTIDOMAIN ORCHESTRATOR SKELETON

Aim: Orchestrator which takes input, filters it, formats it, applies policy, decides parameters, guides it to relevant harness, formats output, observability. It has harness which - Over the course of answer, it repeatedly validated them, if they come out to be wrong (hallucinates), then fix the prompt and run again automatically.

Functionality to be covered now:
- Model for Top level harness: (Maybe use the generic model or something else) - across all
- Embedding Model (For documents and input ) - across all
- (Generic Model) English + Encyclopedia + Literature based research model ( can be used as top level model ) - Wikipedia or some encyclopedia? Plus understand normal queries. Give short/long responses based on tunable params.
  - Existing
    - (NOPE, too putrid) Use this as an example ref: https://huggingface.co/AxiomicLabs/GPT-X2.5-135M
    - Or, find something using wikipedia (https://huggingface.co/datasets/wikimedia/structured-wikipedia)
  - Able to do Research Plan, or even for the current query, able to bifurcate and drill down problems using first principle
- Coding Model
  - Will use an harness agent with proper sandboxed environment and Test driven development
  - Generate coding scripts in: C, JS + HTML + CSS, Python
  - Capable of Planning based on system design
  - Continuously Interact with the previous model through harness


PROBLEM OF LIGHTWEIGHT DEVICES;
- Slow TPS, processing
- Lightweight devices ( ~ 4Core CPU, 4GB RAM) to moderate. 
- Prefer CPU first. Top level harness and other harnesses: Keep bifurcating query top down to save compute resources based on compute resource availability. Harness will take the result, evaluate and validate and query for the next steps.

---

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
