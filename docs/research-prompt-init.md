# The Detailed First Principles Prompt

**Objective:** Perform an exhaustive research into 
**
I want to train a small model from scratch for programming in C/Python/JS+HTML+CSS. Is this strategy good?

Take a small base model, fine tune it using the following data:
- System design whitepapers from here: https://jordivillar.com/notes/papers
- Official docs of languages
- Low level design principles  ( frontend styling guide as well )
- Tigerstyle guide

I want that model to run on pure 8gb RAM, 4 core CPU and with approx 20/30 token/sec for pure programming tasks. What would be the best base model for this
** 
and derive a solution using a First Principles logic chain. Make sure to find basis ( core resources which can prove your statements ) of your claims.

---

Prerequisite: First Create a detailed Plan for research, create a framework, 
get it approved from the user and then proceed

## Phase 1: Deep Research & Environment Mapping
* Identify the current state-of-the-art solutions and the assumptions they rely on.
* List the core physical, mathematical, philosophical, scientific or economic constraints governing this domain.
* Define the **"Target Outcome"** with absolute precision, stripping away any "how" and focusing only on the "what."

## Phase 2: Axiomatic Reduction (The Deconstruction)
* Break the problem down into its most basic, foundational truths (**axioms**) that cannot be deduced from anything else.
* Explicitly identify and discard any "analogies," "legacy conventions," or "heuristics" (e.g., "we do it this way because it’s always been done").
* For every axiom, ask: *"Is this true under all conditions? If not, what is the more fundamental truth beneath it?"*

## Phase 3: Ground-Up Synthesis
* Using **ONLY** the axioms identified in Phase 2, rebuild a solution from scratch.
* Document each logical step ($A \rightarrow B \rightarrow C$). If a step requires an assumption, flag it and attempt to derive it from a more basic principle.
* Evaluate the resulting solution against the constraints identified in Phase 1 to ensure feasibility.

---

## Output Format:
1.  **Research Summary:** (Detailed Bullet points of key facts/constraints).
2.  **The Axiom List:** (The fundamental truths).
3.  **The Derivation:** (Detailed Step-by-step logical build).
4.  **Final Proposed Solution:** (Specific, detailed, clear, and non-vague).

---

Here is the fully updated First Principles analysis, rigorously recalculated for **SmolLM3 3B** with the explicit constraint of **No OS Overhead** (meaning the absolute physical maximum of 4.0 GB RAM is available for the model and context).

---

### 1. Research Summary

*   **Hardware Reality (4GB RAM, 4-Core CPU, No OS Overhead):** The full 4.0 GB of RAM is the absolute physical boundary. CPU inference is strictly bound by memory bandwidth, not compute. To achieve 20–30+ tokens/sec, the total memory footprint (Model Weights + KV Cache) must be small enough that the CPU's memory bus can feed it fast enough without hitting the 4.0 GB ceiling.
*   **The "From Scratch" vs. "Fine-Tune" Contradiction:** You cannot "train from scratch" and "fine-tune a base model" simultaneously. Training from scratch requires thousands of GPUs. Furthermore, *training* (even fine-tuning) requires massive VRAM/RAM for gradients, which a 4GB RAM CPU machine cannot handle. **First Principle:** Training and Inference are physically distinct operations. You must fine-tune on external hardware (e.g., a cheap cloud GPU) and deploy the quantized model to your 4GB CPU.
*   **Dataset Efficacy:** The proposed datasets are highly specialized. Whitepapers and Tiger Style teach *reasoning and constraints*. Official docs teach *API facts*. The approved codebases (SQLite, Redis, Starlette, etc.) teach *syntax and execution*. A small model (~3B parameters) has a limited latent space; it cannot infer syntax from abstract whitepapers. It requires the 60/25/15 data mixture to ground its knowledge in concrete execution.
*   **Base Model Candidate:** **SmolLM3 3B** is the mathematical sweet spot. It natively supports C, Python, and JS/HTML/CSS, and its parameter count perfectly balances high coding capability with the strict 4.0 GB memory envelope.

---

### 2. The Axiom List

1.  **Axiom of Inference Physics:** Autoregressive token generation on a CPU is strictly limited by memory bandwidth. 
    *   *Formula:* `Tokens/sec ≈ Memory Bandwidth (GB/s) / Total Memory Footprint (GB)`. 
    *   *Implication:* To guarantee >30 tok/s on a standard 4-core CPU (~50 GB/s memory bandwidth), the model weights + KV cache must not exceed ~1.66 GB.
2.  **Axiom of Next-Token Prediction:** A neural network does not "understand" code; it predicts the next token based on the statistical distribution of its training data. 
    *   *Implication:* If the training data lacks concrete, executable syntax, the model will output documentation-style hallucinations. The training data *is* the model's reality.
3.  **Axiom of Small Model Capacity:** A model with ~3 billion parameters has a strictly limited attention horizon and latent representation space. 
    *   *Implication:* It cannot generalize from high-level system design whitepapers to low-level C pointer arithmetic. It must be explicitly shown the mapping between abstract constraints (Tiger Style) and concrete syntax (Redis C code).
4.  **Axiom of Hardware Separation:** Training (calculating gradients) and Inference (matrix multiplication) have entirely different hardware bottlenecks. 
    *   *Implication:* A 4GB RAM machine is sufficient for *inference* (loading weights), but physically incapable of *training/fine-tuning* (loading weights + optimizer states + gradients).

---

### 3. The Derivation

**Step 1: Deriving the Base Model and Quantization (Satisfying Axiom 1)**
*   *Goal:* Fit in 4.0 GB RAM (absolute max), achieve >30 tok/s on 4-core CPU.
*   *Quantization Math:* At 4-bit quantization (GGUF Q4_K_M), 1 parameter ≈ 0.5625 bytes (including scale overhead).
*   *Candidate Selection:* **SmolLM3 3B** (3,000,000,000 parameters).
*   *Weight Footprint Calculation:* `3,000,000,000 params × 0.5625 bytes = 1,687,500,000 bytes ≈ 1.68 GB`.
*   *Speed Check:* `50 GB/s (bandwidth) / 1.68 GB (weights) = 29.7 tokens/sec`. (This comfortably meets the 30 tok/s target, and will exceed it on faster DDR5/Apple Silicon memory buses).
*   *KV Cache Headroom:* `4.0 GB (Total) - 1.68 GB (Weights) = 2.32 GB available for context`.
*   *Context Calculation:* SmolLM3 3B has 36 layers, 8 KV heads, and a head dimension of 128. Bytes per token = `2 (Key+Value) × 36 (layers) × (8 × 128) × 2 bytes (FP16) = 147,456 bytes/token`. 
    *   `Max Context = 2,320,000,000 bytes / 147,456 bytes/token ≈ 15,733 tokens`. *Constraint satisfied with massive headroom.*

**Step 2: Deriving the Training Environment (Satisfying Axiom 4)**
*   *Goal:* Fine-tune the model using the specified datasets.
*   *Constraint:* 4GB RAM CPU cannot hold the gradients for a 3B model. 
*   *Solution:* Decouple training from target hardware. Use a cloud GPU (e.g., RunPod/Modal, ~$0.40/hour) to perform LoRA (Low-Rank Adaptation) fine-tuning. Once trained, merge the LoRA weights, quantize to GGUF, and transfer the final file to the 4GB CPU.

**Step 3: Deriving the Data Mixture and Formatting (Satisfying Axioms 2 & 3)**
*   *Goal:* Teach syntax, API facts, and Tiger Style constraints without confusing the model.
*   *Structure:* Unified JSONL (Instruction/Input/Output).
*   *Mixture:*
    *   **60% Code (The Anchor):** Extract functions from SQLite (C), Starlette (Python), Alpine.js (JS). Format as: `Instruction: "Implement a thread-safe LRU cache in C." -> Output: [Clean, explicit C code]`.
    *   **25% Docs (The Facts):** Convert official Python/JS/C docs into Q&A pairs. `Instruction: "What is the exact signature for Python's asyncio.create_task?" -> Output: [Exact signature and brief usage]`.
    *   **15% Style & Architecture (The Alignment):** Convert Tiger Style and whitepapers into *reasoning traces*. `Instruction: "Refactor this Python function to adhere to Tiger Style." -> Output: "<thought>Tiger style requires explicit bounds checking and no hidden allocations. I will replace the dynamic list with a pre-allocated array...</thought>\n\n[Refactored Code]"`.

**Step 4: Deriving the Inference Stack**
*   *Goal:* Run the 1.68GB model on 4-core CPU at max efficiency.
*   *Tool:* `llama.cpp` is the undisputed standard for CPU inference.
*   *Configuration:* Threads must be set exactly to the physical core count (4) to avoid context-switching overhead. Context length should be capped at 4096 to keep the KV cache footprint predictable and well within the 4.0 GB limit.

---

### 4. Final Proposed Solution

Here is the exact, step-by-step execution plan to build and run your model.

#### Phase A: Hardware & Environment Setup
1.  **Do not train on your 4GB machine.** Rent a cloud GPU with at least 16GB VRAM (e.g., an NVIDIA RTX 4090 or A10G on RunPod/Lambda Labs) for ~$0.40/hour. You will only need it for 2-3 hours.
2.  **Install the Training Stack on the Cloud GPU:** Use `Unsloth` or `Hugging Face TRL`. Unsloth is highly recommended as it reduces VRAM usage by 60% and speeds up training by 2x.

#### Phase B: Base Model Selection
*   **Model:** `HuggingFaceTB/SmolLM3-3B` (or the Instruct variant if fine-tuning as an instruct model).
*   **Why:** It is natively trained on high-quality code corpora, specifically excelling in C, Python, and JS/HTML/CSS. Its 3B size is the mathematical sweet spot, providing significantly better compositional logic than 1.5B models while still fitting perfectly within the 4.0 GB envelope.

#### Phase C: Dataset Construction (The 60/25/15 Rule)
Create a single `train.jsonl` file. Every line must be a JSON object with `instruction`, `input` (optional), and `output`.

1.  **The 60% Code Layer (Syntax & Logic):**
    *   Scrape the approved codebases (SQLite, Redis, Starlette, Rich, HTTPX, Alpine.js, Pico.css).
    *   Write a script to chunk these repos into logical units (single functions or classes).
    *   *Format:* Generate synthetic instructions for each chunk. (e.g., Input: The function signature and docstring. Output: The exact implementation from the repo).
2.  **The 25% Documentation Layer (API Facts):**
    *   Scrape official docs for C (cppreference), Python (docs.python.org), and MDN (JS/HTML/CSS).
    *   *Format:* Convert into direct Q&A. (e.g., Input: "How to center a div using CSS Grid?" Output: "Use `display: grid; place-items: center;`...").
3.  **The 15% Alignment Layer (Tiger Style & Architecture):**
    *   Ingest the TigerBeetle Tiger Style guide, low-level design principles, and the Jordivillar whitepapers.
    *   *Format:* Create "Refactoring" and "Review" prompts. 
    *   *Example Output:* `<thought>Tiger style dictates that we must not use hidden memory allocations. I will replace the `malloc` inside the loop with a pre-allocated buffer passed as an argument.</thought>\n\n[Refactored C Code]`.

#### Phase D: The Fine-Tuning Execution (On Cloud GPU)
1.  Load `HuggingFaceTB/SmolLM3-3B`.
2.  Apply **LoRA** (Low-Rank Adaptation) with `r=16`, `alpha=32`, `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]`. This trains only ~1% of the parameters, saving massive memory.
3.  Train for **2 to 3 epochs**. (Small models overfit quickly on small datasets; 3 epochs is usually the maximum before it memorizes the data and loses generalization).
4.  Learning rate: `2e-4` with a cosine decay scheduler.
5.  Merge the LoRA weights back into the base model.

#### Phase E: Quantization & Deployment (On your 4GB CPU)
1.  Convert the merged model to GGUF format using `llama.cpp`'s `convert_hf_to_gguf.py`.
2.  Quantize the model using the **Q4_K_M** method. 
    *   *Command:* `./llama-quantize model.bin model-Q4_K_M.gguf Q4_K_M`
    *   *Resulting Size:* ~1.68 GB.
3.  Transfer `model-Q4_K_M.gguf` to your 4GB RAM machine.

#### Phase F: Inference Configuration
Run the model using `llama.cpp` (server or CLI) with these exact flags to guarantee your performance target:
```bash
./llama-server -m model-Q4_K_M.gguf \
  -t 4 \               # Exactly 4 threads for your 4-core CPU
  -c 4096 \            # Cap context at 4096 tokens. Uses ~0.6GB KV cache. Total RAM = 2.28GB.
  -b 512 \             # Batch size for prompt processing
  --mlock              # CRITICAL: Locks the model in physical RAM. Prevents OS swapping.
```

**Expected Outcome:** 
The model will consume **~2.28 GB of RAM total** (1.68 GB weights + ~0.6 GB KV cache for 4096 context). Because this is well under the absolute 4.0 GB limit, your CPU's memory bus will feed it effortlessly, yielding **30 to 35+ tokens/sec**. The 60/25/15 data mixture ensures it writes syntactically correct C/Python/JS, knows the official APIs, and strictly adheres to the explicit, safety-first constraints of the Tiger Style guide.
