## SUMMARY
# Final Architecture

The system is a Linux-based Harness(execution wrapper that validates and retries model output) controlled by an Orchestrator(deterministic controller, not an always-loaded LLM). It hot-swaps small local models, uses tiered RAG(retrieval-augmented generation) for grounded knowledge, plans work as a DAG(directed acyclic graph of tasks), executes code in a lightweight Sandbox(isolated execution environment), and stores all state on disk.

---

## Locked System Constraints

| Constraint | Decision |
|---|---|
| OS | Linux |
| CPU target | ~4 cores |
| RAM target | 4–5GB |
| Inference | CPU-first |
| Context window | 3072 tokens |
| Model format | GGUF(quantized model file format) |
| Runtime | llama.cpp(CPU inference runtime) |
| State storage | SQLite |
| Artifact storage | Filesystem |
| Sandbox | bubblewrap(lightweight Linux namespace sandbox tool) |
| Research | Hybrid local RAG + online SearXNG |
| Coding validation | TDD(test-driven development) with sandbox execution |
| Retry policy | Valid test first, then exactly 3 code-fix attempts |
| HTML/CSS validation | Static parsing/linting with model unloaded |
| Resource behavior | Shrink context, use cache, or split node under pressure |

---

# Core Architecture Layers

## 1. Input and Policy Layer

Responsibilities:

- receive query,
- normalize query,
- apply safety policy,
- apply resource policy,
- decide if query is research, coding, or mixed.

Outputs:

- accepted query,
- policy flags,
- resource profile.

---

## 2. Orchestrator Layer

Specific tools:

- Python controller,
- SQLite state store,
- psutil(resource monitor),
- event logger.

Responsibilities:

- manage query lifecycle,
- monitor CPU/RAM,
- load and unload models,
- validate plans,
- schedule nodes,
- manage retries,
- store artifacts,
- trigger sandbox,
- log observability events.

Rule:

The Orchestrator owns the loop. Models are temporary tools.

---

## 3. RAG Layer

RAG is now a first-class layer.

### RAG Components

| Component | Specific Choice | Purpose |
|---|---|---|
| Corpus(collection of source documents) | curated docs, Wikipedia, literature, error references | trusted knowledge |
| Chunker(splits documents) | deterministic text splitter | small retrievable units |
| Sparse Index(keyword index) | SQLite FTS5 + BM25(keyword ranking) | primary low-RAM retrieval |
| Dense Index(optional semantic index) | sqlite-vec or FAISS with INT8 vectors | optional semantic fallback |
| Embedding Model(optional semantic encoder) | `bge-small-en-v1.5` or `all-MiniLM-L6-v2` via ONNX Runtime INT8 | semantic search when needed |
| Reranker(optional evidence scorer) | lexical heuristics first; optional small cross-encoder | improve top evidence |
| RAG Router(decides retrieval source) | deterministic logic | choose cache, local sparse, local dense, or online |
| Cache(saved retrieval results) | SQLite | avoid repeated work |
| Knowledge Brief(short evidence summary) | text file/database record | compact evidence for models |

---

## 4. RAG Retrieval Policy

Use this exact priority:

1. Artifact Cache  
   - previous node outputs,
   - previous briefs,
   - dependency contracts.

2. Local Sparse RAG  
   - SQLite FTS5,
   - BM25,
   - best for exact names, errors, APIs, standards.

3. Local Dense RAG, optional  
   - load small embedding model only when needed,
   - unload after retrieval,
   - use only if sparse confidence is low.

4. Online RAG  
   - SearXNG(meta search engine),
   - use only when local confidence is low or fresh data is required,
   - cache results.

Resource rule:

Sparse RAG is primary. Dense RAG is optional. Online RAG is last.

---

## 5. Local RAG Source Packs

### Documentation Pack

For coding tasks.

Sources:

- Python docs,
- C library references,
- GCC error references,
- Node.js docs,
- JavaScript reference,
- HTML structure references,
- CSS selector references.

Best use:

- exact API lookup,
- compiler/runtime error repair,
- syntax validation hints.

---

### Encyclopedia Pack

For research tasks.

Sources:

- selected Wikipedia chunks,
- structured Wikipedia,
- literature summaries,
- science/history reference chunks.

Best use:

- factual grounding,
- short explanations,
- literature-based answers.

---

### Error Pack

For repair loops.

Sources:

- common Python errors,
- common GCC errors,
- common Node.js errors,
- common sandbox failures,
- common lint failures.

Best use:

- retrieve known fixes after a test fails.

---

## 6. Planner Layer

Specific model:

- Planner(model that creates plan and research brief): Qwen2.5-1.5B-Instruct(1.5B general instruction model)
- Quantization(model compression): Q4_K_M(4-bit preset) minimum, Q5_K_M(5-bit preset) preferred if RAM allows.

Responsibilities:

- understand query,
- use Knowledge Brief,
- create strict DAG,
- assign acceptance criteria,
- assign node language,
- assign sandbox constraints,
- split large tasks when needed.

Outputs:

- strict DAG,
- Knowledge Brief,
- context references.

---

## 7. DAG and State Layer

DAG rules:

- strict schema,
- no cycles,
- no missing dependencies,
- no duplicate node IDs,
- max node count,
- immutable once accepted.

State stored separately:

- node status,
- attempt count,
- start/end time,
- sandbox exit code,
- error summary,
- artifact paths.

Storage:

- SQLite for state,
- filesystem for artifacts.

---

## 8. Context Layer

Specific mechanism:

- Context Compiler(prompt builder)
- Sliding Window(moving selection of recent context)
- Dependency Contract(interface summary of previous node)

Context budget:

- max 3072 tokens,
- include only current node,
- include direct dependency contracts,
- include relevant Knowledge Brief,
- include last few summaries,
- include previous error tail during repair.

Exclusions:

- no full old code,
- no full logs,
- no full search results,
- no full DAG details unless required.

---

## 9. Coding Layer

Specific model:

- Coding Model(model that writes tests and code): Qwen2.5-Coder-1.5B-Instruct(1.5B coding model)
- Quantization: Q4_K_M

Responsibilities:

- generate test first,
- generate implementation,
- repair implementation after test failure,
- respect sandbox constraints,
- produce minimal explicit code.

Languages:

- Python,
- C,
- JavaScript,
- HTML,
- CSS.

---

## 10. Sandbox and Validation Layer

Specific tool:

- bubblewrap

Sandbox policy:

- no network by default,
- restricted write paths,
- timeout,
- output size limit,
- no unauthorized subprocesses.

Language validators:

- Python: syntax check + test execution,
- C: compile with GCC + run test binary,
- JavaScript: syntax check + Node.js execution,
- HTML: static parse with lxml(Python HTML/XML parser),
- CSS: static parse with lxml or jsdom(Node.js DOM parser).

HTML/CSS rule:

Use Compute Shifting(unload model to free RAM) before running HTML/CSS validation.

---

## 11. TDD Loop

Phase 1: Test generation

- model generates test,
- test must become structurally valid,
- test repair attempts are limited.

Phase 2: Code repair

- model generates implementation,
- sandbox runs valid test,
- if test fails, model receives error tail,
- exactly 3 code-fix attempts are allowed.

Failure rule:

Sandbox violation fails immediately.

---

## 12. Resource-Aware Splitting

Resource Monitor watches:

- CPU,
- RAM,
- context size,
- retry count,
- node complexity.

If pressure is high:

1. reduce context window,
2. shrink Knowledge Brief,
3. use cached RAG,
4. split current node,
5. store remaining work on disk,
6. resume later.

Limits:

- max total nodes: 24,
- max split depth: 3,
- max replans per node: 2,
- max replans per query: 5.

---

## 13. Observability Layer

Log events:

- query received,
- policy accepted/rejected,
- RAG retrieval mode,
- planner loaded/unloaded,
- plan accepted/rejected,
- node started,
- test generated,
- sandbox started,
- sandbox failed,
- node completed,
- node failed,
- resource threshold crossed.

Metrics:

- CPU,
- RAM,
- tokens in/out,
- model load time,
- sandbox runtime,
- retry count,
- retrieval source,
- cache hit rate.

---

# Full Execution Flow

1. Query enters.
2. Policy checks query.
3. Resource Monitor selects safe context size.
4. RAG Router checks Artifact Cache.
5. Local Sparse RAG searches SQLite FTS5.
6. If needed, Local Dense RAG runs with small embedding model.
7. If still needed, Online RAG queries SearXNG.
8. Retrieved evidence is compressed into Knowledge Brief.
9. Planner model loads.
10. Planner creates DAG and Knowledge Brief references.
11. DAG Validator checks schema and dependencies.
12. Plan is saved.
13. Planner unloads.
14. Coding Model loads.
15. Context Compiler builds prompt from node, dependencies, brief, and error history.
16. Coding Model generates test.
17. Sandbox validates test.
18. Coding Model generates implementation.
19. Sandbox runs test.
20. If test fails, Error Pack RAG retrieves known fixes.
21. Coding Model repairs code up to 3 times.
22. Completed node artifacts are stored and compressed.
23. Next node becomes ready.
24. Final Output Formatter produces final answer.

---

# Build Steps

## Step 1: Build Base Skeleton

Build:

- project folders,
- config,
- controller,
- SQLite state store,
- artifact store,
- event logger,
- resource monitor.

Outcome:

- system can receive query,
- system can store state,
- system can log events,
- system can monitor resources,
- system can resume after crash.

---

## Step 2: Build Local Sparse RAG

Build:

- corpus ingestion,
- chunking,
- metadata storage,
- SQLite FTS5 index,
- BM25 retrieval.

Sources:

- documentation pack,
- encyclopedia pack,
- error pack.

Outcome:

- offline keyword retrieval works,
- no extra model required,
- low RAM usage,
- evidence can be attached to planner/coder prompts.

---

## Step 3: Build Knowledge Brief Builder

Build:

- evidence selection,
- source ranking,
- short summary creation,
- confidence marking,
- cache storage.

Outcome:

- models receive compact grounded evidence,
- context stays under 3072 tokens,
- hallucination is reduced.

---

## Step 4: Build Model Runner

Build:

- llama.cpp wrapper,
- model loading,
- model unloading,
- hot-swap logic,
- token limit enforcement.

Models:

- Qwen2.5-1.5B-Instruct for Planner,
- Qwen2.5-Coder-1.5B-Instruct for Coding Model,
- optional Qwen2.5-0.5B-Instruct for Judge.

Outcome:

- only one main model loaded at a time,
- safe hot-swapping,
- predictable RAM usage.

---

## Step 5: Build Planner and DAG Engine

Build:

- planner prompt,
- DAG schema,
- DAG validator,
- node splitter,
- plan storage.

Outcome:

- planner outputs strict DAG,
- invalid DAG is rejected,
- cycles are blocked,
- large tasks can be split safely.

---

## Step 6: Build Context Compiler

Build:

- token budgeting,
- sliding window,
- dependency contracts,
- Knowledge Brief injection,
- error tail injection.

Outcome:

- coding model receives only relevant context,
- old nodes do not overflow prompt,
- dependency interfaces are preserved.

---

## Step 7: Build Sandbox and TDD Loop

Build:

- bubblewrap runner,
- sandbox policy,
- timeout enforcement,
- stdout/stderr capture,
- language validators,
- test generation loop,
- code repair loop.

Outcome:

- code runs safely,
- network is blocked by default,
- valid test exists before code repair,
- exactly 3 code-fix attempts are allowed,
- sandbox violations fail immediately.

---

## Step 8: Build RAG-Based Repair

Build:

- error message extraction,
- Error Pack search,
- known-fix retrieval,
- repair hint injection.

Outcome:

- retries are smarter,
- fewer repeated failures,
- lower CPU/token usage.

---

## Step 9: Build Optional Dense RAG

Build only after sparse RAG works.

Build:

- embedding ingestion,
- INT8 vector storage,
- similarity search,
- embedding model load/unload.

Models/tools:

- `bge-small-en-v1.5` or `all-MiniLM-L6-v2`,
- ONNX Runtime INT8,
- sqlite-vec or FAISS.

Outcome:

- better semantic retrieval,
- still low-resource because model is optional and unloaded.

---

## Step 10: Build Final Output and Failure Reporting

Build:

- final formatter,
- failure formatter,
- uncertainty markers,
- source list,
- artifact list.

Outcome:

- user receives answer,
- user receives evidence,
- user receives failure reason if system stops.

---

# Expected Outcomes

## User Outcome

User receives:

- final answer,
- generated code or research result,
- test status,
- sources,
- confidence level,
- failure reason if incomplete.

---

## System Outcome

System should:

- stay within 4–5GB RAM,
- avoid Docker,
- avoid always-loaded large models,
- avoid context overflow,
- avoid infinite retries,
- work offline for local RAG,
- use online search only when needed.

---

## Quality Outcome

System should produce:

- fewer hallucinations,
- fewer code retries,
- better API usage,
- better factual grounding,
- smaller prompts,
- cleaner failures.

---

# Final Target

A low-resource Linux Harness with tiered RAG: cache first, local sparse RAG second, optional local dense RAG third, online RAG last. It uses small hot-swapped Qwen models, strict DAG planning, dependency-aware context compression, sandboxed TDD, and resource-aware splitting to produce validated results on 4–5GB RAM.
