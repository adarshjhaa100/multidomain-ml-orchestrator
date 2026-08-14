## 1. Core System Goal

The system is an orchestrator and harness for low-resource devices.

Its job is to:

1. Receive a user request.
2. Filter and normalize the input.
3. Apply policy rules.
4. Estimate required compute.
5. Decide which model or tool is needed.
6. Break the task into small, testable subtasks.
7. Route work to the correct harness.
8. Validate the generated result.
9. If the result is wrong, repair the prompt and retry.
10. Format the final answer.
11. Log everything for observability.

The key requirement is that the system does not blindly trust model output. It must validate, correct, and retry automatically when possible.

---

## 2. Hardware First Principles

The target devices are lightweight.

Expected constraints:

- Around 4 CPU cores.
- Around 4GB RAM baseline.
- Possibly 5GB RAM if an extra judge model is allowed.
- CPU-first inference.
- Slow tokens per second is acceptable.
- Disk storage is available.
- Compute must be conserved by splitting tasks.

From these constraints, several facts follow.

### 2.1 Model size must stay small

Large models are not practical.

The working range is:

- 0.5B models for tiny helper tasks.
- 1.5B models for planning, research, and coding.
- Avoid 3B or larger unless RAM increases significantly.

### 2.2 Quantization is mandatory

Models should be quantized.

Recommended minimum:

- 4-bit quantization for practical reliability.
- 5-bit for the planner if RAM allows.
- Avoid 2-bit and be careful with 3-bit because small models lose too much reasoning ability.

Important clarification: 2-bit or 3-bit is more quantized, not less quantized.

### 2.3 Context length must be controlled

The context window consumes RAM through the key-value cache.

For a 1.5B model, a rough estimate is:

- 2048 tokens uses about 450MB of key-value cache.
- 4096 tokens uses about 900MB.
- 8192 tokens uses about 1.8GB.

Therefore, the system should use a small controlled context window.

Recommended defaults:

- 2048 tokens for 4GB RAM.
- 3072 tokens for 5GB RAM.
- 4096 tokens only if RAM headroom is confirmed.

---

## 3. Main Architectural Rule

The persistent orchestrator should not be a large always-running model.

The persistent orchestrator should be a deterministic controller.

The models are tools used by the orchestrator.

The controller handles:

- state,
- scheduling,
- retries,
- resource checks,
- sandbox execution,
- disk storage,
- validation,
- observability.

The models handle:

- planning,
- research summarization,
- code generation,
- code repair,
- optional judging.

This keeps RAM usage predictable.

---

## 4. Model Roles

### 4.1 Persistent Orchestrator

Role:

- Control loop.
- Resource monitor.
- Task queue.
- State machine.
- Model loader and unloader.
- Sandbox runner.
- Context builder.
- Output formatter.

This should be deterministic and lightweight.

It should not be a 1.5B model permanently loaded in RAM.

### 4.2 Planner / Research / Bifurcation Model

Recommended model family:

- Qwen2.5-1.5B-Instruct.

Purpose:

- Understand normal English queries.
- Perform encyclopedia-style research.
- Summarize retrieved knowledge.
- Break problems into small subtasks.
- Produce a strict task graph.
- Decide when a query must be divided further.

Quantization:

- Prefer 4-bit minimum.
- Prefer 5-bit if RAM allows.

This model should be hot-swapped.

It should not remain loaded if the coding model is needed and RAM is tight.

### 4.3 Coding Model

Recommended model family:

- Qwen2.5-Coder-1.5B-Instruct.

Purpose:

- Generate code in C, Python, JavaScript, HTML, and CSS.
- Generate tests.
- Repair failing code.
- Follow test-driven development.
- Respect strict sandbox constraints.
- Produce small, explicit, testable units.

Quantization:

- 4-bit is the practical default.

This model should also be hot-swapped.

### 4.4 Optional Judge or Policy Model

Recommended model family:

- Qwen2.5-0.5B-Instruct.

Purpose:

- Lightweight judgment.
- Policy triage.
- Simple validation.
- Summarizing failures.
- Checking whether an answer appears inconsistent.

However, deterministic validation should be preferred wherever possible.

A judge model should not be the primary truth source.

### 4.5 Optional Retrieval or Ranking Model

A small reranker or embedding model may be used for fact validation.

Purpose:

- Rank search snippets.
- Compare claims against retrieved evidence.
- Improve factual grounding.

This should be small and optional.

### 4.6 Tiny 100M to 150M Models

A tiny model can be used only for narrow classification tasks.

Examples:

- intent detection,
- language detection,
- toxicity triage,
- prompt injection triage,
- simple routing.

A 135M model is not suitable for planning, system design, or complex decomposition.

---

## 5. Hot-Swapping Strategy

You chose hot-swapping instead of keeping all models loaded.

This is correct for low RAM.

The flow is:

1. Load planner model.
2. Generate plan and knowledge brief.
3. Save results to disk.
4. Unload planner model.
5. Load coding model.
6. Execute the current task.
7. Save artifacts.
8. Retry if needed.
9. Move to next task.
10. Repeat until done.

### 5.1 Model weight caching

Model weights can be cached by the operating system or read from disk using memory mapping.

This can make reloading faster.

However, if RAM is under pressure, the system may still need to read weights from disk again.

### 5.2 Key-value cache

The key-value cache represents the model’s current context memory.

It can be written to disk if the same model is paused and later resumed.

However, the key-value cache cannot be transferred between different models.

For example, the research model’s context state cannot be handed directly to the coding model.

The correct method is:

- save the text output,
- save the knowledge brief,
- save the plan,
- load the next model,
- give the next model a fresh prompt built from saved artifacts.

### 5.3 Interaction between models

Continuous direct interaction between models is expensive because it requires repeated swapping.

The preferred approach is:

- precompute a knowledge context packet,
- store it on disk,
- let the coding model read it,
- only hot-swap back to the research model for exceptional cases.

---

## 6. Research and Fact Validation

The research model should not rely only on memorized knowledge.

It should use retrieval and validation.

### 6.1 Knowledge sources

Possible sources:

- Wikipedia.
- Structured Wikipedia dumps.
- Encyclopedia chunks.
- Literature chunks.
- SearXNG search results.
- Local offline index.

For a low-resource system, a local keyword index is often better than heavy dense-vector search.

Recommended first approach:

- chunked text,
- keyword search,
- BM25-style ranking,
- simple reranking,
- short knowledge brief.

### 6.2 Knowledge context vector

You mentioned a knowledge context vector.

For this architecture, the practical version should be a knowledge context packet or knowledge brief.

It should be text stored on disk, not a raw embedding vector.

Reason:

- the coding model cannot directly consume a raw embedding vector,
- the model consumes tokens,
- text is inspectable,
- text can be truncated,
- text can be validated,
- text can be cached.

### 6.3 Knowledge brief contents

A knowledge brief should include:

- original query,
- short facts,
- source references,
- confidence level,
- relevance to specific tasks,
- warnings or uncertainties.

It should be small enough to fit into the context window.

### 6.4 Hallucination validation

The harness should not ask the same model to judge itself without external evidence.

Preferred validation order:

1. Extract atomic claims from the generated research text.
2. Retrieve evidence from search or local index.
3. Compare claims with retrieved evidence.
4. Use a small reranker to score relevance.
5. Optionally use a tiny judge model for final triage.
6. Mark facts as accepted, rejected, or low confidence.

If a fact is low confidence:

- do not use it for planning,
- or trigger another research task,
- or mark the final answer as uncertain.

### 6.5 Knowledge graphs

Knowledge graphs are useful but heavy.

A full knowledge graph requires:

- entity extraction,
- relation extraction,
- triple storage,
- graph querying,
- graph validation.

For a 4GB to 5GB device, this should be deferred.

The first version should use:

- atomic claims,
- retrieval,
- text briefs,
- simple confidence scoring.

A lightweight triple store can be added later.

---

## 7. Coding Model Requirements

The coding model must support:

- C,
- Python,
- JavaScript,
- HTML,
- CSS.

It must work inside a harness with:

- sandboxing,
- test-driven development,
- retries,
- artifact storage,
- observability.

### 7.1 Planning style

The coding model should follow a Tiger Style-like philosophy.

Core principles:

- explicit state,
- explicit errors,
- no hidden control flow,
- no silent failure,
- minimal dependencies,
- small functions,
- testable units,
- deterministic behavior where possible,
- avoid unnecessary complexity.

### 7.2 Extreme bifurcation

The coding model should not be asked to solve a large task in one shot.

The task should be broken into the smallest possible testable units.

Example concept:

Original task:

- Build a program to count words in a document.

Broken into:

- Read file.
- Convert bytes to text.
- Normalize whitespace.
- Count tokens.
- Return structured result.
- Integrate components.
- Test end-to-end behavior.

Each node should be independently testable.

---

## 8. Task Graph / DAG Requirement

The planner must output a strict directed acyclic graph of tasks.

This is the central planning artifact.

The DAG must be validated before execution.

### 8.1 DAG top-level fields

The plan should contain:

- schema version,
- query identifier,
- objective,
- resource profile,
- policy constraints,
- knowledge artifacts,
- nodes.

### 8.2 Node fields

Each node should contain:

- identifier,
- title,
- kind,
- language,
- dependencies,
- inputs,
- outputs,
- acceptance criteria,
- context references,
- sandbox requirements,
- retry policy.

### 8.3 Node kinds

Allowed node kinds:

- research,
- code,
- test,
- integrate,
- validate,
- plan.

### 8.4 Supported languages

Allowed languages:

- python,
- c,
- javascript,
- html,
- css,
- shell,
- none.

### 8.5 Dependency rules

The DAG must have:

- no cycles,
- no missing dependencies,
- no duplicate node identifiers,
- no self-dependencies,
- no forward dependencies,
- no excessive number of nodes.

The deterministic orchestrator should validate these rules.

### 8.6 Plan immutability

The plan itself should be immutable once accepted.

Runtime state should be stored separately.

This prevents the model from confusing planning with execution state.

---

## 9. Runtime State

Runtime state should not live inside the model prompt.

It should live on disk, preferably in a lightweight database or structured state files.

The state should track:

- query identifier,
- node identifier,
- node status,
- current attempt number,
- start time,
- end time,
- exit result,
- standard output artifact path,
- standard error artifact path,
- error summary,
- resource usage,
- sandbox violations.

Possible node statuses:

- pending,
- ready,
- running,
- test failed,
- code failed,
- sandbox violation,
- timeout,
- failed,
- done.

The deterministic orchestrator updates the state.

The model only receives the information needed for the next step.

---

## 10. Resource-Aware Bifurcation

You want the system to split work when compute is scarce.

This is correct.

The orchestrator should monitor:

- CPU load,
- available RAM,
- disk pressure,
- queue depth,
- model load time,
- sandbox execution time.

### 10.1 Low-resource behavior

If resources are low, the orchestrator should do one or more of the following:

1. Reduce context window size.
2. Reduce knowledge brief size.
3. Reduce sliding window size.
4. Reduce output token budget.
5. Split the current node into smaller nodes.
6. Store unfinished work on disk.
7. Resume later.
8. Avoid loading optional judge models.

### 10.2 Splitting strategy

The system should prefer local splitting.

That means:

- do not re-plan the whole query unless necessary,
- split only the current node,
- preserve completed work,
- store the remaining work on disk,
- schedule the new smaller nodes.

### 10.3 Splitting limits

To prevent infinite splitting, the system needs hard limits.

Recommended defaults:

- maximum total nodes: 24,
- maximum split depth: 3,
- maximum replans per node: 2,
- maximum replans per query: 5.

If these limits are exceeded, the task should be marked failed or escalated.

---

## 11. Sliding Window Context Management

You chose a sliding window to avoid context overflow.

The sliding window must be dependency-aware.

### 11.1 What should be included

For the current coding step, include only:

- system and policy instructions,
- current node definition,
- short DAG topology summary,
- relevant knowledge brief,
- direct dependency contracts,
- summaries of the last one or two completed nodes,
- previous error summary if retrying,
- output format requirements.

### 11.2 What should not be included

Do not include:

- full code from all previous nodes,
- full stdout from all previous nodes,
- full search results,
- full conversation history,
- full DAG with every detail,
- irrelevant knowledge.

### 11.3 Dependency contracts

When a node depends on a previous node, include the interface, not the full implementation.

For code, the contract should include:

- function name,
- inputs,
- outputs,
- errors,
- expected behavior,
- artifact location.

For HTML, the contract should include:

- file path,
- important identifiers,
- expected structure,
- expected behavior.

For CSS, the contract should include:

- file path,
- important selectors,
- expected styling constraints.

### 11.4 Eviction policy

When the context is too large, remove information in this order:

1. Oldest non-dependency node summary.
2. Less relevant knowledge brief sections.
3. Middle portions of long logs.
4. Full implementation details, replaced by contracts.
5. Detailed DAG description, replaced by minimal topology.

Never remove:

- current node definition,
- direct dependency contracts,
- output instructions,
- sandbox constraints,
- previous error summary during retry.

### 11.5 Artifact compression

After each node completes, store full artifacts on disk.

But only pass compressed summaries into the next prompt.

Each node should have:

- full code,
- full test,
- stdout,
- stderr,
- result summary,
- contract summary,
- error summary.

The prompt uses summaries.

The disk keeps the full history.

---

## 12. Test-Driven Development Loop

The coding harness should use a test-driven loop.

### 12.1 Basic loop

For each node:

1. Build prompt from sliding window.
2. Ask coding model for test and implementation.
3. Write artifacts to workspace.
4. Run static validation.
5. Run sandboxed test.
6. If success, mark node done.
7. If failure, store error summary.
8. Retry with repaired prompt.
9. Stop after maximum attempts.

### 12.2 Attempt limit

You chose three attempts.

The simplest rule is:

- three total attempts per node.

An attempt can include:

- test generation,
- code generation,
- syntax validation,
- sandbox execution,
- test result evaluation.

A sandbox violation should be treated specially.

Recommended rule:

- sandbox violation causes immediate failure,
- no automatic retry unless explicitly allowed by policy.

### 12.3 Failure handling

If a node fails after three attempts:

- mark the node failed,
- store all artifacts,
- record the error summary,
- stop automatic retries,
- optionally trigger a replan,
- optionally request human review.

---

## 13. Sandbox Design

You chose lightweight OS-level sandboxing.

The preferred tool category is:

- bubblewrap,
- nsjail,
- firejail.

The best fit for ephemeral code execution is usually bubblewrap or nsjail.

Docker and Podman are too heavy for this RAM budget unless much more RAM is available.

### 13.1 Sandbox defaults

Default sandbox policy:

- no network,
- write only to a temporary workspace,
- read only approved paths,
- strict timeout,
- limited output size,
- limited file write size,
- no subprocess spawning unless required,
- no environment variable leakage,
- kill process on timeout.

### 13.2 Network policy

Default:

- network disabled.

If research needs network:

- the orchestrator does the retrieval,
- the coding sandbox still does not get direct network access unless required.

### 13.3 Language-specific validation

For Python:

- syntax check,
- test execution,
- structured result.

For C:

- compile with warnings treated seriously,
- run test binary,
- capture exit code.

For JavaScript:

- syntax check,
- run test file,
- capture output.

For HTML:

- parse structure,
- validate required elements,
- validate identifiers and classes.

For CSS:

- validate syntax,
- validate expected selectors,
- avoid heavy browser rendering.

A headless browser is likely too heavy for 4GB RAM.

---

## 14. Policy Layer

The system needs policy checks at input, planning, execution, and output.

### 14.1 Input policy

Input policy should check:

- unsafe requests,
- prompt injection,
- prohibited content,
- sensitive personal data,
- secret leakage,
- unsupported task types.

### 14.2 Planning policy

Planning policy should check:

- DAG size,
- allowed languages,
- allowed node kinds,
- allowed sandbox capabilities,
- network requests,
- file access requests,
- resource limits.

### 14.3 Execution policy

Execution policy should enforce:

- sandbox isolation,
- timeouts,
- output limits,
- file write limits,
- no unauthorized network,
- no dangerous commands,
- no recursive process spawning.

### 14.4 Output policy

Output policy should check:

- final answer format,
- leaked secrets,
- personal data,
- unsafe instructions,
- failed validation flags,
- uncertainty markers.

---

## 15. Observability

Observability must be built in from the start.

The system should record events, metrics, and traces.

### 15.1 Events

Important events:

- query received,
- policy accepted,
- policy rejected,
- planner loaded,
- planner unloaded,
- plan generated,
- plan rejected,
- plan accepted,
- node ready,
- node started,
- attempt started,
- sandbox started,
- sandbox finished,
- sandbox timeout,
- sandbox violation,
- test failed,
- node done,
- node failed,
- resource threshold crossed,
- node split,
- knowledge brief updated,
- final output produced.

### 15.2 Metrics

Important metrics:

- model load time,
- model unload time,
- tokens in,
- tokens out,
- tokens per second,
- prompt building time,
- context size,
- key-value cache estimate,
- CPU usage,
- RAM usage,
- sandbox runtime,
- retry count,
- failure count,
- schema validation failure count,
- hot-swap count.

### 15.3 Tracing

Every action should be linked to:

- query identifier,
- node identifier,
- attempt number,
- model role,
- artifact identifier.

This makes debugging possible.

---

## 16. Model and Memory Budget Summary

Assuming a 5GB usable budget:

### Persistent layer

- Deterministic orchestrator: small memory footprint.
- State database: small.
- Observability buffer: small.

### Active model layer

Only one main model should be active at a time.

Typical active model:

- 1.5B planner or coder at 4-bit or 5-bit quantization.

Approximate active memory:

- model weights: around 1.1GB to 1.4GB,
- context cache: around 450MB to 900MB depending on context length,
- sandbox process: variable but should be limited,
- optional small validator: around 300MB to 600MB if loaded.

### Practical rule

Do not keep both planner and coder loaded unless RAM is comfortably above the budget.

Use hot-swapping.

Use disk artifacts for communication.

---

## 17. Recommended Defaults

The current recommended defaults are:

- Persistent orchestrator: deterministic controller.
- Planner model: 1.5B instruct model.
- Coding model: 1.5B coding model.
- Judge model: optional 0.5B model.
- Fact validation: retrieval plus small reranker plus atomic claims.
- Knowledge graph: deferred.
- Research context: text knowledge brief stored on disk.
- State storage: structured database or structured files.
- Artifact storage: filesystem.
- Context window: 3072 tokens if 5GB RAM is available, otherwise 2048.
- Sliding window: dependency-aware.
- Retry limit: three attempts per node.
- Sandbox: lightweight Linux namespace sandbox.
- Default network: disabled.
- Maximum nodes: 24.
- Maximum split depth: 3.
- Maximum replans per node: 2.
- Maximum replans per query: 5.
- HTML and CSS validation: static and structural validation, no headless browser in first version.

---

## 18. Remaining Open Questions

These are the remaining unclear points that still need decisions.

### Question 1: Operating system

Are the target devices running Linux?

Lightweight namespace sandboxing depends on Linux.

If the devices run Windows, macOS, Android, or something else, the sandbox plan must change.

---

### Question 2: Context window target

Should the default context window be:

- 2048 tokens,
- 3072 tokens,
- 4096 tokens?

The choice affects RAM usage and answer quality.

Current recommendation:

- 2048 for 4GB RAM,
- 3072 for 5GB RAM.

---

### Question 3: Available runtimes

Which runtimes are installed on the device?

Need to know availability of:

- Python,
- GCC or Clang,
- Node.js,
- HTML validators,
- CSS validators,
- shell tools.

This affects how tests are executed.

---

### Question 4: Internet availability

Does the device have reliable internet access?

If yes:

- SearXNG can be used.
- Online Wikipedia can be used.

If no:

- a local Wikipedia or literature index is required.
- keyword search is preferred over heavy vector search.

---

### Question 5: Knowledge graph requirement

Do you need a knowledge graph in the first version?

A knowledge graph is powerful but heavy.

For low-resource devices, the first version should probably use:

- atomic claims,
- retrieval,
- evidence scoring,
- text briefs.

The knowledge graph can be added later.

---

### Question 6: Exact meaning of three attempts

The three-attempt limit needs one precise definition.

Option A:

- Three total attempts for everything, including test generation, syntax errors, code generation, and runtime failures.

Option B:

- Three implementation attempts after a valid test has been generated, with a separate small budget for test repair.

Option A is simpler and safer.

Option B is more flexible but more complex.

---

### Question 7: Maximum DAG limits

Need final limits for:

- maximum total nodes,
- maximum split depth,
- maximum replans per node,
- maximum replans per query.

Recommended defaults:

- 24 nodes,
- depth 3,
- 2 replans per node,
- 5 replans per query.

---

### Question 8: HTML and CSS validation method

HTML and CSS are harder to test than Python, C, or JavaScript.

Options:

1. Static linting and structural checks only.
2. Static linting plus simple document structure assertions.
3. Headless browser validation.

For 4GB RAM, option 3 is too heavy for the first version.

Recommended:

- static linting plus structural assertions.

---

### Question 9: State storage choice

Preferred choice:

- structured database for state,
- filesystem for artifacts.

Alternative:

- pure file-based state.

Structured database is better for crash recovery, querying, and observability.

---

### Question 10: Mandatory policies

Need to define which policies are mandatory.

Possible policies:

- personally identifiable information redaction,
- secret detection,
- malicious code blocking,
- network blocking,
- file path restriction,
- copyright filtering,
- unsafe command blocking,
- prompt injection filtering.

The first version needs a clear minimum policy set.

---

## 19. Final Planning Position

The system should be built as a deterministic, disk-backed, resource-aware orchestration layer around small hot-swapped models.

The planner model creates a strict task graph and a knowledge brief.

The coding model executes the task graph using test-driven development inside a lightweight sandbox.

The orchestrator manages retries, context compression, resource limits, and observability.

Validation should rely on retrieval, evidence comparison, deterministic checks, and sandbox execution, not only on another language model.

The most important design constraints are:

- keep models small,
- keep context small,
- keep state on disk,
- split tasks aggressively,
- validate every node,
- limit retries,
- avoid heavy containers,
- avoid always-loaded large models,
- log everything.
