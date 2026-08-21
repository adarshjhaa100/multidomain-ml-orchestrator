Across development of this project, we're to test fuzzily, benchmark and profile the results

# STAGE 1 - MULTIDOMAIN ORCHESTRATOR SKELETON

## Step 0:
  Prepare data

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
