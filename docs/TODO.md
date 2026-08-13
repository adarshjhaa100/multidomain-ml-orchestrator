# STAGE 1 - MULTIDOMAIN ORCHESTRATOR SKELETON

## **INPUT AND SYSTEM SPECIFICATION:**

- three domain definitions: EXACT SPECIFICATIONS on what type of functions I need to deliver: USE CRITICAL THINKING AND RESEARCH FORMAT LLMS ITERATIVELY AND REPEATEDLY
- target device list,
- non-goals,
- request schema,
- response schema,
- safety policy,
- energy metrics.
- FOR ALL PROMPTS: KEEP BREAKING IT DOWN INTO SMALLER TASKS UNTIL THE MODEL IS “CONFIDENTLY CAPABLE” with certain accuracy. What Do I aim out of 6 “optimizations” listed below
- Success Criteria of exactly the level we want to reach in each: Energy req, resource, accuracy etc.

**ORCHESTRATOR SKELETON:**

- CLI input,
- router rules,
- domain dispatcher,
- prompt breaker ( top down approach )
- config system,
- logging,
- timeout handling.
- See if we can build something or utilize some open source software for OUR USE CASE

**GENERIC HIGH LEVEL SUPERVISOR HARNESS:**

- Policies
- Guardrails
- Data filtering and redaction
- Output formatting
- Evaluator and repeated questioning with improvement
- Use open source harness if fits usecases above.

---

## DESCRIPTION OF ORCHESTRATOR:

The orchestrator is the execution environment.

It handles:

1. **Request intake**
    - text input,
    - file input,
    - mode selection ?
    - device profile,
    - privacy level,
    - energy budget.
2. **Normalization**
    - clean input,
    - detect language,
    - detect attachments,
    - detect code blocks,
    - detect URLs,
    - detect medical urgency keywords,
    - detect PII.
3. **Policy engine**
    - allowed domains,
    - forbidden actions,
    - max latency,
    - max tokens,
    - max memory,
    - offline-only mode,
    - network permission.
4. **Router**
    - classify intent,
    - choose harness,
    - ask clarification if uncertain.
5. **Execution manager**
    - run harness,
    - enforce timeout,
    - enforce retry limit,
    - enforce energy budget.
6. **Telemetry**
    - latency,
    - energy estimate,
    - memory,
    - model used,
    - validation result,
    - failure reason.
7. **Audit log**
    - request hash,
    - router decision,
    - model version,
    - harness version,
    - output validation result.

Do not send raw user data to telemetry by default. If you log, redact.

## 

> Energy ≈ data moved × cost of moving data + computation × cost of computation
> 

---

**AT THE END OF EACH**

Produce:

- energy per task,
- accuracy per domain,
- failure modes,
- model comparison,
- next iteration plan.

Correct order - Do this:

1. Define request/response schema.
2. Build deterministic router.
3. Build policy engine.
4. Build telemetry.
5. Build coding harness.
6. Build medical retrieval harness.
7. Build research retrieval harness.
8. Add small models.
9. Add validation.
10. Add energy profiling.
11. Add fallback modes.
12. Add user-selected domain override.
13. Only later add translation/speech/image/video.
