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
