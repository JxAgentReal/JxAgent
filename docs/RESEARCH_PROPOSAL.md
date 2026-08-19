# JxAgent Research Proposal

## Regression-aware post-training for reliable long-horizon computer use

### Abstract

JxAgent studies whether targeted multimodal post-training can improve the difficult remaining failure modes of a strong computer-use model without causing measurable regression in capabilities the base model already performs well. The training target is deliberately narrow: recovery after ineffective actions, verification before completion, loop avoidance, dialogs and modal windows, save/export workflows, exact-quantity tasks, sorting/ranking, multi-target tasks, cross-application workflows, and precise GUI grounding.

The project combines a hardened data pipeline with paired base-versus-adapter evaluation. Rather than maximizing a single computer-use score, it seeks the best checkpoint on a frontier between computer-use improvement, preserved general capability, and compute cost.

### Primary hypothesis

A small, carefully selected specialization dataset and regression-aware checkpoint selection can improve long-horizon computer-use reliability more safely than broad task-domain SFT that replaces a large fraction of the base distribution.

### Research questions

1. Which failure-targeted data cohorts produce the largest improvement per supervised token?
2. How early during LoRA training do measurable preservation regressions appear?
3. Does explicit recovery supervision reduce repeated-action loops and unrecovered errors?
4. Does evidence-gated finish supervision improve task completion calibration without creating finish aversion?
5. Which LoRA module families produce the best computer-use gain for the least general-capability drift?
6. How much preservation replay is required once contribution is measured by actual supervised loss tokens rather than sample count?
7. Can checkpoint selection on a paired preservation frontier outperform selecting the final training checkpoint by default?

### Experimental design

The initial experiment uses a 95,503-sample multimodal mixture. The largest component remains native computer-use action supervision. Auxiliary understanding, grounding, and replay cohorts are constrained so that the specialization objective remains dominant.

The experiment has three stages:

1. **Preproduction:** 5,000 selected samples, all important checkpoints retained.
2. **Regression mapping:** evaluate base and each checkpoint on identical frozen samples.
3. **Production:** only proceed if the preproduction run demonstrates useful computer-use movement without unacceptable preservation loss.

### Evaluation dimensions

Computer-use:

- action validity
- coordinate validity
- grounding accuracy
- end-to-end task success
- recovery after ineffective actions
- loop rate
- premature finishing
- verification behavior
- cross-application reliability

Preservation:

- instruction following
- coding
- math and scientific reasoning
- multilingual behavior
- native tool use
- document and OCR understanding
- chart understanding
- visual math
- natural-image reasoning
- safety/refusal calibration
- ordinary chat style
- long-context retrieval and instruction retention

Systems:

- training throughput
- accelerator memory
- inference latency
- serving throughput
- compute per successful task

### Reproducibility

Every reportable run should record the exact base-model revision, processor revision, source-code commit, native-interface contract hash, dataset manifest, split IDs, seed, trainer command, software versions, accelerator type, precision, and checkpoint identity.

### Publication policy

The project will publish validated positive and negative results. Infrastructure tests are never presented as model-quality benchmarks. Benchmark evaluation data remains held out from training.
