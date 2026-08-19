# Regression Guard

## Motivation

The main risk after data hardening is no longer obviously incorrect supervision. A narrow adapter can still shift a strong base model in undesirable ways. JxAgent therefore treats capability preservation as an explicit experimental variable.

## P0 guard work before production training

### 1. Executable native contract

Training encoding, evaluation encoding, action serialization, parsing, coordinate conversion, tool schema, history, system template, image placement, finish semantics, thinking policy, and generation prefix must come from one frozen contract.

### 2. Exact label accounting

Measure the real supervised labels produced by the installed training template. Token estimates are useful for prebuild planning but are not a release metric.

### 3. Two-sided gradient bands

Enforce both a minimum share for core computer-use actions and a maximum combined auxiliary share. Per-category caps alone can still allow the total auxiliary gradient to dominate.

### 4. Native replay tool format

Tool-preservation examples must use the base model's actual tool protocol instead of a foreign textual wrapper.

### 5. Explicit thinking policy

Training and evaluation must agree on thinking behavior and assistant generation prefixes.

### 6. Architecture-aware LoRA audit

Report trainable parameter mass and adapter update norms by module family. Recurrent-state or memory-gating projections should be treated as a distinct ablation target when the base architecture contains them.

### 7. Frozen preservation canaries

Maintain a small paired suite for general instructions, coding, math, multilingual output, native tool use, documents/OCR, charts, visual math, natural images, safety/refusal, style, and long context.

### 8. True paired regression statistics

Base and adapter must be evaluated on identical sample IDs. Aggregate scores without sample identity are insufficient for detecting small but systematic regressions.

### 9. Validation distribution parity

After group-aware splitting, compare source, task, motif, action, length, image count, and supervised-token distributions between train and validation.

### 10. Preserve preproduction checkpoints

Do not discard early checkpoints before regression evaluation. The best point may occur before the end of the epoch.

### 11. Serving parity

Measure the intended production serving stack for both base and adapter, including speculative/MTP behavior if used.

## Decision rule

The final checkpoint should be chosen from a Pareto frontier of:

- computer-use improvement
- preservation delta
- inference/serving cost

A later checkpoint is not automatically better.
