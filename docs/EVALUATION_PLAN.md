# Evaluation Plan

## Principle

Every base-versus-adapter comparison uses the same task IDs, preprocessing, generation protocol, tool schema, and environment configuration.

## Stage A: infrastructure checks

- output syntax
- coordinate validity
- image preprocessing parity
- checkpoint reload
- deterministic manifest generation

These checks do not count as benchmark results.

## Stage B: preproduction checkpoint sweep

Evaluate checkpoints across training progress rather than only the final checkpoint. For each checkpoint record:

- computer-use task success
- recovery success
- loop rate
- finish calibration
- invalid actions
- paired preservation deltas
- latency and memory

## Stage C: held-out external evaluation

Run OSWorld/OSWorld Verified only after training data and checkpoint selection rules are frozen. Evaluation tasks must not be added to the training set.

## Regression canary policy

A regression canary is frozen before the run and compared sample-by-sample. If a capability regresses, first diagnose the responsible training cohort or module family. Do not reflexively add large amounts of replay data.

## Reporting

Every table should include:

- base model revision
- adapter checkpoint
- dataset manifest hash
- code commit
- evaluation task-set hash
- generation configuration
- sample count
- point estimate and paired uncertainty where applicable
