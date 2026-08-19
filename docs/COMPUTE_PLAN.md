# Compute Plan

## Purpose

Cloud GPU resources are requested for controlled training, ablation, regression mapping, and reproducibility. They are not requested merely to host an application.

## Phase 1: environment and reproducibility, 10%

- verify the exact base-model revision and native interaction format
- benchmark the intended accelerator/software stack
- run the 3-step training smoke test
- run a bounded throughput test
- verify exact checkpoint reload and LoRA-only updates

## Phase 2: 5,000-sample preproduction, 25%

- train one guarded LoRA run
- preserve checkpoints at multiple training-progress points
- measure exact supervised-token shares
- evaluate paired base-versus-checkpoint preservation deltas

## Phase 3: targeted ablations, 40%

Prioritize experiments that answer a concrete research question:

- recovery cohort ablation
- verification/finish cohort ablation
- preservation replay contribution
- LoRA target-family ablation
- alternate rank or learning-rate run only if preproduction indicates it is necessary

## Phase 4: final reproducibility and evaluation, 25%

- reproduce the strongest safe configuration
- run untouched held-out computer-use evaluation
- run frozen preservation canaries
- collect serving throughput and memory measurements
- publish result tables with configuration hashes

## Requested Lambda use

A Lambda Research Grant would primarily fund NVIDIA GPU training and repeated evaluation. The research benefits from on-demand access because the key requirement is not one very long run; it is a set of controlled paired experiments across checkpoints and ablations.

If awarded less than the maximum grant, the experiment contracts naturally by dropping lower-priority ablations while retaining the 5,000-sample preproduction run and paired regression evaluation.

## AMD path

The repository also contains an AMD MI300X/ROCm workflow. Cross-accelerator experiments can report PyTorch/ROCm compatibility, HBM utilization, training throughput, and reproducibility while preserving identical dataset and evaluation contracts.
