# Dataset Card: JxAgent Run 1

## Summary

JxAgent Run 1 is a planned **95,503-sample** multimodal post-training mixture for computer-use reliability. The repository provides a builder and selection pipeline. It does not redistribute the underlying source datasets.

## Intended use

Research on post-training multimodal computer-use models, especially recovery, verification, grounding, long-horizon reliability, and regression-aware specialization.

## Composition

| Source | Target | Role |
| --- | ---: | --- |
| nvidia/ProCUA-SFT | 46,000 | action trajectories |
| cua-lite/GUI-360 | 16,000 | use, grounding, understanding |
| ServiceNow/VideoCUA | 17,500 | temporal desktop behavior |
| ServiceNow/GroundCUA | 4,000 | grounding |
| henryhe0123/PC-Agent-E | 4,503 | desktop trajectories |
| replay mixture | 7,500 | capability preservation |

Replay targets:

- coding: 1,600
- math: 1,500
- general instruction: 1,700
- VQA: 1,400
- tool style: 1,300

## Deliberate exclusions

The Run 1 training mixture excludes OSWorld trajectories, OSWorld Verified evaluation examples, ScreenSpot Pro training data, CUA-Gym SFT, Click100k, UI-Vision, AgentNet, and the original giant GUI-360 archive.

## Quality controls

- source-specific coordinate parsing
- final-image coordinate normalization
- out-of-bounds rejection
- bounding-box validation where trusted target rectangles exist
- evidence-gated finish targets
- duplicate-referent filtering for grounding
- context-aware perceptual deduplication
- real-transition recovery supervision
- group-aware semantic split
- OSWorld instruction contamination check
- best-valid global selection
- task-motif coverage floors

## Reasoning supervision

Run 1 uses a synthetic reasoning rate of **0**. The goal is to teach observable action policy and recovery behavior without distilling fabricated hidden reasoning.

## Release gates

A production-sized build is not considered releasable until:

1. the model-native interface contract is verified from local model/scaffold evidence;
2. exact trainer labels and supervised-token shares are measured;
3. train/validation semantic groups are disjoint;
4. decontamination passes;
5. motif coverage gates pass;
6. the final dataset can be clean-downloaded and independently validated.

## Limitations

The mixture is not a claim of optimal source weighting. Source quality varies, coverage of rare stateful interactions is limited, and the final value of each cohort must be established experimentally. Sample counts are not equivalent to gradient contribution, so exact supervised-token accounting is part of the preproduction plan.
