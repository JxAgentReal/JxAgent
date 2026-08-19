# Third-Party Data and Models

JxAgent does not redistribute upstream datasets or model weights. The builder accesses upstream sources and creates a derived local training set. Users are responsible for reviewing the current upstream terms before use or redistribution.

## Primary computer-use sources

| Source | Upstream license noted by the project | Notes |
| --- | --- | --- |
| `nvidia/ProCUA-SFT` | CC BY 4.0 | attribution required |
| `cua-lite/GUI-360` | wrapper metadata says `other`; dataset card points to the original `vyokky/GUI-360` MIT license | verify both wrapper and origin before redistribution |
| `ServiceNow/VideoCUA` | MIT | upstream dataset license |
| `ServiceNow/GroundCUA` | MIT | upstream dataset license |
| `henryhe0123/PC-Agent-E` | MIT in the current project audit | verify current upstream card before redistribution |

## Replay sources

The current configuration draws small preservation cohorts from Magicoder Evol Instruct, Orca Math, SmolTalk, selected The Cauldron VQA subsets, and Hermes Function Calling. Each upstream source/subset may have its own license or attribution requirements. The builder does not change those rights.

## Benchmark separation

OSWorld and OSWorld Verified are evaluation references, not Run 1 training sources. Their instructions may be used for contamination detection. Do not copy evaluation trajectories into training data.

## Base model

The current public default is `Qwen/Qwen3.8-27B`, whose official repository states Apache 2.0. Always verify the exact model revision and license before training or redistribution of derived weights.
