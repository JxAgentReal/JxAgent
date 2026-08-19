# Model Compatibility

## Current public target

JxAgent is configured for `Qwen/Qwen3.8-27B`. The project treats the exact model and processor revision as part of the experiment contract rather than relying on a floating model name alone.

## Compatibility policy

Before any reportable training run:

1. download and pin the exact Qwen3.8-27B revision;
2. record the processor revision and software stack;
3. inspect the actual architecture and LoRA candidate modules;
4. derive the native computer-use and tool interface from the pinned local files;
5. freeze and hash the executable interface contract;
6. run golden encode, parse, label-mask, and generation-prefix tests;
7. run the 5,000-sample preproduction stage before any full production run.

JxAgent intentionally does not copy action grammar, chat templates, thinking defaults, or processor behavior from older Qwen releases. The pinned Qwen3.8 artifacts are the source of truth.

## Revision changes

Changing the base model revision, processor revision, chat template, native action protocol, or thinking policy invalidates the previous interface freeze and requires the compatibility gates to be rerun.
