# Lambda / NVIDIA Cloud Path

This directory contains the provider-facing runbook for reproducing JxAgent experiments on a Lambda NVIDIA GPU instance.

The design principle is intentionally conservative: use the provider's working CUDA/PyTorch image when possible, verify it, and avoid replacing the driver stack during a grant-funded run.

## 1. Clone and preflight

```bash
git clone https://github.com/JxAgentReal/JxAgent.git
cd JxAgent
bash cloud/lambda_preflight.sh
```

The preflight is read-only and checks NVIDIA visibility, PyTorch CUDA access, GPU count, BF16 execution, and disk capacity. It does not consume substantial GPU time.

## 2. Create the Python environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest tests/test_second_stage_hardening.py tests/test_training_infrastructure.py -q
```

Install the exact trainer version used for the experiment separately and record it in the run manifest. Do not silently change the provider CUDA stack merely to satisfy an optional acceleration package.

## 3. Model and dataset

The current public base target is `Qwen/Qwen3.8-27B`. Pin the exact Hugging Face model commit before a reportable experiment.

The dataset should be built and validated before expensive GPU training. Model weights and prepared datasets are not committed to this repository.

## 4. Native-interface gate

A reportable preproduction run requires a verified `jxagent_interface_manifest.json` derived from the actual downloaded model and official local interface evidence.

The 5,000+ sample build and the training preflight are intentionally fail-closed if that contract has not been verified.

## 5. Experiment sequence

```text
read-only cloud preflight
clean code checkout + tests
model download + revision pin
native-interface freeze
prepared dataset validation
3-step training smoke
100-step throughput estimate
5,000-sample preproduction run
paired checkpoint evaluation
only then consider production training
```

## 6. Grant accounting

For every grant-funded run record:

- instance/GPU type
- GPU count
- wall time
- model revision
- dataset manifest hash
- source commit
- trainer/framework versions
- precision and batch topology
- checkpoint identities
- evaluation task-set hash

This makes compute use auditable and lets the research report quality improvements against actual accelerator cost.
