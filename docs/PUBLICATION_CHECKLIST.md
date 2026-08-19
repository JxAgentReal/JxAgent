# Publication Checklist

Before pushing this repository publicly:

- [ ] Run `python -m pytest tests/ -q` in a clean environment.
- [ ] Run `python tools/verify_public_repo.py .`.
- [ ] Confirm no `.env`, tokens, model weights, datasets, screenshots, trajectories, or checkpoints are tracked.
- [ ] Confirm no machine-local absolute paths are present.
- [ ] Confirm the README still distinguishes planned experiments from completed results.
- [ ] Confirm upstream dataset/model licenses have not changed.
- [ ] Create the GitHub repository as `JxAgentReal/JxAgent`.
- [ ] Use the repository description: `Regression-aware post-training for reliable long-horizon computer-use agents.`
- [ ] Add topics: `computer-use`, `multimodal`, `qwen`, `agents`, `post-training`, `lora`, `osworld`, `rocm`, `research`.
- [ ] Do not upload generated datasets or model checkpoints into Git history.
