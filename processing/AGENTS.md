# MacBook worker handoff

The user assigned a camera batch. Follow README.md from this folder.

- Use `uv sync --frozen --python 3.11` and the frozen `configs/distributed_job.json`.
- Run only the assigned batch and the user's worker handle. If neither was provided, ask which batch; do not guess and duplicate another laptop's work.
- Use `scripts/distributed_worker.py run`; no road polygon, reviewer form, published baseline release or historical model is needed.
- Preserve model/profile/quality/class settings and source checksums. Fix environment/download/access issues without retuning the experiment. A code or scientific-profile change requires a new coordinator job.
- Resume the same command and checkpoint path after interruption. Never start two writers for the same batch.
- If an agent's restricted execution reports MPS unavailable on a supported Apple Silicon Mac, use normal Mac GPU access or Terminal for the same command. Do not silently switch to CPU or another model to work around a device-access restriction.
- A completed batch must pass packaging checks. Partial diagnostics and test-fixture results must not be uploaded as completed real data.
- The user-authorized return mechanism is a GitHub Release asset containing only the package command's ZIP. Use the central repository if authorized or the user's own fork otherwise. Never print tokens or upload raw images, model weights, virtual environments or checkpoints.
- Return the actual completed release link. Do not message teammates automatically.
- Import/merge commands on the coordinator do not run inference. Preserve source rows, explicit nulls/zeroes, quality flags and batch provenance.
