# autoresearch

This is an experiment to have the LLM do its own research.

Current target platform: **single NVIDIA A100 80GB**.
The checked-in `train.py` baseline is already tuned as a conservative starting point for this machine.

## Resume Notes (April 2, 2026)

- Best local result in `results.tsv` so far: commit `0610fb5`, `val_bpb 1.029759`, `memory_gb 19.7`.
- The checked-in `train.py` should now be treated as the new A100 baseline before doing any further tuning.
- Consult the local `results.tsv` for the full experiment history, but do not commit it.

What worked today:

- Switching intermediate layers to local attention was the biggest win. The current code uses `WINDOW_PATTERN = "S"`, keeps the final layer global, and sets the short window to `sequence_len // 8`.
- Reducing batch size was also a major win. The best run uses `TOTAL_BATCH_SIZE = 2**17` and `DEVICE_BATCH_SIZE = 64`.
- A slightly higher Muon matrix LR helped: `MATRIX_LR = 0.032`.
- Longer decay helped on the small-batch setup. The current best uses `WARMDOWN_RATIO = 0.8` and `FINAL_LR_FRAC = 0.05`.

What did not work today:

- `DEPTH = 8`
- 576d width with 64d heads
- 8 attention heads at 512d
- Shrinking the short window further to `sequence_len // 16`
- Reducing batch again to `TOTAL_BATCH_SIZE = 2**16` and `DEVICE_BATCH_SIZE = 32`
- Lowering `FINAL_LR_FRAC` further to `0.02`
- Increasing `WARMDOWN_RATIO` further to `0.9`

Good next directions:

- Search locally around the current best setup instead of retesting the losing directions above.
- Good nearby knobs are `WARMUP_RATIO`, `EMBEDDING_LR`, `UNEMBEDDING_LR`, `WEIGHT_DECAY`, and small variants around the `sequence_len // 8` short window.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains a tokenizer and at least **64 training shards** plus the pinned validation shard. If not, run `uv run prepare.py --num-shards 64` yourself. Only ask the human if you are blocked by permissions, missing tools, or network access.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row if it does not already exist. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a **single A100 80GB GPU**. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

The current checked-in baseline in `train.py` is the machine-specific starting point for this repo:
- `DEPTH = 7`
- `DEVICE_BATCH_SIZE = 64`
- `TOTAL_BATCH_SIZE = 2**17`
- `WINDOW_PATTERN = "S"`

The current checked-in code also uses a short local window of `sequence_len // 8` for intermediate layers and forces the final layer to full attention.

Do **not** change any of these before the first baseline run. Establish the baseline exactly as checked in, then iterate from there.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script exactly as is.

## Output format

Once the script finishes it prints a summary like this (values are illustrative and depend on the exact machine/config):

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            6
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_bpb	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
