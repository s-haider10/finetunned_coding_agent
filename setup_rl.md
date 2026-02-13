# RL Post-Training Pipeline for Code Generation

**COSC 189.34 - AI Agents - Problem Set 1, Question 22**
Syed Ali Haider - Winter 2026

Tinker documentation: https://tinker-docs.thinkingmachines.ai/docs-outline
Tinker cookbook: https://github.com/thinking-machines-lab/tinker-cookbook

---

## 1. Quick-Start Guide

### Prerequisites

- Docker (for Sandbox Fusion code execution)
- `TINKER_API_KEY` set in the project root `.env` file (auto-loaded via `python-dotenv`)
- Python 3.13+ with dependencies from `requirements.txt`

### Setup Steps

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Start Sandbox Fusion** (code execution sandbox via Docker):

```bash
docker run -it -p 8080:8080 \
  -v ./sandbox_config/local.yaml:/root/sandbox/sandbox/configs/local.yaml \
  volcengine/sandbox-fusion:server-20250609
```

**Configure API key** in the project root `.env` file:

```
TINKER_API_KEY="your-api-key"
```

The key is auto-loaded at startup via `python-dotenv` -- no need to `export` it manually. Optional sandbox overrides can also go in `.env` or be exported:

```bash
# Optional overrides (defaults shown):
export SANDBOX_URL="http://localhost:8080/run_code"
export SANDBOX_MAX_CONCURRENCY="4"
```

**Run training:**

```bash
python train.py
```

The project uses `chz` for configuration. Override any config field via CLI using `key=value` syntax:

```bash
python train.py learning_rate=1e-5 max_steps=100 eval_every=5
```

**Monitor:** Logs are written to `~/code-rl-logs/YYYY_MM_DD-HH_MM_SS/` containing:

- `config.json` -- hyperparameters snapshot
- `metrics.jsonl` -- per-step metrics (append-only)
- `logs.log` -- full console log
- `checkpoints.jsonl` -- checkpoint metadata for resume

---

## 2. Codebase Map

```
finetunned_coding_agent/
  train.py                      # Main file -- 4 scaffolded functions + training loop
  setup_rl.md                   # This document
  requirements.txt              # Pinned deps (tinker 0.7.0, torch 2.9.1, etc.)
  .env                          # Environment config
  sandbox_config/
    local.yaml                  # Sandbox Fusion Docker config
  tinker_utils/
    env.py                      # CodeEnv class, sandbox execution, reward computation
    data.py                     # build_question() -- dataset example -> prompt string
    renderers.py                # Abstract Renderer + get_renderer() factory
    qwen.py                     # Qwen3InstructRenderer (the one we use)
    checkpoint.py               # save_checkpoint(), get_last_checkpoint()
    log.py                      # setup_logging() -> MultiplexLogger (JSON + Console + W&B)
    lcb.py                      # LiveCodeBench test harness, TEST_UTIL, TEST_CODE, prompts
    cli.py                      # check_log_dir() for log directory management
```

### Key Modules at a Glance

**`env.py`** -- The reward engine. `CodeEnv.step()` takes model-generated tokens, extracts Python code from fenced blocks (regex, takes the last block), validates format, and scores via Sandbox Fusion. The reward is `format_coef * format_score + correctness_score`. The sandbox client uses `aiohttp.TCPConnector` for connection pooling with built-in backpressure -- no manual semaphore needed.

**`qwen.py`** -- Three renderer variants. We use `Qwen3InstructRenderer` for the Instruct-2507 model. This renderer uses ChatML format (`<|im_start|>role...content...<|im_end|>`) but does NOT inject `<think>` blocks. It always satisfies the extension property, which matters for multi-turn RL efficiency.

**`data.py`** -- A single function `build_question(example)` that extracts the problem description and optional starter code, then wraps them in the LiveCodeBench system prompt format.

**`lcb.py`** -- Contains `TEST_UTIL` (~800 lines of robust test execution code including timeout handling, stdout capture, output matching) and `TEST_CODE` (the sandbox entry point script). Also contains `normalize_tests()` which converts various test formats (TACO, LCB) into a standard format, and the prompt templates `LCB_SYSTEM_MESSAGE_GENERIC` and formatting instructions.

**`renderers.py`** -- Abstract `Renderer` base class with `build_generation_prompt()` (turns messages into tokenized `ModelInput` for sampling) and `build_supervised_example()` (for training with loss masking). The factory function `get_renderer("qwen3_instruct", tokenizer)` returns our renderer.

**`checkpoint.py`** -- Saves/loads full training state (LoRA weights + Adam optimizer moments). Appends to `checkpoints.jsonl` with metadata including step number and dataset offset for seamless resume.

**`log.py`** -- `setup_logging()` creates a `MultiplexLogger` that forwards to JSON file + Rich console table + optional Weights & Biases. Each backend implements `log_hparams()` and `log_metrics()`.

---

## 3. GRPO Strategy: Critic-Free RL for Code

### Core Idea

Group Relative Policy Optimization (GRPO) is a critic-free approach to reinforcement learning for LLMs. The key insight: instead of training a separate value network (critic) to estimate baselines -- which doubles parameters and adds infrastructure complexity -- GRPO uses the empirical group mean as a free, unbiased baseline.

**How it works:**

1. For each coding problem, sample G completions from the current policy
2. Score each completion with a binary reward (does the code pass all tests?)
3. Compute each completion's advantage as its reward minus the group mean
4. Reinforce above-average completions; suppress below-average ones
5. The group mean is the baseline -- no learned critic required

The advantage for completion g is simply: `A_g = R_g - mean(R_1, R_2, ..., R_G)`

### Why Critic-Free Works Here

For binary correctness rewards (pass/fail), a learned critic would need to solve a trivially structured problem (predicting a near-constant value function) at massive infrastructure cost. GRPO gets equivalent variance reduction by asking "which of these G attempts worked?" -- a contrastive signal that requires zero extra parameters.

The group mean baseline automatically adapts to problem difficulty:

- Easy problems (high mean reward): small advantages, small gradient
- Hard problems (low mean reward): also small advantages, no gradient explosion
- Mixed results: large advantages, strong learning signal

This prevents easy problems from dominating updates -- a common failure mode in naive REINFORCE.

### Degenerate Groups

When all G completions get the same reward (all pass or all fail), every advantage is zero. There's no contrastive information -- no way to distinguish better from worse. These groups contribute zero gradient and should be skipped entirely, saving both the forward pass and optimizer step.

This is why sufficiently large G matters (we use G=16): it maximizes the chance of reward diversity within each group. If G is too small, most groups will be degenerate and training stalls.

### Importance Sampling Correction

Because Tinker may use non-deterministic sampling across replicas, there can be a gap between the sampling policy q and the current training policy p_theta. The `"importance_sampling"` loss function built into Tinker corrects for this by weighting each token's gradient by the ratio p_theta / q. This ensures unbiased gradient estimates even when the sampler and trainer are slightly out of sync.

---

## 4. Data & Prompt Pipeline

### Dataset

We use `agentica-org/DeepCoder-Preview-Dataset` from HuggingFace, which unifies competitive programming problems from multiple sources:

**Training splits** (concatenated):

| Split            | Source                     | Nature                                      |
| ---------------- | -------------------------- | ------------------------------------------- |
| `primeintellect` | PrimeIntellect SYNTHETIC-1 | Synthetically generated, diverse difficulty |
| `taco`           | TACO Verified              | Curated classic competitive programming     |
| `lcbv5`          | LiveCodeBench v5           | Real contest problems, recent               |

**Test splits** (for evaluation):

| Split        | Source              | Nature                               |
| ------------ | ------------------- | ------------------------------------ |
| `codeforces` | Codeforces contests | Real competition problems            |
| `lcbv5`      | LiveCodeBench v5    | Real contest problems (test portion) |

Each problem contains a natural language description, input/output test cases, and optional starter code (function signatures or boilerplate).

### Prompt Construction Chain

The prompt flows through three stages:

1. **`build_question(example)`** in `data.py` -- Extracts the problem description (from `question`, `prompt`, or `problem` fields) and optional `starter_code`, then passes them to the LCB prompt builder.

2. **`fetch_live_code_bench_system_prompt()`** in `lcb.py` -- Wraps the problem in the standard LiveCodeBench format:
   - System message: "You are an expert Python programmer..."
   - Problem description
   - Formatting instructions (stdin/stdout for problems without starter code, or function-based with starter code)
   - Explicit code fence template for the expected output format

3. **`Qwen3InstructRenderer.build_generation_prompt(messages)`** -- Converts the structured conversation (system + user messages) into tokenized `ModelInput` using ChatML format with proper special tokens.

### Configuration

The `Config` class (managed via `chz`) controls all hyperparameters:

| Parameter        | Default                        | Meaning                                                            |
| ---------------- | ------------------------------ | ------------------------------------------------------------------ |
| `model_name`     | `Qwen/Qwen3-4B-Instruct-2507` | Base model for LoRA fine-tuning                                    |
| `batch_size`     | 128                            | Total completions per training step                                |
| `group_size`     | 8                              | Prompts per batch (so 128/8 = 16 completions per prompt)           |
| `learning_rate`  | 5e-6                           | Adam LR (tuned low for G=8 variance, see Section 11)               |
| `lora_rank`      | 32                             | LoRA adapter dimension                                             |
| `max_tokens`     | 8192                           | Max generation length in tokens                                    |
| `temperature`    | 1.0                            | Sampling temperature (high entropy for exploration)                |
| `format_coef`    | 0.1                            | Weight for format reward component                                 |
| `reward_timeout` | 6                              | Sandbox timeout per test case (seconds)                            |
| `save_every`     | 10                             | Checkpoint frequency (0 = disabled)                                |
| `eval_every`     | 10                             | Evaluation frequency (-1 = disabled)                               |
| `max_steps`      | 200                            | Training steps (-1 = unlimited)                                    |

**Derived value:** `completions_per_prompt = batch_size // group_size = 16` -- this is the G in GRPO terminology.

---

## 5. Training Loop Architecture

### Pipeline Overview

The training loop is a three-stage pipeline. At its most optimized, these stages can overlap:

```
STEP N:     [--- SAMPLE ---]  [--- SCORE ---]  [--- TRAIN ---]
STEP N+1:                     [--- SAMPLE ---]  [--- SCORE ---]  [--- TRAIN ---]
```

But the recommended approach is to get the sequential version working first, then optimize.

### Initialization Flow

1. Create `tinker.ServiceClient()` (reads `TINKER_API_KEY` from environment)
2. Create LoRA training client with `service.create_lora_training_client(base_model, rank=32)`
3. Get tokenizer from training client
4. Instantiate `Qwen3InstructRenderer` via `get_renderer("qwen3_instruct", tokenizer)`
5. Set up logging via `setup_logging()` (JSON + console + optional W&B)
6. Check for existing checkpoint via `get_last_checkpoint()` -- if found, resume training client state with optimizer moments preserved
7. Load train and test datasets

### Per-Step Flow

For each training step:

1. **Select prompts** -- Pick the next `group_size` (8) problems from the training dataset, cycling with wrap-around when exhausted.

2. **Build model inputs** -- For each problem, call `build_question()` to get the prompt string, wrap it in conversation messages, and convert to tokenized `ModelInput` via the renderer.

3. **Sync weights** -- Call `training_client.save_weights_and_get_sampling_client()` to create a fresh sampling client with the latest LoRA weights. This is critical -- Tinker docs warn about sampler desync if you skip this step.

4. **Sample completions** -- For each prompt, fire `sample_async()` requesting `completions_per_prompt` (16) samples at temperature 1.0 with `max_tokens=8192`. Use `asyncio.gather()` to run all 8 prompts concurrently. Each result returns token IDs and per-token log-probabilities.

5. **Score completions** -- For each of the 128 completions, extract code and evaluate:
   - Parse the model output to text, extract the last fenced code block
   - If extraction fails (bad format): reward = -0.1, skip sandbox entirely
   - If code extracts successfully: POST to Sandbox Fusion, run against all test cases
   - Correctness reward: 1.0 if all tests pass, 0.0 otherwise
   - Total reward: `0.1 * format_score + correctness_score`
   - Use `asyncio.gather()` for concurrent sandbox calls (connection pool handles backpressure)

6. **Compute advantages & filter** -- For each group of 16 completions:
   - Calculate advantage for each completion: `A_g = R_g - mean(rewards)`
   - If all advantages are zero (degenerate group), skip the entire group
   - Collect non-degenerate `(global_index, advantage)` pairs

6b. **Compute reference logprobs** -- Only for non-degenerate completions (cost optimization):
   - Call `compute_logprobs_async()` on each surviving completion's full (prompt+completion) sequence
   - This typically halves the logprob compute vs. computing for all 128 upfront (~50% of groups are degenerate)

7. **Build training datums** -- For each non-degenerate completion, construct a `tinker.types.Datum` with:
   - Shifted token alignment (input_ids = tokens[:-1], target_ids = tokens[1:])
   - Per-token reference logprobs from the sampling policy (zero on prompt positions)
   - Per-token advantages (zero on prompt, scalar advantage on completion tokens)

8. **Train** -- If there are any datums, submit `forward_backward()` with `"importance_sampling"` loss, then `optim_step()` with Adam parameters. Both calls return `APIFuture` objects -- submit back-to-back (Tinker best practice), then wait on both.

9. **Log metrics** -- Record mean_reward, mean_correct, format_rate, groups_skipped, n_datums, loss, and per-phase timing (time/step_s, time/sync_s, time/sample_s, time/score_s, time/logprobs_s, time/train_s).

10. **Evaluate** -- Every `eval_every` steps, run Pass@1 on the held-out test set with temperature=0.

11. **Checkpoint** -- Every `save_every` steps, save full training state (LoRA weights + Adam moments) and record step + dataset offset for resume.

### The Four Core Functions

The training loop is built around four helper functions scaffolded in `train.py`:

**`should_skip(advantages)`** -- Detects degenerate groups where all rewards are identical. When all advantages are effectively zero (within floating-point tolerance), the gradient contribution is zero and we save compute by skipping. The threshold should be very small (e.g., 1e-8) to handle floating-point imprecision while catching true degeneracy.

**`compute_advantages(rewards)`** -- Computes per-completion advantages as deviation from the group mean. No standard deviation normalization -- GRPO uses raw deviations, which preserves magnitude information. A group with rewards {0, 1} should produce advantages {-0.5, +0.5}, not z-scored values.

**`make_datum(tokens, logprobs, ob_len, advantage)`** -- Constructs a Tinker `Datum` object. The key subtlety is the shift-by-1 alignment: `input_ids = tokens[:-1]`, `target_ids = tokens[1:]`. Logprobs and advantages must be zero on prompt positions and only non-zero on completion positions. Tinker's `Datum.convert_tensors()` auto-converts numpy arrays in `loss_fn_inputs` to the correct `TensorData` format.

**`train_step(training_client, datums, adam_params)`** -- Submits forward_backward and optimizer step back-to-back. Both `forward_backward()` and `optim_step()` are synchronous methods that return `APIFuture` objects (not coroutines). Submit both before waiting on either -- Tinker pipelines them on the GPU.

### Async Coordination

The main loop has a mix of sync and async operations:

- **Sync:** Tinker's `forward_backward()`, `optim_step()`, `save_weights_and_get_sampling_client()` -- these return `APIFuture` objects
- **Async:** `sample_async()`, `sandbox_check_correctness()`, `CodeEnv.step()` -- these are true coroutines

The pattern is to wrap async stages in a helper function and call them via `asyncio.run()`:

```
main(config):
    setup...
    for step in range(max_steps):
        prompts = select_next_batch()
        model_inputs = build_inputs(prompts)
        sampler = sync_weights()
        results = asyncio.run(sample_all(sampler, model_inputs))
        rewards = asyncio.run(score_all(results))
        datums = compute_datums(results, rewards)
        if datums:
            train_step(...)
        log_and_checkpoint()
```

### AdamParams

```
learning_rate = config.learning_rate  (5e-6)
beta1 = 0.9      (Tinker default)
beta2 = 0.95     (Tinker default)
eps = 1e-12      (Tinker default)
weight_decay = 0.1
grad_clip_norm = 1.0
```

The 5e-6 learning rate is conservative for G=8 GRPO — with only 16 completions per prompt, gradient variance is ~8x higher than a G=64 setup (like DeepSeekMath), requiring a proportionally lower LR. Grad clipping at 1.0 prevents catastrophic updates from high-variance groups. Weight decay at 0.1 is standard regularization. See Section 11 for the full derivation.

---

## 6. Reward System

### Reward Formula

`R = 0.1 * format_score + correctness_score`

| Format OK? | Tests Pass?  | format_score | correctness_score | Total R |
| ---------- | ------------ | ------------ | ----------------- | ------- |
| No         | -- (skipped) | -1           | 0                 | -0.1    |
| Yes        | No           | 0            | 0                 | 0.0     |
| Yes        | Yes          | 0            | 1                 | 1.0     |

### Format Scoring

The format check is deterministic and runs before any sandbox call:

1. Regex extracts all fenced code blocks from the model output (pattern: triple backticks with optional language tag, then content, then triple backticks)
2. Takes the **last** block (models often refine in later blocks)
3. If no valid block is found: `format_score = -1`
4. If found: `format_score = 0`

The format reward is a penalty, not a bonus. A badly formatted incorrect response (-0.1) is strictly worse than a well-formatted incorrect one (0.0). This establishes format compliance as a baseline expectation.

**Cost savings:** When format fails, the sandbox is skipped entirely. With ~30% format failures early in training, this saves ~30% of execution wall time.

### Sandbox Execution

For each well-formatted completion, Sandbox Fusion evaluates correctness:

1. The extracted Python code is packaged with `TEST_CODE` (the entry point script) and `TEST_UTIL` (the test harness) as base64-encoded file assets
2. A POST request is sent to `SANDBOX_URL` with the code, test cases, and per-problem timeout
3. The total timeout per problem is `(reward_timeout + 1) * num_test_cases + 5` seconds
4. The sandbox runs the code against each test case, comparing stdout output against expected results
5. `correctness_score = 1` if ALL test cases pass, 0 otherwise (binary, not fractional)

**Security:** Sandbox Fusion runs in Docker with process isolation -- infinite loops are killed by timeout, memory is bounded, and the sandbox has no access to the training code or Tinker API. This is strictly safer than raw `exec()`.

---

## 7. Optimizations & Performance

### Pipeline Architecture (Current)

The loop runs: **sync → sample → score → logprobs → train** sequentially, with one overlap optimization:

- **Prompt pre-selection**: While the GPU is busy with `forward_backward` + `optim_step`, the next step's prompts are selected and tokenized on the CPU. This hides prompt-selection latency inside the train phase.
- **Deferred logprobs**: Reference logprobs are computed only after scoring identifies non-degenerate completions, saving ~40-50% of logprob prefill cost (see below).

Sampling dominates wall-clock time (~85% of each step). Further pipeline overlap (sampling batch N+1 during training batch N) was considered but adds weight-sync complexity for marginal gain given the sampling bottleneck.

### Sandbox Concurrency

The default `SANDBOX_MAX_CONCURRENCY=4` on the client side is conservative. The sandbox Docker config allows up to 34 concurrent runners. Increasing the client concurrency to 16-32 can significantly reduce scoring time for large batches.

The `aiohttp.TCPConnector` in `env.py` handles backpressure automatically -- when all connections are busy, additional requests queue until a connection frees up. No manual semaphore management needed.

```bash
export SANDBOX_MAX_CONCURRENCY="16"
```

### Early Termination

The reward pipeline has a built-in early exit: if code extraction fails (no valid fenced block), the completion is immediately scored -0.1 and the sandbox call is skipped. This avoids wasting sandbox resources on obviously bad outputs.

### Deferred Logprobs (Cost Optimization)

An earlier version computed reference logprobs for all 128 completions in parallel with scoring ("speculative logprobs"). This wasted ~50% of Tinker's sampler prefill tokens on degenerate groups whose logprobs are never used.

The current approach: **score first, then compute logprobs only for non-degenerate completions**. With ~50% degenerate groups typical in early training, this cuts logprob prefill from ~128 to ~64 calls per step — saving ~40% of total prefill cost with negligible wall-clock impact (logprob compute is fast relative to sampling).

### Degenerate Group Skipping

When `should_skip()` returns True for a group (all advantages zero), we avoid:

- Computing reference logprobs for those completions (saves prefill tokens)
- Building `Datum` objects (saves CPU + memory allocation)
- The `forward_backward()` call for those datums (saves GPU)
- The `optim_step()` if ALL groups are degenerate (saves GPU entirely)

Early in training, expect 30-50% of groups to be degenerate (too hard for the base model). As the model improves on easy problems, degenerate rate may actually increase (all-pass groups) — this is a convergence signal.

### Connection Pool Reuse

The sandbox session in `env.py` is a singleton `aiohttp.ClientSession` -- it's created once and reused across all steps. This avoids the overhead of TCP handshakes and connection establishment for each scoring batch.

---

## 8. Evaluation Strategy

### Online Metrics (Every Step)

| Metric             | What It Measures                                                    |
| ------------------ | ------------------------------------------------------------------- |
| `mean_reward`      | Average R across all 128 completions in the batch                   |
| `mean_correct`     | Fraction with correctness_score = 1 (approximate train-time Pass@1) |
| `format_rate`      | Fraction with valid code fences                                     |
| `groups_skipped`   | How many of the 8 groups were degenerate                            |
| `groups_total`     | Total groups (always 8)                                             |
| `n_datums`         | Number of non-degenerate training datums                            |
| `time/step_s`      | Total wall-clock time for the step                                  |
| `time/sync_s`      | Weight sync (save_weights_and_get_sampling_client)                  |
| `time/sample_s`    | Sampling all 128 completions                                        |
| `time/score_s`     | Sandbox scoring all completions                                     |
| `time/logprobs_s`  | Reference logprob computation (non-degenerate only)                 |
| `time/train_s`     | forward_backward + optim_step + next prompt selection               |

### Periodic Evaluation (Every `eval_every` Steps)

- Sample from the test dataset (codeforces + lcbv5 test splits)
- Use temperature=0 (greedy decoding) for deterministic evaluation
- Compute **Pass@1** -- the gold standard metric for code generation
- Check format adherence rate (should be >95% after the first few steps)

### Smoke Test Protocol (First 20-50 Steps)

Expected behavior for a correctly wired pipeline:

| Steps | Expected Observation                                              |
| ----- | ----------------------------------------------------------------- |
| 1-5   | Format adherence rises sharply (model learns code fences quickly) |
| 5-15  | `mean_reward` trends upward (noisy but positive slope)            |
| 10-30 | `mean_correct` rises from ~0.05 baseline toward 0.10-0.20         |
| 20-50 | Pass@1 on eval set shows modest improvement vs. baseline          |

If you don't see these patterns, something is wrong. See Section 9 for diagnostics.

---

## 9. Troubleshooting & Diagnostics

### Common Errors

| Error                              | Likely Cause                                  | Fix                                                  |
| ---------------------------------- | --------------------------------------------- | ---------------------------------------------------- |
| `TINKER_API_KEY` not set           | Missing environment variable                  | `export TINKER_API_KEY="..."`                        |
| Connection refused on port 8080    | Sandbox Docker not running                    | Start the sandbox container                          |
| `SandboxError` in sandbox response | Bad sandbox config or resource limits         | Check `sandbox_config/local.yaml`                    |
| `Unknown renderer` ValueError      | Wrong renderer name string                    | Use `"qwen3_instruct"`                               |
| All groups degenerate every step   | Problems too easy or too hard, or G too small | Check reward distribution, try different dataset mix |
| `chz` CLI parse error              | Wrong invocation syntax                       | Use `python train.py main --field value`             |
| OOM during sampling                | max_tokens too high for GPU memory            | Reduce `max_tokens` or `batch_size`                  |
| Checkpoint not found on resume     | `checkpoints.jsonl` missing or corrupted      | Check log directory path                             |

### Diagnostic Signals

| Signal                        | Diagnosis                                      | Remedy                                           |
| ----------------------------- | ---------------------------------------------- | ------------------------------------------------ |
| Reward flat at 0              | Problems too hard; no positive signal          | Filter easier problems, increase G               |
| Reward flat at 1              | Too easy; model already solves everything      | Use harder subset                                |
| Format rate dropping          | Model degenerating                             | Check renderer config, reduce learning rate      |
| Reward rising but Pass@1 flat | Possible reward hacking (hardcoded outputs)    | Inspect generated code samples                   |
| Loss explodes                 | Learning rate too high                         | Reduce to 1e-5                                   |
| All groups degenerate         | G too small or difficulty too uniform          | Increase G or diversify dataset                  |
| Sandbox timeouts dominating   | Code has infinite loops or excessive recursion | Reduce `reward_timeout`, this is normal early on |

### Log File Interpretation

**`metrics.jsonl`** -- Each line is a JSON object with step number and metrics. Plot `mean_reward` and `mean_correct` over steps to verify learning progress. A healthy run shows noisy but upward-trending curves.

**`logs.log`** -- Contains detailed per-step information including individual group rewards, degenerate group warnings, and Tinker API responses. Search for `WARNING` or `ERROR` to diagnose issues.

**`checkpoints.jsonl`** -- Each line records a checkpoint's name, step, dataset offset, and file paths. The last entry with a `state_path` key is the most recent resumable checkpoint.

---

## 10. Design Principles

**Optimize for learning progress per second, not per step.** Robustness, parallelism, and latency elimination matter as much as algorithmic correctness. A pipeline that runs 3x faster with slightly noisier gradients will outperform a clean but slow implementation.

**Fail gracefully.** Bad completions get low rewards, which the policy gradient naturally de-emphasizes. Sandbox timeouts return correctness=0, not crashes. Malformed outputs get penalized and skipped. The pipeline is self-healing.

**Get sequential working first.** Pipeline overlap, advanced concurrency tuning, and latency hiding are Level 2 optimizations. The sequential version validates correctness and gives a baseline to measure speedups against.

**Use the utilities.** Everything in `tinker_utils/` is battle-tested. Don't reimplement prompt construction, test execution, or checkpoint management. Wire the existing pieces together.

### Running

```bash
caffeinate -s uv run python train.py
```

The `caffeinate -s` prevents macOS sleep — critical since the laptop orchestrates all remote Tinker API calls. If the laptop sleeps mid-step, the sampling/training calls will timeout and the step is lost.

Monitor progress with:

```bash
uv run python plot_metrics.py
```

This auto-finds the latest run in `~/code-rl-logs/` and displays a text summary with trend arrows plus a 4-panel matplotlib plot (pass rate, loss, datums, groups skipped).

---

## 11. Development Journey: How We Got Here

This section documents the key bugs, optimizations, and hyperparameter decisions made during development, and the reasoning behind each change.

### Bug Fixes (v0 → v1)

**Bug 1: Missing logprobs.** The initial implementation passed zero logprobs for all tokens. With `importance_sampling` loss, the importance weight `exp(log π_θ - log q)` was computed against fake zeros, producing meaningless gradients. **Fix**: Call `compute_logprobs_async()` on the sampling client to get actual reference logprobs from the sampling policy.

**Bug 2: asyncio session dying.** Using `asyncio.run()` for each phase (sample, score) created and destroyed a new event loop each time. The `aiohttp.ClientSession` in `env.py` is a module-level singleton — when the first loop closed, the session was invalidated. Subsequent scoring calls failed silently or raised `RuntimeError`. **Fix**: Create a single persistent `asyncio.new_event_loop()` at the start and reuse it for all `loop.run_until_complete()` calls.

**Bug 3: No epoch reshuffling.** When the dataset index wrapped around, the same order was used in every epoch, biasing training toward problems at the start of the dataset. **Fix**: Track epoch count and re-shuffle with `seed=42 + epoch` on each wrap-around.

### Cost Optimization: Sequential Logprobs

**Problem**: Computing reference logprobs for all 128 completions in parallel with scoring ("speculative logprobs") wasted ~50% of Tinker's sampler prefill budget on degenerate groups whose logprobs are never used. A 2-step smoke test consumed 2.49M prefill tokens — ~1M of which were wasted on degenerate completions.

**Solution**: Score all completions first, identify non-degenerate groups, then compute logprobs only for surviving completions. With ~50% degenerate groups, this cuts logprob calls from 128 to ~64 per step.

**Trade-off**: Scoring and logprobs now run sequentially instead of in parallel. In practice this adds negligible wall-clock time because logprob computation (~20s) is dwarfed by sampling (~250s), and we're calling half as many logprob requests.

### Hyperparameter Tuning

Each change below was motivated by specific failure modes observed in smoke tests.

#### max_tokens: 24576 → 8192

**Problem**: Sampling dominated step time at 430s/step. The model was generating up to 24K tokens per completion — far more than needed for competitive programming solutions.

**Evidence**: simpleRL-reason uses 8192 for math, Swift-GRPO uses 2048. Code solutions rarely exceed 2000 tokens. The 24K budget was burning time on padding and unnecessary generation.

**Result**: Step time dropped from 460s to 288s (37% reduction) with no observable impact on pass rate.

#### learning_rate: 4e-5 → 5e-6

**Problem**: A 2-step run with real logprobs showed pass rate dropping from 57.8% to 39.8% and loss flipping sign (+9563), indicating catastrophic policy updates.

**Analysis**: With G=8 (16 completions per prompt), gradient variance is ~8x higher than the G=64 setup used by DeepSeekMath (which uses lr=1e-6). The effective gradient noise amplification was estimated at ~80x above DeepSeekMath's regime. Literature review:

| Paper           | G    | LR   | Model Size |
| --------------- | ---- | ---- | ---------- |
| DeepSeekMath    | 64   | 1e-6 | 7B         |
| DAPO            | 16   | 1e-6 | 7-32B      |
| simpleRL-reason | 8    | 5e-6 | 7B         |
| **Ours**        | **8**| **5e-6** | **4B** |

The 5e-6 LR matches the simpleRL-reason setting for comparable G and model size.

#### grad_clip_norm: 0.0 → 1.0

With G=8 and importance sampling, individual groups can produce outsized gradients when the advantage magnitude is large and the importance weight `exp(log π - log q)` diverges. Clipping at norm 1.0 prevents single-step catastrophic updates without suppressing the learning signal during normal training.

#### weight_decay: 0.0 → 0.1

Standard regularization. Prevents LoRA adapter weights from growing unboundedly during the 200-step run. The 0.1 value is the default in most LLM training recipes (LLaMA, Qwen pretraining).

### Final Configuration Summary

```
Model:            Qwen/Qwen3-4B-Instruct-2507
LoRA rank:        32
Batch size:       128 (8 groups × 16 completions)
Learning rate:    5e-6
Grad clip norm:   1.0
Weight decay:     0.1
Max tokens:       8192
Temperature:      1.0
Max steps:        200
Eval every:       10 steps
Save every:       10 steps
```

**Expected per-step profile** (from smoke tests):
- Sampling: ~250s (dominates)
- Scoring: ~5s
- Logprobs: ~15-25s (for ~64 non-degenerate completions)
- Training: ~7s (includes next-step prompt pre-selection)
- **Total: ~290s/step → ~16 hours for 200 steps**
