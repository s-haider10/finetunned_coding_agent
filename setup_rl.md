Note: This document is created without looking at the codebase, so it has very less context. Do not take it true as it is.

# `setup_RL.md` — RL Post-Training Pipeline for Code Generation

**COSC 189.34 · AI Agents · Problem Set 1, Question 22**  
Syed Ali Haider · Winter 2026

---

## 1. Algorithmic Overview: Why GRPO (Critic-Free RL)

### 1.1 GRPO in One Paragraph

Group Relative Policy Optimization (GRPO) is a critic-free policy gradient method for LLM post-training. For each prompt $P$, we sample a **group** of $G$ completions, score each with a scalar reward, and use the **group mean** as a zero-parameter baseline. Above-average completions are reinforced; below-average are suppressed. No value network, no bootstrapping, no extra forward pass.

### 1.2 The Advantage Signal

Given $G$ completions per prompt with rewards $\{R_g\}$:

$$
\bar{R} = \frac{1}{G}\sum_{i=1}^{G} R_i, \qquad A_g = R_g - \bar{R}
$$

The policy gradient becomes:

$$
\nabla_\theta J(\theta) \approx \frac{1}{G}\sum_{g=1}^{G} A_g \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(y_t^{(g)} \mid P, y_{<t}^{(g)})
$$

Because $\bar{R}$ does not depend on the sampled $y$, this replacement does not change the expected gradient (Problem 19), but **reduces variance** substantially.

### 1.3 Contrast with PPO / Actor–Critic

|                | PPO (Actor–Critic)                     | GRPO (Critic-Free)                        |
| -------------- | -------------------------------------- | ----------------------------------------- |
| Value network  | Required — doubles parameters          | **None**                                  |
| Bootstrapping  | Yes → introduces bias                  | **No** — episode-level rewards only       |
| Baseline       | Learned $V_\phi(s)$ — slow to converge | Empirical $\bar{R}$ — free, unbiased      |
| Binary rewards | Critic must learn degenerate function  | **Natural fit** — group contrast suffices |
| Extra latency  | Critic forward pass per token          | **Zero**                                  |

For binary correctness rewards ($R \in \{0,1\}$), a learned critic solves a trivially structured problem at massive infrastructure cost. GRPO gets equivalent variance reduction by asking _"which of these G attempts worked?"_

### 1.4 Variance Reduction Without Latency

The group mean $\bar{R}$ is a **problem-specific baseline** (cf. Problem 19): it's computed from the same samples used for the gradient, requires no additional model queries, and adapts automatically to problem difficulty. Easy problems (high $\bar{R}$) and hard problems (low $\bar{R}$) contribute comparable gradient magnitudes, preventing easy problems from dominating updates.

### 1.5 Degenerate Groups

When all $G$ rewards are identical ($R_1 = \cdots = R_G$), all advantages are zero (Problem 21). The gradient contribution is exactly zero — no contrastive information exists. These groups are **skipped** (`should_skip` returns `True`), saving both the `forward_backward` and `optim_step` cost.

This happens when the problem is too easy (all correct) or too hard (all incorrect). Skipping is why a sufficiently large $G$ matters — it maximizes the chance of reward diversity within each group.

---

## 2. Data Pipeline & Prompt Construction

### 2.A Dataset: DeepCoder-Preview

**DeepCoder-Preview** (`agentica-org/DeepCoder-Preview`) unifies competitive programming problems from three sources:

| Source                         | Nature                                                      |
| ------------------------------ | ----------------------------------------------------------- |
| **TACO Verified**              | Curated classic competitive programming                     |
| **PrimeIntellect SYNTHETIC-1** | Synthetically generated, diverse difficulty                 |
| **LiveCodeBench**              | Real contest problems, recent and unseen during pretraining |

Each problem contains:

- `description` — natural language problem statement (stdin/stdout convention)
- `test_cases` / `input_output` — input/output pairs for correctness evaluation
- `starter_code` (optional) — function signature or boilerplate

The provided `tinker_utils/data.py` handles loading and converting problems into model-ready prompts. `tinker_utils/lcb.py` provides LiveCodeBench-specific parsing and test execution logic.

**Loading:**

```python
from datasets import load_dataset
dataset = load_dataset("agentica-org/DeepCoder-Preview", split="train").shuffle(seed=42)
```

We reserve the first `eval_size` problems for periodic evaluation; the rest form the training pool, iterated in rolling batches.

### 2.B Prompt Template for Qwen 3 4B Instruct

Qwen 3 Instruct uses **ChatML** (`<|im_start|>role ... <|im_end|>`). The provided `tinker_utils/qwen.py` and `tinker_utils/renderers.py` handle conversation-to-token conversion using the correct Qwen 3 renderer. We condition the model for code-only output and disable Qwen 3's thinking mode to avoid `<think>` blocks:

````
<|im_start|>system
You are a Python programming assistant. Write a complete Python program
that reads from stdin and writes to stdout. Output ONLY the code inside
a single ```python code fence. No explanations.<|im_end|>
<|im_start|>user
{problem_description}

{starter_code_if_present}<|im_end|>
<|im_start|>assistant
````

**Design rationale:**

| Choice                                  | Why                                                                         |
| --------------------------------------- | --------------------------------------------------------------------------- |
| `"reads from stdin / writes to stdout"` | Matches DeepCoder-Preview's I/O convention                                  |
| `"single ```python code fence"`         | Simplifies extraction; reduces format ambiguity under high-entropy sampling |
| `"No explanations"`                     | Suppresses post-code rambling at $T=1.0$                                    |
| System role                             | Keeps formatting instructions persistent across the conversation            |

The `tinker_utils/renderers.py` module converts this structured prompt into tokenized `ModelInput` objects compatible with Tinker's `sample` API, ensuring the correct special tokens are applied.

---

## 3. Tinker API Integration

### 3.1 Initialization

```python
import tinker

MODEL = "Qwen/Qwen3-4B"
service = tinker.ServiceClient()  # reads TINKER_API_KEY from env

training_client = service.create_lora_training_client(
    base_model=MODEL,
    rank=32,   # LoRA rank as specified
)
```

Only the low-rank adapter matrices ($B \in \mathbb{R}^{d \times 32}$, $A \in \mathbb{R}^{32 \times k}$) are trainable. The base model weights are frozen. The `tinker_utils/checkpoint.py` module wraps `save_state`/`load_state` for full optimizer-state checkpointing.

### 3.2 Sampling

After each policy update, we **sync weights** to a fresh sampling client (critical — Tinker docs warn about sampler desync):

```python
sampling_client = training_client.save_weights_and_get_sampling_client(name=f"step_{i}")

result = await sampling_client.sample_async(
    prompt=model_input,
    num_samples=G,    # G=16 completions per prompt
    sampling_params=tinker.types.SamplingParams(
        max_tokens=2048, temperature=1.0, top_p=0.95,
    ),
)
# result.samples[g].tokens → token IDs
# result.samples[g].logprobs → per-token log p_q(y_t)
```

Multiple prompts are sampled concurrently via `asyncio.gather`:

```python
futures = [sampling_client.sample_async(prompt=mi, num_samples=G, ...) for mi in inputs]
results = await asyncio.gather(*futures)
```

### 3.3 Training: `forward_backward` + `optim_step`

Tinker's built-in `"importance_sampling"` loss implements the IS-corrected policy gradient:

$$
\mathcal{L}_{\text{IS}}(\theta) = -\sum_t \frac{p_\theta(y_t \mid y_{<t})}{q(y_t \mid y_{<t})} \cdot A_t
$$

where $q$ is the sampling policy and $A_t$ is the per-token advantage. This corrects for the gap between sampling and training policies (due to non-determinism across Tinker replicas).

Each `tinker.Datum` requires three loss inputs:

- `target_tokens` — the sampled token IDs (shifted left by 1)
- `logprobs` — per-token log-probabilities from the sampler $q$
- `advantages` — per-token advantages (0 on prompt, $A_g$ on completion)

The `train_step` function submits **back-to-back** (Tinker best practice per `AGENTS.md`):

```python
async def train_step(training_client, datums, adam_params):
    fwd_future = await training_client.forward_backward_async(
        datums, loss_fn="importance_sampling"
    )
    opt_future = await training_client.optim_step_async(adam_params)
    await fwd_future
    await opt_future
```

**Learning rate:** LoRA needs ~10× higher LR than full fine-tuning. We use `4e-5` following `tinker_utils` recommendations and `hyperparam_utils.get_lr()`.

### 3.4 Checkpointing

Via `tinker_utils/checkpoint.py`:

```python
training_client.save_state(path)   # full state: LoRA weights + Adam moments
training_client.load_state(path)   # resume from checkpoint
```

---

## 4. Reward Function

$$
R = \alpha \cdot R_{\text{format}} + R_{\text{correct}}, \qquad \alpha = 0.1
$$

### 4.A Format Reward: $R_{\text{format}} \in \{-1, 0\}$

**Deterministic, cheap, runs before any sandbox call:**

1. Regex-match all fenced code blocks: ` ```python ... ``` `
2. Prefer the **last** fence (models often refine in later blocks)
3. `ast.parse()` validation on the candidate
4. If a valid block is found: $R_{\text{format}} = 0$. Otherwise: $R_{\text{format}} = -1$.

**Why negative?** Format failure is a _penalty_ ($R = -0.1$), not a missing bonus. This establishes format compliance as a baseline expectation and differentiates failure modes: a badly formatted incorrect response ($-0.1$) is strictly worse than a well-formatted incorrect one ($0.0$). See Problem 17.

**Cost savings:** When $R_{\text{format}} = -1$, we skip sandbox execution entirely. With ~30% format failures early in training, this saves ~30% of execution wall time.

### 4.B Correctness Reward: $R_{\text{correct}} \in \{0, 1\}$

Code execution uses **Sandbox Fusion** (ByteDance), running as a Docker container:

```bash
docker run -it -p 8080:8080 \
  -v ./sandbox_config/local.yaml:/root/sandbox/sandbox/configs/local.yaml \
  volcengine/sandbox-fusion:server-20250609
```

The `tinker_utils/env.py` module wraps the sandbox HTTP API:

```python
SANDBOX_URL = os.getenv("SANDBOX_URL", "http://localhost:8080/run_code")
```

For each test case, we POST to the sandbox:

```json
{
  "code": "<extracted python code with test harness>",
  "language": "python"
}
```

The response contains `run_result.status` (Finished/TimeLimitExceeded/etc.) and `run_result.stdout`. We compare stdout against expected output.

$R_{\text{correct}} = 1$ iff **all** test cases pass. Binary, not fractional — see Problem 12 for why.

### 4.C Security (Sandbox Fusion)

| Threat                      | Mitigation                                           |
| --------------------------- | ---------------------------------------------------- |
| Infinite loops / fork bombs | Sandbox enforces configurable timeout (default 10s)  |
| Memory exhaustion           | Sandbox config limits memory per execution           |
| Filesystem / network        | Docker container isolation; no host access           |
| Reward manipulation         | Sandbox has no access to training code or Tinker API |

This is strictly safer than raw `exec()` on the host — see Problem 16.

### 4.D Reward Outcome Table

| Format? | Tests Pass? | $R_{\text{format}}$ | $R_{\text{correct}}$ |  $R$   |
| :-----: | :---------: | :-----------------: | :------------------: | :----: |
|    ✗    |  — (skip)   |        $-1$         |         $0$          | $-0.1$ |
|    ✓    |      ✗      |         $0$         |         $0$          | $0.0$  |
|    ✓    |      ✓      |         $0$         |         $1$          | $1.0$  |

---

## 5. Training Loop Architecture

### Design Principle

> The loop is a **three-stage async pipeline**, not a sequential loop.

```
┌──────────────────┐      ┌──────────────────────┐      ┌────────────────┐
│  SAMPLE (GPU)     │ ───> │  EXECUTE (Sandbox)    │ ───> │  TRAIN (GPU)   │
│  batch N+1        │      │  batch N              │      │  batch N-1     │
│  Tinker remote     │      │  Sandbox Fusion local  │      │  Tinker remote  │
└──────────────────┘      └──────────────────────┘      └────────────────┘
```

### 5.A Sampling (Async, Concurrent)

For each batch of $B$ prompts, fire $B$ concurrent `sample_async` calls via `asyncio.gather`:

```python
futures = [sampling_client.sample_async(prompt=mi, num_samples=G, ...) for mi in batch]
results = await asyncio.gather(*futures)
```

Each returns $G$ completions with per-token logprobs. Tinker batches internally on the GPU.

### 5.B Execution & Scoring (Parallel via Sandbox Fusion)

All $B \times G$ completions are scored concurrently. The `SANDBOX_MAX_CONCURRENCY` setting (default 4, configurable) limits concurrent sandbox requests:

```python
sem = asyncio.Semaphore(SANDBOX_MAX_CONCURRENCY)

async def score_with_backpressure(code, test_cases):
    async with sem:
        return await execute_in_sandbox(code, test_cases)

tasks = [score_with_backpressure(code, tc) for code, tc in pairs]
rewards = await asyncio.gather(*tasks)
```

Format-failing completions are scored instantly ($R = -0.1$) without any sandbox call.

### 5.C Advantage Computation

Maps directly to the scaffolded `compute_advantages` and `should_skip`:

```python
def compute_advantages(rewards: list[float]) -> list[float]:
    r_bar = sum(rewards) / len(rewards)
    return [r - r_bar for r in rewards]

def should_skip(advantages: list[float]) -> bool:
    return all(abs(a) < 1e-8 for a in advantages)
```

When `should_skip` returns `True`, the entire group is dropped — zero gradient contribution, zero compute wasted.

### 5.D Datum Construction

Maps to the scaffolded `make_datum`:

```python
def make_datum(tokens, logprobs, ob_len, advantage) -> tinker.types.Datum:
    # tokens: full sequence (prompt + completion)
    # logprobs: per-token logprobs from sampler (0 on prompt positions)
    # ob_len: length of prompt (observation) prefix
    # advantage: scalar A_g for this completion
    input_ids  = tokens[:-1]
    target_ids = tokens[1:]
    N = len(input_ids)

    lp = np.zeros(N, dtype=np.float32)
    lp[ob_len-1:] = logprobs[ob_len:]  # align with target positions

    adv = np.zeros(N, dtype=np.float32)
    adv[ob_len-1:] = advantage          # only on completion tokens

    return tinker.Datum(
        model_input=tinker.ModelInput(
            chunks=[tinker.types.EncodedTextChunk(tokens=input_ids)]
        ),
        loss_fn_inputs={
            "target_tokens": TensorData.from_numpy(np.array(target_ids, dtype=np.int64)),
            "logprobs": TensorData.from_numpy(lp),
            "advantages": TensorData.from_numpy(adv),
        },
    )
```

### 5.E Policy Update

Maps to the scaffolded `train_step`:

```python
async def train_step(training_client, datums, adam_params):
    # Back-to-back submission (Tinker best practice: submit before awaiting)
    fwd = await training_client.forward_backward_async(datums, loss_fn="importance_sampling")
    opt = await training_client.optim_step_async(adam_params)
    await fwd
    await opt
```

### 5.F Latency Hiding

| Stage                             | Blocked on                   | Overlapped with              |
| --------------------------------- | ---------------------------- | ---------------------------- |
| `sample_async` (batch $N+1$)      | Tinker GPU                   | Execute + train on batch $N$ |
| Sandbox execution (batch $N$)     | HTTP calls to Sandbox Fusion | Sampling batch $N+1$         |
| `forward_backward` + `optim_step` | Tinker GPU                   | Next batch sampling          |

Effective step time: $\max(t_{\text{sample}},\ t_{\text{execute}},\ t_{\text{train}})$ instead of the sum.

---

## 6. Systems Design Considerations

### Async I/O vs Multiprocessing

The workload is **I/O-bound**: waiting for Tinker API responses (remote GPU) and Sandbox Fusion HTTP responses. `asyncio` is the natural fit — it manages thousands of concurrent I/O operations on one thread. No threading, no multiprocessing, no shared-state bugs.

Sandbox Fusion runs in Docker and is accessed via HTTP POST. The `aiohttp` or `httpx` library with async handles concurrent sandbox requests cleanly.

### Batching Strategy

| What                     | Size             | Why                                                      |
| ------------------------ | ---------------- | -------------------------------------------------------- |
| Prompts per step         | $B = 8$          | Saturates Tinker sampling without excessive memory       |
| Completions per prompt   | $G = 16$         | Enough diversity for meaningful advantages               |
| Total completions / step | $128$            | $8 \times 16$                                            |
| Sandbox concurrency      | 4 (configurable) | `SANDBOX_MAX_CONCURRENCY` — respects local Docker limits |

### Backpressure Handling

The `asyncio.Semaphore(SANDBOX_MAX_CONCURRENCY)` ensures we don't overwhelm the local Sandbox Fusion container. If sandbox execution is the bottleneck, sampling for the next batch proceeds concurrently, keeping the GPU occupied.

### Failure Tolerance

| Failure Mode                     | Response                                                       |
| -------------------------------- | -------------------------------------------------------------- |
| Sandbox timeout                  | Sandbox returns `TimeLimitExceeded` → $R_{\text{correct}} = 0$ |
| Sandbox crash / OOM              | HTTP error or non-`Finished` status → $R_{\text{correct}} = 0$ |
| Malformed model output           | Code extraction fails → $R = -0.1$, skip sandbox               |
| Empty completion                 | Extraction returns $\varnothing$ → $R = -0.1$                  |
| Tinker API error                 | Retry with exponential backoff (3 attempts)                    |
| All groups degenerate in a batch | Log warning, skip `train_step` entirely                        |
| Dataset exhausted                | Wrap around with re-shuffle                                    |

The pipeline is self-healing: bad completions get low rewards, which the policy gradient naturally de-emphasizes.

### Determinism vs Throughput

We prioritize throughput. Tinker sampling may be non-deterministic across replicas; we accept this and rely on the importance sampling correction in the loss function to account for $p_\theta \neq q$. Sandbox execution order is also non-deterministic, but rewards are idempotent.

---

## 7. Evaluation & Verification

### 7.1 Online Metrics (Logged Every Step via `tinker_utils/log.py`)

| Metric                          | Meaning                                                    |
| ------------------------------- | ---------------------------------------------------------- |
| `mean_reward`                   | Average $R$ across all completions in the batch            |
| `mean_correct`                  | Fraction passing all tests (≈ train-time Pass@1)           |
| `format_rate`                   | Fraction with valid code fences                            |
| `groups_skipped / groups_total` | Degenerate group rate                                      |
| `loss`                          | IS-weighted policy gradient loss (from `forward_backward`) |
| `n_datums`                      | Non-degenerate training signal volume                      |

### 7.2 Periodic Evaluation (Every `eval_every` Steps)

- **Pass@1:** Greedy decoding (temperature=0) on held-out problems — the gold standard metric.
- **Format adherence:** Should exceed 95% within the first 5–10 steps.

### 7.3 Smoke Test Protocol (20–50 Steps)

Expected behavior for a correctly wired pipeline:

| Steps | Expected Observation                                      |
| ----- | --------------------------------------------------------- |
| 1–5   | Format adherence rises sharply (model learns fences)      |
| 5–15  | `mean_reward` trends upward (noisy but positive slope)    |
| 10–30 | `mean_correct` rises from ~0.05 baseline toward 0.10–0.20 |
| 20–50 | Pass@1 on eval set shows modest improvement vs. baseline  |

### 7.4 Diagnostic Signals

| Signal                   | Diagnosis                                 | Remedy                               |
| ------------------------ | ----------------------------------------- | ------------------------------------ |
| Reward flat at 0         | Too hard; no positive signal              | Filter easier problems, increase $G$ |
| Reward flat at 1         | Too easy; model already solves everything | Use harder subset                    |
| Format rate dropping     | Model degenerating                        | Check renderer, reduce LR            |
| Reward ↑ but Pass@1 flat | Reward hacking (hardcoded outputs?)       | Inspect generated code               |
| Loss explodes            | LR too high                               | Reduce to 1e-5                       |
| All groups degenerate    | $G$ too small or difficulty too uniform   | Increase $G$ or diversify dataset    |

---

## 8. Deliverables

### 8.1 File Structure

```
├── train.py                         # Main RL training loop
├── requirements.txt                 # Dependencies
├── setup_RL.md                      # This document
├── sandbox_config/
│   └── local.yaml                   # Sandbox Fusion config
└── tinker_utils/                    # Provided utilities
    ├── checkpoint.py                # Save/load training state
    ├── data.py                      # Dataset → prompt conversion
    ├── env.py                       # Sandbox execution environment
    ├── log.py                       # JSON + console + W&B logging
    ├── lcb.py                       # LiveCodeBench format handling
    ├── renderers.py                 # Conversation → token rendering
    └── qwen.py                      # Qwen 3 specific renderer
```

### 8.2 `train.py` — Key Functions (Scaffolded)

```
should_skip(advantages)              → bool    # True if all advantages ≈ 0
compute_advantages(rewards)          → list    # A_g = R_g − R̄
make_datum(tokens, logprobs,         → Datum   # Tinker Datum with IS inputs
           ob_len, advantage)
train_step(training_client,          → None    # forward_backward + optim_step
           datums, adam_params)
```

These compose into the main loop:

```
for step in range(max_steps):
    sampler = sync_weights()                    # fresh sampling client
    results = await gather(sample_all_prompts)  # async sampling
    rewards = await gather(score_all_codes)     # async sandbox execution
    for group in groups:
        advs = compute_advantages(group.rewards)
        if should_skip(advs): continue
        datums += [make_datum(...) for each completion]
    await train_step(training_client, datums, adam_params)
    if step % eval_every == 0: run_eval()
    if step % save_every == 0: save_checkpoint()
```

### 8.3 `requirements.txt`

```
tinker                    # Tinker API client
transformers              # Tokenizer (Qwen 3)
datasets                  # HuggingFace datasets (DeepCoder-Preview)
torch                     # Tensor ops for TensorData
numpy                     # Array construction
aiohttp                   # Async HTTP for Sandbox Fusion
```

### 8.4 Async + Parallel Execution — Summary

| Where                | Mechanism                                                                 | Wall-clock savings                               |
| -------------------- | ------------------------------------------------------------------------- | ------------------------------------------------ |
| **Sampling**         | `asyncio.gather` over $B$ `sample_async` calls                            | $B$ prompts in parallel, not serial              |
| **Execution**        | `asyncio.gather` over $B \times G$ sandbox HTTP calls (semaphore-bounded) | Parallelizes the CPU/IO bottleneck               |
| **Training**         | Back-to-back `forward_backward_async` + `optim_step_async`                | Tinker pipelines gradient + optimizer internally |
| **Pipeline overlap** | Sample batch $N+1$ while training on batch $N$                            | GPU never idles between steps                    |

**Net effect:** Step time is $\max(t_{\text{sample}}, t_{\text{execute}}, t_{\text{train}})$, not the sum. For $B=8, G=16$, this yields **~2–3× speedup** in wall-clock time per update.

> **Guiding principle:** Optimize for _learning progress per second_, not per step. Robustness, parallelism, and latency elimination matter as much as algorithmic correctness.
