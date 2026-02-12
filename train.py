import os
import asyncio
import datetime
import logging
import numpy as np
import chz
import datasets
import tinker
from dotenv import load_dotenv
from typing import cast, Any

load_dotenv()

from tinker_utils.data import build_question
from tinker_utils.env import CodeEnv
from tinker_utils.lcb import normalize_tests
from tinker_utils.renderers import get_renderer, Message
from tinker_utils.log import setup_logging
from tinker_utils.checkpoint import save_checkpoint, get_last_checkpoint


logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = os.path.join(
        os.path.expanduser("~/code-rl-logs"),
        datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    )
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    batch_size: int = 128
    group_size: int = 8
    learning_rate: float = 4e-5
    lora_rank: int = 32
    save_every: int = 10  # 0 = disabled
    eval_every: int = 10 # -1 = disabled
    max_tokens: int = 24576
    format_coef: float = 0.1
    reward_timeout: int = 6
    temperature: float = 1.0
    max_steps: int = -1  # -1 = unlimited


def _get_tests(example: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract and normalize test cases from a dataset example."""
    raw = example.get("test_cases") or example.get("input_output") or example.get("tests")
    metadata = example.get("metadata", {})
    if isinstance(metadata, str):
        import json as _json
        try:
            metadata = _json.loads(metadata)
        except Exception:
            metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    return normalize_tests(raw, metadata)


def main(config: Config):
    # Load dataset
    logger.info("Loading dataset...")
    train_dataset = datasets.concatenate_datasets(
        [
            cast(
                datasets.Dataset,
                datasets.load_dataset("agentica-org/DeepCoder-Preview-Dataset", name=name, split="train")
            ) for name in ("primeintellect", "taco", "lcbv5")
        ]
    )

    test_dataset = datasets.concatenate_datasets(
        [
            cast(
                datasets.Dataset,
                datasets.load_dataset("agentica-org/DeepCoder-Preview-Dataset", name=name, split="test")
            ) for name in ("codeforces", "lcbv5")
        ]
    )

    train_dataset = train_dataset.shuffle(seed=42)

    completions_per_prompt = config.batch_size // config.group_size

    # Initialize Tinker
    logger.info("Initializing Tinker service...")
    service = tinker.ServiceClient(base_url=config.base_url)
    training_client = service.create_lora_training_client(
        base_model=config.model_name,
        rank=config.lora_rank,
    )

    # Tokenizer and renderer
    tokenizer = training_client.get_tokenizer()
    renderer = get_renderer("qwen3_instruct", tokenizer)

    # Logging
    ml_logger = setup_logging(
        log_dir=config.log_path,
        config=config,
    )

    # Adam params
    adam_params = tinker.types.AdamParams(learning_rate=config.learning_rate)

    # Sampling params
    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        stop=renderer.get_stop_sequences(),
    )

    # Check for checkpoint resume
    start_step = 0
    dataset_offset = 0
    last_ckpt = get_last_checkpoint(config.log_path)
    if last_ckpt is not None:
        logger.info(f"Resuming from checkpoint: {last_ckpt}")
        training_client.load_state_with_optimizer(last_ckpt["state_path"]).result()
        start_step = last_ckpt.get("step", 0) + 1
        dataset_offset = last_ckpt.get("dataset_offset", 0)
        logger.info(f"Resumed at step {start_step}, dataset offset {dataset_offset}")

    # Determine max steps
    max_steps = config.max_steps if config.max_steps > 0 else float("inf")

    # Async helpers
    async def sample_batch(
        sampling_client: tinker.SamplingClient,
        model_inputs: list[tinker.ModelInput],
    ) -> list[tinker.types.SampleResponse]:
        futures = [
            sampling_client.sample_async(
                prompt=mi,
                num_samples=completions_per_prompt,
                sampling_params=sampling_params,
            )
            for mi in model_inputs
        ]
        return await asyncio.gather(*futures)

    async def score_completion(
        completion_tokens: list[int],
        tests: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Score a single completion using the CodeEnv."""
        env = CodeEnv(
            problem="",
            tests=tests,
            renderer=renderer,
            format_coef=config.format_coef,
            reward_timeout=config.reward_timeout,
        )
        result = await env.step(completion_tokens)
        return {
            "reward": result.reward,
            "format": result.metrics.get("format", 0.0),
            "correct": result.metrics.get("correct", 0.0),
        }

    async def score_all_completions(
        all_tokens: list[list[int]],
        all_tests: list[list[dict[str, Any]]],
    ) -> list[dict[str, float]]:
        tasks = [
            score_completion(tokens, tests)
            for tokens, tests in zip(all_tokens, all_tests)
        ]
        return await asyncio.gather(*tasks)

    # Main training loop
    step = start_step
    idx = dataset_offset
    n_train = len(train_dataset)

    logger.info(f"Starting training from step {step}, dataset has {n_train} examples")

    while step < max_steps:
        logger.info(f"=== Step {step} ===")

        # 1. Select prompts — skip individual bad examples instead of failing entire batch
        model_inputs = []
        batch_tests = []
        attempts = 0
        max_attempts = config.group_size * 10  # safety limit

        while len(model_inputs) < config.group_size and attempts < max_attempts:
            example = train_dataset[idx % n_train]
            idx += 1
            attempts += 1

            question = build_question(example)
            if question is None:
                logger.debug(f"Skipping example at idx {idx - 1}: no valid question")
                continue
            tests = _get_tests(example)
            if not tests:
                logger.debug(f"Skipping example at idx {idx - 1}: no valid tests")
                continue

            messages = [Message(role="user", content=question)]
            mi = renderer.build_generation_prompt(messages)
            model_inputs.append(mi)
            batch_tests.append(tests)

        if len(model_inputs) < config.group_size:
            logger.warning(f"Could only find {len(model_inputs)}/{config.group_size} valid examples after {max_attempts} attempts, skipping step")
            continue

        # 3. Sync weights and get sampling client
        sampling_client = training_client.save_weights_and_get_sampling_client(
            name=f"step_{step}"
        )

        # 4. Sample completions
        logger.info(f"Sampling {config.group_size} x {completions_per_prompt} completions...")
        sample_results = asyncio.run(sample_batch(sampling_client, model_inputs))

        # 5. Score all completions
        all_completion_tokens = []
        all_completion_tests = []
        ob_lens = []

        for prompt_idx, (mi, sr) in enumerate(zip(model_inputs, sample_results)):
            ob_len = mi.length
            for seq in sr.sequences:
                all_completion_tokens.append(seq.tokens)
                all_completion_tests.append(batch_tests[prompt_idx])
                ob_lens.append(ob_len)

        logger.info(f"Scoring {len(all_completion_tokens)} completions...")
        score_results = asyncio.run(
            score_all_completions(all_completion_tokens, all_completion_tests)
        )

        # 6. Compute advantages and build datums
        datums: list[tinker.types.Datum] = []
        all_rewards = []
        all_formats = []
        all_corrects = []
        groups_skipped = 0

        for g in range(config.group_size):
            start = g * completions_per_prompt
            end = start + completions_per_prompt

            group_rewards = [score_results[i]["reward"] for i in range(start, end)]
            group_formats = [score_results[i]["format"] for i in range(start, end)]
            group_corrects = [score_results[i]["correct"] for i in range(start, end)]

            all_rewards.extend(group_rewards)
            all_formats.extend(group_formats)
            all_corrects.extend(group_corrects)

            advantages = compute_advantages(group_rewards)

            if should_skip(advantages):
                groups_skipped += 1
                continue

            for i in range(completions_per_prompt):
                global_idx = start + i
                sr = sample_results[g]
                seq = sr.sequences[i]

                full_tokens = list(model_inputs[g].to_ints()) + list(seq.tokens)
                logprobs_full = [0.0] * ob_lens[global_idx] + (seq.logprobs or [0.0] * len(seq.tokens))

                datum = make_datum(
                    tokens=full_tokens,
                    logprobs=logprobs_full,
                    ob_len=ob_lens[global_idx],
                    advantage=advantages[i],
                )
                datums.append(datum)

        # 7. Train
        train_metrics: dict[str, float] = {}
        if datums:
            logger.info(f"Training on {len(datums)} datums ({groups_skipped}/{config.group_size} groups skipped)...")
            train_metrics = train_step(training_client, datums, adam_params)
        else:
            logger.warning("All groups degenerate, skipping train step")

        # 8. Log metrics
        mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        format_rate = sum(1 for f in all_formats if f >= 0) / len(all_formats) if all_formats else 0.0
        mean_correct = sum(all_corrects) / len(all_corrects) if all_corrects else 0.0

        metrics = {
            "mean_reward": mean_reward,
            "mean_correct": mean_correct,
            "format_rate": format_rate,
            "groups_skipped": groups_skipped,
            "groups_total": config.group_size,
            "n_datums": len(datums),
            **train_metrics,
        }
        ml_logger.log_metrics(metrics, step=step)

        # 9. Evaluation
        if config.eval_every > 0 and step > 0 and step % config.eval_every == 0:
            logger.info(f"Running evaluation at step {step}...")
            eval_sampling_client = training_client.save_weights_and_get_sampling_client(
                name=f"eval_step_{step}"
            )
            eval_params = tinker.types.SamplingParams(
                max_tokens=config.max_tokens,
                temperature=0.0,
                stop=renderer.get_stop_sequences(),
            )

            n_eval = min(20, len(test_dataset))
            eval_examples = [test_dataset[i] for i in range(n_eval)]

            async def run_eval():
                eval_inputs = []
                eval_tests_list = []
                for ex in eval_examples:
                    q = build_question(ex)
                    if q is None:
                        continue
                    t = _get_tests(ex)
                    if not t:
                        continue
                    msgs = [Message(role="user", content=q)]
                    eval_inputs.append(renderer.build_generation_prompt(msgs))
                    eval_tests_list.append(t)

                eval_futures = [
                    eval_sampling_client.sample_async(
                        prompt=mi, num_samples=1, sampling_params=eval_params
                    )
                    for mi in eval_inputs
                ]
                eval_results = await asyncio.gather(*eval_futures)

                eval_tokens = []
                eval_tests_flat = []
                for i, er in enumerate(eval_results):
                    for seq in er.sequences:
                        eval_tokens.append(seq.tokens)
                        eval_tests_flat.append(eval_tests_list[i])

                return await score_all_completions(eval_tokens, eval_tests_flat)

            eval_scores = asyncio.run(run_eval())
            if eval_scores:
                eval_correct = sum(s["correct"] for s in eval_scores) / len(eval_scores)
                eval_format = sum(1 for s in eval_scores if s["format"] >= 0) / len(eval_scores)
                eval_metrics = {
                    "eval/pass_at_1": eval_correct,
                    "eval/format_rate": eval_format,
                    "eval/n_problems": len(eval_scores),
                }
                ml_logger.log_metrics(eval_metrics, step=step)
                logger.info(f"Eval: Pass@1={eval_correct:.3f}, Format={eval_format:.3f}")

        # 10. Checkpoint
        if config.save_every > 0 and step > 0 and step % config.save_every == 0:
            logger.info(f"Saving checkpoint at step {step}...")
            save_checkpoint(
                training_client=training_client,
                name=f"step_{step}",
                log_path=config.log_path,
                loop_state={"step": step, "dataset_offset": idx},
                kind="both",
            )

        step += 1

    logger.info("Training complete.")
    ml_logger.close()


########################################################################
# Helper functions
########################################################################
def should_skip(advantages: list[float]) -> bool:
    return all(abs(a) < 1e-8 for a in advantages)


def compute_advantages(rewards: list[float]) -> list[float]:
    mean_r = sum(rewards) / len(rewards)
    return [r - mean_r for r in rewards]


def make_datum(
    tokens: list[int],
    logprobs: list[float],
    ob_len: int,
    advantage: float
) -> tinker.types.Datum:
    input_ids = tokens[:-1]
    target_ids = tokens[1:]
    N = len(input_ids)

    lp = np.zeros(N, dtype=np.float32)
    lp[ob_len - 1:] = logprobs[ob_len:]

    adv = np.zeros(N, dtype=np.float32)
    adv[ob_len - 1:] = advantage

    return tinker.types.Datum(
        model_input=tinker.ModelInput(
            chunks=[tinker.types.EncodedTextChunk(tokens=input_ids)]
        ),
        loss_fn_inputs={
            "target_tokens": np.array(target_ids, dtype=np.int64),
            "logprobs": lp,
            "advantages": adv,
        },
    )


def train_step(
    training_client: tinker.TrainingClient,
    datums: list[tinker.types.Datum],
    adam_params: tinker.types.AdamParams
) -> dict[str, float]:
    fwd_future = training_client.forward_backward(datums, "importance_sampling")
    opt_future = training_client.optim_step(adam_params)
    fwd_result = fwd_future.result()
    opt_future.result()
    return fwd_result.metrics


if __name__ == "__main__":
    chz.nested_entrypoint(main)
