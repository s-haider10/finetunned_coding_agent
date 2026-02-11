import os
import datetime
import logging
import chz
import datasets
import tinker
from typing import cast


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


########################################################################
# Helper functions
########################################################################
def should_skip(advantages: list[float]) -> bool:
    raise NotImplementedError("This function needs to be implemented.")


def compute_advantages(rewards: list[float]) -> list[float]:
    raise NotImplementedError("This function needs to be implemented.")


def make_datum(
    tokens: list[int],
    logprobs: list[float],
    ob_len: int,
    advantage: float
) -> tinker.types.Datum:
    raise NotImplementedError("This function needs to be implemented.")


def train_step(
    training_client: tinker.TrainingClient,
    datums: list[tinker.types.Datum],
    adam_params: tinker.types.AdamParams
) -> None:
    raise NotImplementedError("This function needs to be implemented.")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
