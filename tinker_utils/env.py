import os
import re
import logging
import json
import base64
import asyncio
import aiohttp
import tinker
from dataclasses import dataclass, field
from tinker_utils.renderers import Renderer, Message, get_text_content
from tinker_utils.lcb import TEST_CODE, TEST_UTIL
from typing import Literal, Any, TypeAlias


Action: TypeAlias = list[int]
Observation: TypeAlias = tinker.ModelInput
Logprobs: TypeAlias = list[float]
Metrics: TypeAlias = dict[str, float | int]
StopCondition: TypeAlias = list[str] | list[int]
LossFnType: TypeAlias = Literal["cross_entropy", "importance_sampling", "ppo", "cispo", "dro"]


@dataclass
class StepResult:
    reward: float
    episode_done: bool
    next_observation: Observation
    next_stop_condition: StopCondition
    metrics: Metrics = field(default_factory=dict)


logger = logging.getLogger(__name__)


# Sandbox configuration
SANDBOX_URL = os.getenv("SANDBOX_URL", "http://localhost:8080/run_code")
SANDBOX_MAX_CONCURRENCY = int(os.getenv("SANDBOX_MAX_CONCURRENCY", "4"))
CLIENT_TIMEOUT_SECONDS = int(os.getenv("SANDBOX_CLIENT_TIMEOUT_SECONDS", "6000"))

# Sandbox session management
_SANDBOX_SESSION: aiohttp.ClientSession | None = None
_SANDBOX_SESSION_LOCK = asyncio.Lock()


async def _get_sandbox_session() -> aiohttp.ClientSession:
    """
    Get or create a shared aiohttp session with connection limits.

    The TCPConnector limits concurrent connections to SANDBOX_MAX_CONCURRENCY.
    When all connections are busy, additional requests automatically wait in a queue
    until a connection becomes available.
    """
    global _SANDBOX_SESSION

    async with _SANDBOX_SESSION_LOCK:
        if _SANDBOX_SESSION is None or _SANDBOX_SESSION.closed:
            connector = aiohttp.TCPConnector(
                limit=SANDBOX_MAX_CONCURRENCY,
                limit_per_host=SANDBOX_MAX_CONCURRENCY,
            )
            timeout = aiohttp.ClientTimeout(total=int(CLIENT_TIMEOUT_SECONDS))
            _SANDBOX_SESSION = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
        return _SANDBOX_SESSION


def postprocess_lcb_sample(sample: list[dict[str, Any]]) -> dict[str, str]:
    sample_inputs = [item["input"] for item in sample]
    sample_outputs = [item["output"] for item in sample]

    sample_dict: dict[str, Any] = {
        "inputs": sample_inputs,
        "outputs": sample_outputs,
    }

    if sample[0].get("testtype") == "functional":
        metadata = sample[0].get("metadata", {})
        fn_name = metadata.get("func_name")
        if fn_name is None:
            raise AssertionError(f"Function name missing in metadata: {metadata}. Sample: {sample}")
        sample_dict["fn_name"] = fn_name

    return {
        "input_output": json.dumps(sample_dict),
    }


async def sandbox_check_correctness(
    sample: list[dict[str, Any]], generation: str, timeout: int = 6
) -> tuple[bool, dict[str, Any]]:
    """Check correctness of generated code using sandbox execution."""
    assert len(sample) >= 1, "Sample must contain at least one test case"

    # Process test cases
    test_cases = postprocess_lcb_sample(sample)

    b64encode = lambda s: base64.b64encode(s.encode("utf-8")).decode("utf-8")
    try:
        test_cnt = len(json.loads(test_cases["input_output"])["inputs"])
        total_timeout = (timeout + 1) * test_cnt + 5

        test_code = TEST_CODE % {"timeout": timeout}
        asset = {
            "test_cases.txt": b64encode(json.dumps(test_cases)),
            "code.py": b64encode(generation),
            "testing_util.py": b64encode(TEST_UTIL),
        }

        payload = {
            "code": test_code,
            "language": "python",
            "run_timeout": total_timeout,
            "files": asset,
        }

        session = await _get_sandbox_session()
        async with session.post(SANDBOX_URL, json=payload) as result:
            if result.status != 200:
                raise Exception(
                    f"Sandbox API responded with code {result.status}: {await result.text()}"
                )
            resp = await result.json()
            if resp.get("status") == "SandboxError":
                raise Exception(f"Sandbox responded with error: {resp.get('message')}")

            # Check if all tests passed
            all_passed = resp.get("status") == "Success"
            return all_passed, resp
    except Exception as e:
        return False, {"error": str(e)}


# --- Fractional scoring (Run 2) ---
# Unlike TEST_CODE which exits -1 on first failure, this runs all tests
# and prints {"passed": N, "total": M} as JSON to stdout.
# run_test() does early-return, so len(result) may be < total — we get
# total from the input count, not from len(result).
_FRACTIONAL_TEST_CODE = """
import json
import sys
from testing_util import run_test

with open('test_cases.txt', 'r') as fin:
    sample = json.load(fin)
with open('code.py', 'r') as fin:
    code = fin.read()

result, metadata = run_test(sample, code, debug=False, timeout=%(timeout)s)
passed = sum(1 for x in result if x is True)
total = len(json.loads(sample["input_output"])["inputs"])
print(json.dumps({"passed": passed, "total": total}))
"""


async def sandbox_check_correctness_fractional(
    sample: list[dict[str, Any]], generation: str, timeout: int = 6
) -> tuple[float, dict[str, Any]]:
    """Check correctness with fractional score (passed_tests / total_tests)."""
    assert len(sample) >= 1, "Sample must contain at least one test case"

    test_cases = postprocess_lcb_sample(sample)
    b64encode = lambda s: base64.b64encode(s.encode("utf-8")).decode("utf-8")
    try:
        test_cnt = len(json.loads(test_cases["input_output"])["inputs"])
        total_timeout = (timeout + 1) * test_cnt + 5

        test_code = _FRACTIONAL_TEST_CODE % {"timeout": timeout}
        asset = {
            "test_cases.txt": b64encode(json.dumps(test_cases)),
            "code.py": b64encode(generation),
            "testing_util.py": b64encode(TEST_UTIL),
        }

        payload = {
            "code": test_code,
            "language": "python",
            "run_timeout": total_timeout,
            "files": asset,
        }

        session = await _get_sandbox_session()
        async with session.post(SANDBOX_URL, json=payload) as result:
            if result.status != 200:
                raise Exception(
                    f"Sandbox API responded with code {result.status}: {await result.text()}"
                )
            resp = await result.json()
            # Parse per-test results from stdout
            # Sandbox puts stdout in resp["run_result"]["stdout"], not resp["output"]
            run_result = resp.get("run_result") or {}
            output = run_result.get("stdout", "") if isinstance(run_result, dict) else ""
            try:
                counts = json.loads(output.strip().split("\n")[-1])
                passed = counts["passed"]
                total = counts["total"]
                return passed / total if total > 0 else 0.0, resp
            except (json.JSONDecodeError, KeyError, IndexError):
                # Fallback: if we can't parse, treat as all-fail
                return 0.0, resp
    except Exception as e:
        return 0.0, {"error": str(e)}


class CodeEnv:
    def __init__(
        self,
        problem: str,
        tests: list[dict[str, Any]],
        renderer: Renderer,
        convo_prefix: list[Message] | None = None,
        format_coef: float = 0.1,
        reward_timeout: int = 6
    ):
        self.renderer = renderer
        self.convo_prefix = convo_prefix or []
        self.format_coef = format_coef
        self.problem = problem
        self.tests = tests
        self.reward_timeout = reward_timeout

    @property
    def stop_condition(self) -> list[str] | list[int]:
        return self.renderer.get_stop_sequences()

    def get_question(self) -> str:
        return self.problem

    def check_format(self, sample_str: str) -> bool:
        return self.extract_code_from_model(sample_str) is not None

    def check_answer(self, sample_str: str) -> bool:
        """Not used - CodeEnv uses async check_sandbox_correctness instead."""
        return False

    def get_reference_answer(self) -> str:
        return ""

    async def step(self, action: Action) -> StepResult:
        message, parse_success = self.renderer.parse_response(action)
        content = get_text_content(message)
        format_ok_bool = bool(parse_success) and self.check_format(content)

        code = self.extract_code_from_model(content)
        if code is None:
            correct_answer_bool = False
        else:
            correct_answer_bool, _ = await sandbox_check_correctness(
                self.tests,
                code,
                timeout=self.reward_timeout
            )

        format_score = float(format_ok_bool) - 1 # 0 if format is ok, or -1 otherwise
        correct_score = float(correct_answer_bool)
        total_reward = (self.format_coef * format_score) + correct_score

        return StepResult(
            reward=total_reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "format": format_score,
                "correct": correct_score
            }
        )

    def extract_code_from_model(self, model_response: str) -> str | None:
        """
        Extract the last fenced code block from a model response.
        """
        code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", model_response, re.DOTALL)
        if not code_blocks:
            return None
        return code_blocks[-1].strip()
