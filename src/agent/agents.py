from abc import ABC, abstractmethod
import os
import time

from langchain_openai import ChatOpenAI
from openai import RateLimitError, APIError, APITimeoutError, APIConnectionError

from .prompts import construct_message


def _invoke_with_retry(llm, messages, max_retries: int = 3, base_delay: float = 3.0):
    """Invoke an LLM with exponential backoff on rate limits / transient API errors.

    Honors OpenRouter's `retry_after_seconds` hint when present.
    Does NOT retry on permanent 4xx errors (auth, not found, bad request).
    """
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except RateLimitError as e:
            # 429 — retry with backoff (and upstream hint if any)
            retry_after = base_delay * (2 ** attempt)
            try:
                body = getattr(e, "body", None) or {}
                meta = (body.get("error", {}) or {}).get("metadata", {}) or {}
                hint = meta.get("retry_after_seconds")
                if hint:
                    retry_after = max(retry_after, float(hint) + 2)
            except Exception:
                pass
            if attempt == max_retries - 1:
                raise
            print(f"[Rate limit] Retrying in {retry_after:.0f}s "
                  f"(attempt {attempt + 1}/{max_retries})", flush=True)
            time.sleep(retry_after)
        except (APITimeoutError, APIConnectionError) as e:
            # Transient network issues — retry with backoff
            delay = base_delay * (2 ** attempt)
            if attempt == max_retries - 1:
                raise
            print(f"[Transient error] {type(e).__name__}: {e}. "
                  f"Retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})",
                  flush=True)
            time.sleep(delay)
        except APIError as e:
            # Permanent errors (401, 403, 404, 400, etc.) — fail fast, don't retry
            status = getattr(e, "status_code", None) or getattr(e, "code", None)
            if status and 400 <= int(status) < 500 and int(status) != 429:
                print(f"[Fatal API error] {type(e).__name__} ({status}): {e}", flush=True)
                raise
            # Treat 5xx and unknown status as transient
            delay = base_delay * (2 ** attempt)
            if attempt == max_retries - 1:
                raise
            print(f"[Server error] {type(e).__name__}: {e}. "
                  f"Retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})",
                  flush=True)
            time.sleep(delay)
    raise RuntimeError("Exhausted LLM retries")


class Agent(ABC):
    @abstractmethod
    def get_response(self, task: dict) -> list[str]:
        pass


class BaseAgent(Agent):
    def __init__(self, agent_config: dict):
        self.agent_config = agent_config
        self._setup()

    def _setup(self):
        # vLLM was launched with: --api-key EMPTY
        # Some config files may still contain "None" or "dummy"; normalize them.
        api_key = self.agent_config.get("api_key") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        if str(api_key).lower() in {"none", "null", "dummy"}:
            api_key = "EMPTY"

        self.llm = ChatOpenAI(
            model=self.agent_config["model"],
            base_url=self.agent_config["base_url"],
            temperature=self.agent_config.get("temperature", 0.0),
            top_p=self.agent_config.get("top_p", 0.9),
            api_key=api_key,
            # Aggressive timeout — a single LLM call should never block the
            # main loop for more than this. Combined with per-example timeout
            # in run_2wikiqa.py, prevents the whole run from hanging on one
            # stuck request.
            timeout=45,
            max_retries=0,  # disable langchain's internal retries; we do our own
            max_tokens=self.agent_config.get("max_new_tokens", 1000),
        )
        self.num_samples = self.agent_config.get("num_samples", 1)


class RoleAgent(BaseAgent):
    """
    A role-specific agent for 2WikiMultiHopQA.

    Examples of prompt_type:
    - single_agent
    - decomposer
    - retriever
    - verifier
    - composer
    - multi_agent_unstructured
    """

    def __init__(self, agent_config: dict, prompt_type: str):
        super().__init__(agent_config)
        self.prompt_type = prompt_type

    def get_response(self, task: dict) -> list[str]:
        messages = construct_message(
            prompt_type=self.prompt_type,
            trajectories=task["input_info"]["trajectories"],
            few_shot_examples=task.get("few_shot_examples", None),
            system_prompt=task.get("system_prompt", None),
        )

        responses = []
        for _ in range(self.num_samples):
            response = _invoke_with_retry(self.llm, messages)
            responses.append(response.content)

        return responses


class SingleAgent(RoleAgent):
    def __init__(self, agent_config: dict):
        super().__init__(agent_config, prompt_type="single_agent")
