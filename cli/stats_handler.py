import threading
from typing import Any, Dict, List, Union

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import AIMessage


class StatsCallbackHandler(BaseCallbackHandler):
    """Callback handler that tracks LLM calls, tool calls, and token usage."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.llm_calls = 0
        self.tool_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.cache_read = 0

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        """Increment LLM call counter when an LLM starts."""
        with self._lock:
            self.llm_calls += 1

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        **kwargs: Any,
    ) -> None:
        """Increment LLM call counter when a chat model starts."""
        with self._lock:
            self.llm_calls += 1

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Extract token usage from LLM response."""
        try:
            generation = response.generations[0][0]
        except (IndexError, TypeError):
            return

        usage_metadata = None
        if hasattr(generation, "message"):
            message = generation.message
            if isinstance(message, AIMessage) and hasattr(message, "usage_metadata"):
                usage_metadata = message.usage_metadata

        if usage_metadata:
            with self._lock:
                self.tokens_in += usage_metadata.get("input_tokens", 0)
                self.tokens_out += usage_metadata.get("output_tokens", 0)
                # Extract cache read from input_token_details
                details = usage_metadata.get("input_token_details") or {}
                self.cache_read += details.get("cache_read", 0)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Increment tool call counter when a tool starts."""
        with self._lock:
            self.tool_calls += 1

    def get_stats(self) -> Dict[str, Any]:
        """Return current statistics."""
        with self._lock:
            return {
                "llm_calls": self.llm_calls,
                "tool_calls": self.tool_calls,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "cache_read": self.cache_read,
            }

    def get_cost(self, input_per_m: float = 6.0, output_per_m: float = 24.0,
                 cache_per_m: float = 1.3) -> Dict[str, float]:
        """Calculate cost in CNY based on per-million-token rates.

        Default: GLM-5.1 pricing (input <32k context).
        """
        with self._lock:
            regular_in = self.tokens_in - self.cache_read
            cost_in = regular_in * input_per_m / 1_000_000
            cost_cache = self.cache_read * cache_per_m / 1_000_000
            cost_out = self.tokens_out * output_per_m / 1_000_000
            total = cost_in + cost_cache + cost_out
            return {
                "input_cny": round(cost_in, 4),
                "cache_cny": round(cost_cache, 4),
                "output_cny": round(cost_out, 4),
                "total_cny": round(total, 4),
            }
