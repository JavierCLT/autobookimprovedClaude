"""Thin wrapper around the Anthropic Messages API with multi-model support."""

from __future__ import annotations

import json
import logging
import time
from typing import Callable, TypeVar

import anthropic
from pydantic import BaseModel

from novel_factory.config import AppConfig

logger = logging.getLogger(__name__)
SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LlmRequestError(RuntimeError):
    """Raised when an LLM request fails after retries."""


class AnthropicClient:
    """Encapsulates the Anthropic SDK for text and structured calls.

    Supports multi-model routing: different models for drafting vs QA.
    Uses extended thinking for high-effort reasoning tasks.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = anthropic.Anthropic(
            api_key=config.api_key,
            timeout=config.request_timeout_seconds,
        )

    def text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        task_name: str,
        reasoning_effort: str,
        temperature: float,
        max_output_tokens: int = 5_000,
        model_override: str = "",
    ) -> str:
        """Generates free-form text using the Anthropic Messages API."""

        model = model_override or self.config.model

        def _request() -> str:
            request_kwargs = self._build_request_kwargs(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
            )

            response = self.client.messages.create(**request_kwargs)
            output_text = self._extract_text_from_response(response)
            if not output_text:
                raise LlmRequestError(f"Empty response body for task '{task_name}'.")
            return output_text

        return self._with_retries(task_name=task_name, callback=_request)

    def structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[SchemaT],
        task_name: str,
        reasoning_effort: str,
        temperature: float,
        max_output_tokens: int = 4_000,
        verbosity: str = "medium",
        model_override: str = "",
    ) -> SchemaT:
        """Generates structured output and parses it into a Pydantic model.

        Uses Anthropic's native structured output (messages.parse) when available,
        with JSON fallback for robustness.
        """

        model = model_override or self.config.model

        def _request() -> SchemaT:
            # Try native structured output first
            try:
                return self._structured_native(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    schema=schema,
                    task_name=task_name,
                    reasoning_effort=reasoning_effort,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
            except Exception as parse_error:  # noqa: BLE001
                logger.warning(
                    "LLM task %s native parse failed; using JSON fallback: %s",
                    task_name,
                    parse_error,
                )
                return self._structured_json_fallback(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    schema=schema,
                    task_name=task_name,
                    reasoning_effort=reasoning_effort,
                    temperature=temperature,
                    max_output_tokens=max(max_output_tokens, 5_000),
                )

        return self._with_retries(task_name=task_name, callback=_request)

    # ── Native structured output ─────────────────────────────────────

    def _structured_native(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema: type[SchemaT],
        task_name: str,
        reasoning_effort: str,
        temperature: float,
        max_output_tokens: int,
    ) -> SchemaT:
        """Uses Anthropic's messages.parse() for native Pydantic structured output."""
        request_kwargs = self._build_request_kwargs(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )

        # Use the parse method with output_format for native structured output
        response = self.client.messages.parse(
            **request_kwargs,
            output_format=schema,
        )

        parsed = response.parsed_output
        if parsed is None:
            raise LlmRequestError(f"Empty parsed body for task '{task_name}'.")
        return parsed

    # ── JSON fallback ────────────────────────────────────────────────

    def _structured_json_fallback(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema: type[SchemaT],
        task_name: str,
        reasoning_effort: str,
        temperature: float,
        max_output_tokens: int,
    ) -> SchemaT:
        """Falls back to asking for raw JSON and parsing manually."""
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=True, separators=(",", ":"))
        fallback_prompt = (
            f"{user_prompt}\n\n"
            "Return only a valid JSON object. Do not wrap it in markdown fences. "
            "Every required field must be present.\n"
            f"JSON schema:\n{schema_json}"
        )
        fallback_system = f"{system_prompt}\n\nReturn only a valid JSON object with no surrounding prose."

        request_kwargs = self._build_request_kwargs(
            model=model,
            system_prompt=fallback_system,
            user_prompt=fallback_prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )

        response = self.client.messages.create(**request_kwargs)
        output_text = self._extract_text_from_response(response)
        if not output_text:
            raise LlmRequestError(f"Empty JSON fallback body for task '{task_name}'.")
        return schema.model_validate_json(self._extract_json_object(output_text))

    # ── Request building ─────────────────────────────────────────────

    def _build_request_kwargs(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
        temperature: float,
        reasoning_effort: str,
    ) -> dict:
        """Builds the kwargs dict for Anthropic API calls."""
        kwargs: dict = {
            "model": model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": max_output_tokens,
        }

        # Extended thinking for high-effort tasks
        use_thinking = reasoning_effort in ("high",)
        if use_thinking:
            # When using extended thinking, temperature must be 1 and
            # we need a thinking budget within max_tokens
            thinking_budget = min(max_output_tokens, 8_000)
            kwargs["max_tokens"] = max_output_tokens + thinking_budget
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
            kwargs["temperature"] = 1  # Required when thinking is enabled
        else:
            kwargs["temperature"] = temperature

        return kwargs

    # ── Response parsing ─────────────────────────────────────────────

    def _extract_text_from_response(self, response) -> str:
        """Extracts text content from an Anthropic response, skipping thinking blocks."""
        parts: list[str] = []
        for block in response.content:
            if block.type == "text":
                parts.append(block.text)
        return "\n".join(parts).strip()

    # ── Retry logic ──────────────────────────────────────────────────

    def _with_retries(self, *, task_name: str, callback: Callable[[], SchemaT | str]) -> SchemaT | str:
        attempts = self.config.retry_attempts
        base_delay = self.config.retry_base_delay_seconds
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                logger.info("LLM task %s attempt %s/%s", task_name, attempt, attempts)
                return callback()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= attempts:
                    break
                sleep_seconds = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "LLM task %s failed on attempt %s/%s: %s",
                    task_name,
                    attempt,
                    attempts,
                    exc,
                )
                time.sleep(sleep_seconds)

        raise LlmRequestError(f"LLM task '{task_name}' failed after {attempts} attempts.") from last_error

    # ── Helpers ───────────────────────────────────────────────────────

    def _extract_json_object(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if stripped.lower().startswith("json"):
                stripped = stripped[4:].strip()
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise LlmRequestError("JSON fallback did not return a parseable JSON object.")
        return stripped[start : end + 1]


# Backward-compatible alias
OpenAIResponsesClient = AnthropicClient
