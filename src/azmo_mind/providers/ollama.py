from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from typing import Any

import httpx
from pydantic import ValidationError

from azmo_mind.config import ProviderConfig
from azmo_mind.providers.base import (
    LLMProvider,
    ProviderError,
    ProviderResult,
    SpeechStream,
)
from azmo_mind.schemas import AzmoResponse, coerce_response_payload, salvage_embedded_fields
from azmo_mind.streaming import SpeechFieldStreamer

_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE)


def _duration_ms(data: dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    if isinstance(value, (int, float)):
        # Ollama reports durations in nanoseconds.
        return round(float(value) / 1_000_000, 2)
    return None


def _extract_json_object(raw: str) -> str:
    """Recover a JSON object if a model adds a fence or small amount of prose."""
    candidate = raw.strip()
    fence = _JSON_FENCE_RE.match(candidate)
    if fence:
        candidate = fence.group(1).strip()

    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(candidate):
        if char != "{":
            continue
        try:
            _, end = decoder.raw_decode(candidate[index:])
            return candidate[index : index + end]
        except json.JSONDecodeError:
            continue
    return candidate


class OllamaProvider(LLMProvider):
    def __init__(self, config: ProviderConfig):
        self.config = config
        timeout = httpx.Timeout(
            connect=10.0,
            read=config.timeout_seconds,
            write=30.0,
            pool=10.0,
        )
        self.client = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            timeout=timeout,
        )

    def _chat_payload(self, messages: list[dict[str, str]], stream: bool) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "messages": messages,
            "stream": stream,
            "format": AzmoResponse.model_json_schema(),
            "think": self.config.think,
            "keep_alive": self.config.keep_alive,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "repeat_penalty": self.config.repeat_penalty,
                "num_ctx": self.config.context_tokens,
                "num_predict": self.config.max_output_tokens,
            },
        }

    def _timeout_error(self, exc: Exception) -> ProviderError:
        return ProviderError(
            f"Ollama did not finish within {self.config.timeout_seconds:.0f} seconds. "
            "The model may still be loading, or GPU inference may have stalled."
        )

    def _connect_error(self, exc: Exception) -> ProviderError:
        return ProviderError(
            "Could not connect to Ollama at "
            f"{self.config.base_url}. Confirm that Ollama is running."
        )

    def _finalize(self, raw: str, data: dict[str, Any], started: float) -> ProviderResult:
        """Validate a completed document into a turn.

        Shared by the streaming and non-streaming paths so the repair rules -
        which exist because local models violate numeric ranges under structured
        output - cannot drift apart between them.
        """
        normalized = _extract_json_object(raw)
        # Repair recoverable violations (out-of-range numbers, unknown enums)
        # before strict validation. Ollama's grammar enforces JSON shape and
        # enum tokens but not numeric min/max, so a value like
        # subharmonic_mix=1.5 would otherwise fail the entire turn.
        repairs: list[str] = []
        try:
            payload: Any = json.loads(normalized)
        except json.JSONDecodeError:
            payload = None

        try:
            if isinstance(payload, dict):
                payload = salvage_embedded_fields(payload, raw)
                payload, repairs = coerce_response_payload(payload)
                parsed = AzmoResponse.model_validate(payload)
            else:
                parsed = AzmoResponse.model_validate_json(normalized)
        except ValidationError as exc:
            preview = raw[:500].replace("\n", " ")
            raise ProviderError(
                "Ollama answered, but the response did not match AZMO's structured schema. "
                f"Response preview: {preview!r}. Validation error: {exc}"
            ) from exc

        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        metrics = {
            "elapsed_ms": elapsed_ms,
            "total_duration_ms": _duration_ms(data, "total_duration"),
            "load_duration_ms": _duration_ms(data, "load_duration"),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "prompt_eval_duration_ms": _duration_ms(data, "prompt_eval_duration"),
            "eval_count": data.get("eval_count"),
            "eval_duration_ms": _duration_ms(data, "eval_duration"),
            "done_reason": data.get("done_reason"),
            "repairs": repairs,
        }
        return ProviderResult(response=parsed, raw_content=raw, metrics=metrics)

    def generate(self, messages: list[dict[str, str]]) -> ProviderResult:
        started = time.perf_counter()
        try:
            response = self.client.post("/api/chat", json=self._chat_payload(messages, False))
            response.raise_for_status()
            data = response.json()
            raw = str(data["message"]["content"])
        except httpx.ReadTimeout as exc:
            raise self._timeout_error(exc) from exc
        except httpx.ConnectError as exc:
            raise self._connect_error(exc) from exc
        except (httpx.HTTPError, KeyError, json.JSONDecodeError) as exc:
            raise ProviderError(f"Ollama request failed: {exc}") from exc

        return self._finalize(raw, data, started)

    def generate_stream(self, messages: list[dict[str, str]]) -> SpeechStream:
        """Yield his words as the model writes them, then the finished turn.

        Ollama streams the *JSON document*, not prose, because AZMO uses
        structured output. ``SpeechFieldStreamer`` decodes the ``speech`` value
        out of that document as it is written, which is why the text can reach
        XTTS long before the gesture and voice fields exist.

        The response is still accumulated in full and validated by the same
        ``_finalize`` the blocking path uses. Streaming changes *when* the words
        are available, never *what* the turn is - memory, emotional state and
        motion all still act on one validated document.
        """

        def factory(stream: SpeechStream) -> Iterator[str]:
            started = time.perf_counter()
            speech = SpeechFieldStreamer()
            pieces: list[str] = []
            final: dict[str, Any] = {}
            try:
                with self.client.stream(
                    "POST", "/api/chat", json=self._chat_payload(messages, True)
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line.strip():
                            continue
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        fragment = str(event.get("message", {}).get("content", ""))
                        if fragment:
                            pieces.append(fragment)
                            delta = speech.feed(fragment)
                            if delta:
                                yield delta
                        if event.get("done"):
                            final = event
            except httpx.ReadTimeout as exc:
                raise self._timeout_error(exc) from exc
            except httpx.ConnectError as exc:
                raise self._connect_error(exc) from exc
            except (httpx.HTTPError, KeyError) as exc:
                raise ProviderError(f"Ollama request failed: {exc}") from exc

            stream.complete(self._finalize("".join(pieces), final, started))

        return SpeechStream(factory)

    def warmup(self) -> dict[str, Any]:
        # num_ctx MUST match generate(). Ollama keys a resident model on its
        # runtime options, so warming at 4096 and then generating at 8192 makes
        # it evict and fully reload the model on the first real turn - the exact
        # opposite of warming up. On a 9B model that is ~6 s and a multi-GB VRAM
        # allocation spike per session, and it made "model load" appear in the
        # metrics of every single turn.
        payload = {
            "model": self.config.model,
            "prompt": "Reply with exactly READY.",
            "stream": False,
            "think": False,
            "keep_alive": self.config.keep_alive,
            "options": {
                "temperature": 0,
                "num_ctx": self.config.context_tokens,
                "num_predict": 8,
            },
        }
        started = time.perf_counter()
        try:
            response = self.client.post("/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.ReadTimeout as exc:
            raise ProviderError(
                f"The model warm-up exceeded {self.config.timeout_seconds:.0f} seconds."
            ) from exc
        except httpx.ConnectError as exc:
            raise ProviderError(
                f"Could not connect to Ollama at {self.config.base_url}."
            ) from exc
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise ProviderError(f"Ollama warm-up failed: {exc}") from exc

        return {
            "ok": True,
            "response": str(data.get("response", "")).strip(),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "load_duration_ms": _duration_ms(data, "load_duration"),
            "eval_count": data.get("eval_count"),
        }

    def health(self) -> dict[str, object]:
        try:
            response = self.client.get("/api/tags", timeout=5)
            response.raise_for_status()
            models = [str(m.get("name", "")) for m in response.json().get("models", [])]
            model_present = any(
                name == self.config.model
                or name.startswith(self.config.model + ":")
                or self.config.model.startswith(name + ":")
                for name in models
            )

            loaded_models: list[str] = []
            try:
                ps_response = self.client.get("/api/ps", timeout=5)
                ps_response.raise_for_status()
                loaded_models = [
                    str(m.get("name", "")) for m in ps_response.json().get("models", [])
                ]
            except Exception:
                pass

            return {
                "ok": True,
                "base_url": self.config.base_url,
                "configured_model": self.config.model,
                "model_present": model_present,
                "model_loaded": self.config.model in loaded_models,
                "models": models,
                "loaded_models": loaded_models,
            }
        except Exception as exc:
            return {
                "ok": False,
                "base_url": self.config.base_url,
                "configured_model": self.config.model,
                "error": str(exc),
            }
