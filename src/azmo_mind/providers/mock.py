from __future__ import annotations

from azmo_mind.providers.base import LLMProvider, ProviderResult
from azmo_mind.schemas import AzmoResponse, GestureCommand, VoiceDirection


class MockProvider(LLMProvider):
    def generate(self, messages: list[dict[str, str]]) -> ProviderResult:
        user_text = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        response = AzmoResponse(
            speech=f"Nephalem, your words have reached the throne: {user_text[:100]}",
            emotion="commanding",
            emotional_intensity=0.45,
            gesture=GestureCommand(
                name="loom",
                intensity=0.30,
                duration_ms=1200,
                target="speaker",
            ),
            voice=VoiceDirection(preset="close_ominous", pace=0.90),
            internal_note="Mock provider response.",
        )
        return ProviderResult(
            response=response,
            raw_content=response.model_dump_json(),
            metrics={"elapsed_ms": 1.0, "provider": "mock"},
        )

    def warmup(self) -> dict[str, object]:
        return {"ok": True, "elapsed_ms": 0.0, "provider": "mock"}

    def health(self) -> dict[str, object]:
        return {"ok": True, "provider": "mock"}
