"""LLM access with a primary model and an automatic fallback.

Primary:  GPT-6 Sol via OpenAI's Responses API (the only OpenAI endpoint that
          allows function calling while reasoning is on).
Fallback: DeepSeek V4 Flash via OpenRouter's Chat Completions API.

Callers keep one provider-neutral transcript; each provider renders it into its
own wire format on every call. Calls are stateless on the provider side
(store=False, reasoning state round-trips as encrypted content), and a run can
move to the fallback mid-conversation without losing context. Once the primary
fails, an LLMClient stays on the fallback for the rest of the run instead of
paying the primary's timeout again on every call.

Neutral transcript entries:
    {"role": "user", "content": str}
    {"role": "assistant", "turn": AssistantTurn}
    {"role": "tool", "tool_call_id": str, "content": str}
"""
import json
import logging
import time
from dataclasses import dataclass, field

from openai import OpenAI

from app.config import settings

log = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """No configured provider produced a response."""


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict


@dataclass
class ToolCall:
    id: str
    name: str
    raw_arguments: str
    arguments: dict | None  # None when the model emitted invalid JSON


@dataclass
class AssistantTurn:
    text: str
    tool_calls: list[ToolCall]
    provider: str
    model: str
    # Raw output items, replayed verbatim only to the provider that made them
    # (carries OpenAI's encrypted reasoning between tool-calling turns).
    provider_items: list = field(default_factory=list)


def _parse_arguments(raw: str) -> dict | None:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


class _OpenAIResponses:
    provider = "openai"

    def __init__(self) -> None:
        self.model = settings.llm_primary_model
        self._client = OpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout_seconds,
            max_retries=2,
        )

    def _render(self, transcript: list[dict]) -> list:
        items: list = []
        for entry in transcript:
            role = entry["role"]
            if role == "user":
                items.append({"role": "user", "content": entry["content"]})
            elif role == "tool":
                items.append({
                    "type": "function_call_output",
                    "call_id": entry["tool_call_id"],
                    "output": entry["content"],
                })
            else:
                turn: AssistantTurn = entry["turn"]
                if turn.provider == self.provider and turn.provider_items:
                    items.extend(turn.provider_items)
                    continue
                if turn.text:
                    items.append({"role": "assistant", "content": turn.text})
                for tc in turn.tool_calls:
                    items.append({
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": tc.raw_arguments,
                    })
        return items

    def complete(self, system, transcript, tools, tool_choice, effort) -> AssistantTurn:
        kwargs: dict = {
            "model": self.model,
            "instructions": system,
            "input": self._render(transcript),
            "reasoning": {"effort": effort or settings.llm_primary_reasoning_effort},
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        if tools:
            kwargs["tools"] = [
                {"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters}
                for t in tools
            ]
        if tool_choice:
            kwargs["tool_choice"] = {"type": "function", "name": tool_choice}
        resp = self._client.responses.create(**kwargs)
        tool_calls = [
            ToolCall(id=item.call_id, name=item.name, raw_arguments=item.arguments,
                     arguments=_parse_arguments(item.arguments))
            for item in resp.output
            if item.type == "function_call"
        ]
        return AssistantTurn(
            text=resp.output_text or "",
            tool_calls=tool_calls,
            provider=self.provider,
            model=resp.model or self.model,
            provider_items=[item.model_dump(exclude_none=True) for item in resp.output],
        )


class _OpenRouterChat:
    provider = "openrouter"

    def __init__(self) -> None:
        self.model = settings.openrouter_model
        self._client = OpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=2,
            default_headers={
                "HTTP-Referer": settings.openrouter_referer,
                "X-Title": settings.openrouter_app_title,
            },
        )

    def _render(self, system: str, transcript: list[dict]) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": system}]
        for entry in transcript:
            role = entry["role"]
            if role == "user":
                messages.append({"role": "user", "content": entry["content"]})
            elif role == "tool":
                messages.append({"role": "tool", "tool_call_id": entry["tool_call_id"], "content": entry["content"]})
            else:
                turn: AssistantTurn = entry["turn"]
                msg: dict = {"role": "assistant", "content": turn.text or None}
                if turn.tool_calls:
                    msg["tool_calls"] = [
                        {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.raw_arguments}}
                        for tc in turn.tool_calls
                    ]
                messages.append(msg)
        return messages

    def complete(self, system, transcript, tools, tool_choice, effort) -> AssistantTurn:
        kwargs: dict = {
            "model": self.model,
            "messages": self._render(system, transcript),
            "temperature": 0.2,
        }
        if tools:
            kwargs["tools"] = [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
                for t in tools
            ]
        if tool_choice:
            kwargs["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}
        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, raw_arguments=tc.function.arguments or "{}",
                     arguments=_parse_arguments(tc.function.arguments))
            for tc in (msg.tool_calls or [])
        ]
        return AssistantTurn(
            text=msg.content or "",
            tool_calls=tool_calls,
            provider=self.provider,
            model=resp.model or self.model,
        )


class LLMClient:
    """Create one per run: it remembers a failed primary and records every call."""

    def __init__(self) -> None:
        self._primary = _OpenAIResponses() if settings.openai_api_key else None
        self._fallback = _OpenRouterChat() if settings.openrouter_api_key else None
        self._primary_failed = False
        self.calls: list[dict] = []

    @property
    def configured(self) -> bool:
        return self._primary is not None or self._fallback is not None

    def _providers(self) -> list:
        chain = []
        if self._primary is not None and not self._primary_failed:
            chain.append(self._primary)
        if self._fallback is not None:
            chain.append(self._fallback)
        return chain

    def complete(
        self,
        system: str,
        transcript: list[dict],
        tools: list[ToolSpec] | None = None,
        tool_choice: str | None = None,
        effort: str | None = None,
    ) -> AssistantTurn:
        """One model turn. `tool_choice` forces a call to that tool; `effort`
        overrides the primary's reasoning effort (the fallback ignores it)."""
        errors: list[str] = []
        for p in self._providers():
            started = time.monotonic()
            try:
                turn = p.complete(system, transcript, tools, tool_choice, effort)
            except Exception as e:
                log.warning("LLM call to %s (%s) failed: %s", p.provider, p.model, e)
                self.calls.append({"provider": p.provider, "model": p.model, "ok": False,
                                   "error": str(e)[:300], "seconds": round(time.monotonic() - started, 2)})
                errors.append(f"{p.provider}/{p.model}: {e}")
                if p is self._primary:
                    self._primary_failed = True
                continue
            self.calls.append({"provider": p.provider, "model": turn.model, "ok": True,
                               "seconds": round(time.monotonic() - started, 2)})
            return turn
        if not errors:
            raise LLMUnavailable("No LLM is configured. Set OPENAI_API_KEY and/or OPENROUTER_API_KEY.")
        raise LLMUnavailable("All LLM providers failed: " + " | ".join(errors))

    def text(self, system: str, user: str, effort: str | None = None) -> str:
        return self.complete(system, [{"role": "user", "content": user}], effort=effort).text.strip()

    def summary(self) -> dict:
        """What actually answered during this run, for the result's audit trail."""
        return {
            "primary": settings.llm_primary_model if self._primary else None,
            "fallback": settings.openrouter_model if self._fallback else None,
            "fell_back": self._primary_failed,
            "calls": len(self.calls),
            "failed_calls": sum(1 for c in self.calls if not c["ok"]),
            "models_used": sorted({c["model"] for c in self.calls if c["ok"]}),
        }
