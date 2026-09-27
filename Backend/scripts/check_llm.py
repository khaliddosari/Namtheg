"""Verify each configured LLM provider end to end, including a tool-calling
round trip (the path the analyst agent depends on).

    cd Backend
    python -m scripts.check_llm

Exit code 0 only if every configured provider passes. Costs a fraction of a cent.
"""
import sys

from app.config import settings
from app.llm import ToolSpec, _OpenAIResponses, _OpenRouterChat

TOOL = ToolSpec(
    name="count_rows",
    description="Return the number of rows in the dataset.",
    parameters={"type": "object", "properties": {}, "required": []},
)
SYSTEM = "You answer questions about a dataset. Use tools to get facts; never guess."
QUESTION = "How many rows does the dataset have? Reply with just the number."
ROWS = "48213"


def check(provider) -> bool:
    label = f"{provider.provider}/{provider.model}"
    try:
        transcript = [{"role": "user", "content": QUESTION}]
        first = provider.complete(SYSTEM, transcript, [TOOL], None, "low")
        if not first.tool_calls:
            print(f"FAIL {label}: expected a tool call, got text {first.text!r}")
            return False
        transcript.append({"role": "assistant", "turn": first})
        transcript += [{"role": "tool", "tool_call_id": tc.id, "content": ROWS} for tc in first.tool_calls]
        final = provider.complete(SYSTEM, transcript, [TOOL], None, "low")
    except Exception as e:
        print(f"FAIL {label}: {e}")
        return False
    ok = ROWS in final.text.replace(",", "")
    print(f"{'PASS' if ok else 'FAIL'} {label} (answered by {final.model}): {final.text.strip()[:120]!r}")
    return ok


def main() -> int:
    providers = []
    if settings.openai_api_key:
        providers.append(_OpenAIResponses())
    else:
        print("SKIP primary: OPENAI_API_KEY is not set")
    if settings.openrouter_api_key:
        providers.append(_OpenRouterChat())
    else:
        print("SKIP fallback: OPENROUTER_API_KEY is not set")
    if not providers:
        return 1
    return 0 if all([check(p) for p in providers]) else 1


if __name__ == "__main__":
    sys.exit(main())
