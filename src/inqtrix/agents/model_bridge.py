"""Bridge from the Inqtrix ``LLMProvider`` to a LangChain chat model.

The deepagents harness (:mod:`inqtrix.agents.harness`) needs a LangChain
``BaseChatModel``; Inqtrix providers expose ``complete()``. This bridge
adapts one to the other WITHOUT importing any provider SDK — it simply
flattens the chat messages into the provider's (system, prompt) shape and
returns the completion as an AI message. Tool binding is accepted but the
tools are surfaced to the model as prompt context only (the harness gives
its sub-agents read-only tools whose results are injected as messages).

Import guard: LangChain lives behind the optional ``agent`` extra;
importing this module without it raises the loud install hint (E10 — one
file owns the breakage surface).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from inqtrix.agents.patterns._structured import observe_bound_model_retries
from inqtrix.constants import REASONING_TIMEOUT
if TYPE_CHECKING:
    from inqtrix.providers.base import LLMProvider

_IMPORT_HINT = (
    "Der Workspace-Agent braucht das 'agent'-Extra "
    "(uv sync --extra agent)."
)


def build_chat_model(
    llm: "LLMProvider",
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
    timeout: float = REASONING_TIMEOUT,
    node: str = "agent_file_analysis",
) -> Any:
    """Wrap *llm* as a LangChain ``BaseChatModel`` for the harness.

    Args:
        llm: The Inqtrix provider carrying the actual API access.
        model: Optional provider model override (tier resolution result).
        reasoning_effort: Optional effort override, forwarded verbatim.
        timeout: Per-call timeout in seconds.
        node: Agent node used for retry telemetry projection.

    Raises:
        RuntimeError: LangChain is not installed (missing ``agent`` extra).
    """
    try:
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage, BaseMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise RuntimeError(_IMPORT_HINT) from exc

    class _InqtrixChatModel(BaseChatModel):
        """LangChain adapter over one Inqtrix ``LLMProvider``."""

        @property
        def _llm_type(self) -> str:
            return "inqtrix-provider-bridge"

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> ChatResult:
            system_parts: list[str] = []
            prompt_parts: list[str] = []
            for message in messages:
                role = getattr(message, "type", "human")
                text = _message_text(message)
                if role == "system":
                    system_parts.append(text)
                elif role == "ai":
                    prompt_parts.append(f"Assistent: {text}")
                elif role == "tool":
                    prompt_parts.append(f"Werkzeug-Ergebnis: {text}")
                else:
                    prompt_parts.append(text)
            with observe_bound_model_retries(llm, node):
                completion = llm.complete(
                    "\n\n".join(prompt_parts),
                    system="\n\n".join(system_parts) or None,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    timeout=timeout,
                )
            answer = (
                completion
                if isinstance(completion, str)
                else getattr(completion, "text", str(completion))
            )
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=answer))]
            )

        def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
            # Tool calls are not emitted by the bridged provider; the
            # harness injects tool results as messages instead. Binding
            # is accepted so deepagents' setup path does not break.
            return self

    def _message_text(message: Any) -> str:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)

    return _InqtrixChatModel()
