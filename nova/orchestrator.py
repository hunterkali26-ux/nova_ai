"""Hybrid AI orchestrator: connects user input, model, knowledge retrieval,
prompt construction, inference, and conversation persistence.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import List, Optional

from .core.logging import get_logger
from .database.models import ConversationStore, MessageStore
from .runtime.model_manager import ModelManager

logger = get_logger("orchestrator")


@dataclass
class ChatResponse:
    text: str
    model: str
    sources: List[dict]
    conversation_id: str
    rag_used: bool


# Model-specific prompt templates. NOVA Tiny is not instruction-tuned, so we
# keep the template neutral and let the model complete naturally.
PROMPT_TEMPLATES = {
    "nova-tiny": "{context}{user_input}",
    "default": "{context}{user_input}",
}


class Orchestrator:
    def __init__(
        self,
        model_manager: ModelManager,
        conversation_store: ConversationStore,
        message_store: MessageStore,
        rag_pipeline=None,
    ) -> None:
        self.model_manager = model_manager
        self.conversation_store = conversation_store
        self.message_store = message_store
        self.rag_pipeline = rag_pipeline

    def _default_model_name(self) -> str:
        active = self.model_manager.active_model_name()
        if active:
            return active
        models = self.model_manager.list_models()
        return models[0]["name"] if models else "nova-tiny"

    def chat(
        self,
        user_input: str,
        model_name: Optional[str] = None,
        conversation_id: Optional[str] = None,
        rag_mode: str = "auto",  # "auto", "enabled", "disabled"
    ) -> ChatResponse:
        model_name = model_name or self._default_model_name()

        if conversation_id is None:
            conversation_id = str(uuid.uuid4())
            self.conversation_store.create(
                conversation_id, "New Conversation", model_name, rag_enabled=(rag_mode != "disabled")
            )
        else:
            self.conversation_store.update_touch(conversation_id)

        # Persist user message.
        user_msg_id = str(uuid.uuid4())
        self.message_store.add(user_msg_id, conversation_id, "user", user_input)

        # Optional RAG retrieval.
        sources: List[dict] = []
        context_str = ""
        rag_used = False
        if self.rag_pipeline is not None and rag_mode in ("auto", "enabled"):
            rag_ctx = self.rag_pipeline.build_context(user_input)
            if rag_ctx.has_sufficient_context:
                context_str = rag_ctx.context
                sources = rag_ctx.sources
                rag_used = True

        # Build model-specific prompt.
        template = PROMPT_TEMPLATES.get("nova-tiny", PROMPT_TEMPLATES["default"])
        prompt = template.format(context=(context_str + "\n\n" if context_str else ""), user_input=user_input)

        # Generate.
        backend = self.model_manager.get_active_backend()
        if backend is None or not backend.is_loaded():
            # Try to load the requested model.
            loaded = self.model_manager.load_model(model_name)
            backend = loaded
        assert backend is not None
        result = backend.generate(prompt)

        assistant_text = result.text
        if not rag_used and self.rag_pipeline is not None and rag_mode == "enabled":
            assistant_text = (
                "I don't have supporting documents in my knowledge base for that. "
                + assistant_text
            )

        # Persist assistant message.
        assistant_msg_id = str(uuid.uuid4())
        self.message_store.add(
            assistant_msg_id, conversation_id, "assistant", assistant_text, sources=sources
        )

        logger.info("Chat response: model=%s rag=%s sources=%d", model_name, rag_used, len(sources))
        return ChatResponse(
            text=assistant_text,
            model=model_name,
            sources=sources,
            conversation_id=conversation_id,
            rag_used=rag_used,
        )

    def stream_chat(self, *args, **kwargs):
        """Non-streaming wrapper; yields the full response once."""
        response = self.chat(*args, **kwargs)
        yield response

    def get_conversation(self, conversation_id: str):
        conv = self.conversation_store.get(conversation_id)
        messages = self.message_store.list_for_conversation(conversation_id)
        return conv, messages
