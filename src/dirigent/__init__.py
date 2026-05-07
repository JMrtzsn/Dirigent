"""Dirigent — Agent-to-agent interaction framework for LLM-native systems."""

from dirigent.core import (
    Agent,
    AgentInfo,
    Capability,
    Channel,
    DelegateRequest,
    DelegateResponse,
    Feedback,
    InProcessChannel,
    Message,
    MessageKind,
    Registry,
    agent,
)
from dirigent.llm import CopilotLLMProvider, LLMAgent, LLMProvider, LLMResponse
from dirigent.runtime import Runtime, RuntimeConfig

__all__ = [
    "Agent",
    "AgentInfo",
    "Capability",
    "Channel",
    "DelegateRequest",
    "DelegateResponse",
    "Feedback",
    "InProcessChannel",
    "CopilotLLMProvider",
    "LLMAgent",
    "LLMProvider",
    "LLMResponse",
    "Message",
    "MessageKind",
    "Registry",
    "Runtime",
    "RuntimeConfig",
    "agent",
]
