"""Core primitives for agent-to-agent interaction."""

from dirigent.core.agent import Agent, AgentInfo, agent
from dirigent.core.capability import Capability
from dirigent.core.channel import Channel, InProcessChannel
from dirigent.core.message import (
    DelegateRequest,
    DelegateResponse,
    Feedback,
    Message,
    MessageKind,
)
from dirigent.core.registry import Registry

__all__ = [
    "Agent",
    "AgentInfo",
    "Capability",
    "Channel",
    "DelegateRequest",
    "DelegateResponse",
    "Feedback",
    "InProcessChannel",
    "Message",
    "MessageKind",
    "Registry",
    "agent",
]
