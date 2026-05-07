"""Typed message envelope for agent-to-agent communication.

Messages are serializable dataclasses — no shared memory assumptions.
Designed to cross a network boundary without modification.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class MessageKind(StrEnum):
    """Discriminator for message routing."""

    DELEGATE_REQUEST = "delegate_request"
    DELEGATE_RESPONSE = "delegate_response"
    FEEDBACK = "feedback"


@dataclass(frozen=True, slots=True)
class Message:
    """Base envelope shared by all message types."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    sender: str = ""
    receiver: str = ""
    correlation_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (network-ready)."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "sender": self.sender,
            "receiver": self.receiver,
            "correlation_id": self.correlation_id,
        }


@dataclass(frozen=True, slots=True)
class DelegateRequest(Message):
    """Agent A asks agent B (or a capability) to perform work."""

    kind: MessageKind = field(default=MessageKind.DELEGATE_REQUEST, init=False)
    capability: str = ""
    task: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            "kind": self.kind.value,
            "capability": self.capability,
            "task": self.task,
            "context": self.context,
            "constraints": self.constraints,
        })
        return base


@dataclass(frozen=True, slots=True)
class DelegateResponse(Message):
    """Result returned from a delegated task."""

    kind: MessageKind = field(default=MessageKind.DELEGATE_RESPONSE, init=False)
    success: bool = True
    result: Any = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            "kind": self.kind.value,
            "success": self.success,
            "result": self.result,
            "error": self.error,
        })
        return base


@dataclass(frozen=True, slots=True)
class Feedback(Message):
    """Structured critique sent back to an agent for revision."""

    kind: MessageKind = field(default=MessageKind.FEEDBACK, init=False)
    accepted: bool = False
    comments: list[str] = field(default_factory=list)
    revision_hints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        base = super().to_dict()
        base.update({
            "kind": self.kind.value,
            "accepted": self.accepted,
            "comments": self.comments,
            "revision_hints": self.revision_hints,
        })
        return base
