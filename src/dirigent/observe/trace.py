"""Message trace — records the full conversation between agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """Single event in an agent interaction trace."""

    timestamp: datetime
    sender: str
    receiver: str
    message_kind: str
    message_id: str
    correlation_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class Trace:
    """Ordered log of all message events in a runtime session."""

    def __init__(self) -> None:
        self._events: list[TraceEvent] = []

    def record(
        self,
        *,
        sender: str,
        receiver: str,
        message_kind: str,
        message_id: str,
        correlation_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._events.append(
            TraceEvent(
                timestamp=datetime.now(UTC),
                sender=sender,
                receiver=receiver,
                message_kind=message_kind,
                message_id=message_id,
                correlation_id=correlation_id,
                metadata=metadata or {},
            )
        )

    @property
    def events(self) -> list[TraceEvent]:
        return list(self._events)

    def to_list(self) -> list[dict[str, Any]]:
        """Serialize all events to a list of dicts."""
        return [
            {
                "timestamp": e.timestamp.isoformat(),
                "sender": e.sender,
                "receiver": e.receiver,
                "message_kind": e.message_kind,
                "message_id": e.message_id,
                "correlation_id": e.correlation_id,
                "metadata": e.metadata,
            }
            for e in self._events
        ]
