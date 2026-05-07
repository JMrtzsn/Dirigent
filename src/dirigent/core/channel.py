"""Message channels — transport layer between agents.

InProcessChannel uses asyncio queues. The interface is designed so that
a network-backed channel (WebSocket, gRPC, etc.) can be swapped in
without changing agent code.
"""

from __future__ import annotations

import asyncio
from typing import Protocol

from dirigent.core.message import Message


class Channel(Protocol):
    """Abstract transport for delivering messages between agents."""

    async def send(self, message: Message) -> None: ...

    async def receive(self) -> Message: ...


class InProcessChannel:
    """Async queue-backed channel for single-process execution."""

    def __init__(self, max_size: int = 0) -> None:
        self._queue: asyncio.Queue[Message] = asyncio.Queue(maxsize=max_size)

    async def send(self, message: Message) -> None:
        await self._queue.put(message)

    async def receive(self) -> Message:
        return await self._queue.get()

    @property
    def pending(self) -> int:
        return self._queue.qsize()
