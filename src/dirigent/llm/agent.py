"""LLM-backed agent — base class for agents that use an LLM to handle requests.

Subclass this and override `build_messages` to control prompting.
Override `parse_response` to extract structured data from the LLM output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dirigent.core.agent import AgentInfo
from dirigent.core.capability import Capability
from dirigent.core.message import DelegateRequest, DelegateResponse
from dirigent.llm.provider import LLMProvider, LLMResponse


@dataclass
class LLMAgent:
    """Base class for LLM-native agents.

    Subclass and override:
        - build_messages(request) -> list of chat messages
        - parse_response(llm_response, request) -> result dict
    """

    name: str
    capabilities: list[str] = field(default_factory=list)
    provider: LLMProvider | None = None
    system_prompt: str = ""
    model: str = ""
    temperature: float = 0.0

    @property
    def info(self) -> AgentInfo:
        return AgentInfo(
            name=self.name,
            capabilities=[Capability(name=c) for c in self.capabilities],
        )

    def build_messages(self, request: DelegateRequest) -> list[dict[str, str]]:
        """Build the message list for the LLM call. Override for custom prompting."""
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        user_content = f"Task: {request.task}"
        if request.context:
            user_content += f"\n\nContext: {request.context}"
        if request.constraints:
            user_content += f"\n\nConstraints: {request.constraints}"

        messages.append({"role": "user", "content": user_content})
        return messages

    def parse_response(self, llm_response: LLMResponse, request: DelegateRequest) -> Any:
        """Parse the LLM response into a structured result. Override for custom parsing."""
        return llm_response.content

    async def handle(self, request: DelegateRequest) -> DelegateResponse:
        """Handle a delegation request by calling the LLM."""
        if self.provider is None:
            return DelegateResponse(
                sender=self.name,
                receiver=request.sender,
                correlation_id=request.id,
                success=False,
                error="No LLM provider configured",
            )

        messages = self.build_messages(request)

        try:
            llm_response = await self.provider.complete(
                messages,
                model=self.model,
                temperature=self.temperature,
            )
            result = self.parse_response(llm_response, request)
            return DelegateResponse(
                sender=self.name,
                receiver=request.sender,
                correlation_id=request.id,
                success=True,
                result=result,
            )
        except Exception as exc:
            return DelegateResponse(
                sender=self.name,
                receiver=request.sender,
                correlation_id=request.id,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )
