"""Tests for the core agent interaction framework."""

import pytest

from dirigent.core import DelegateRequest, DelegateResponse, agent
from dirigent.patterns import delegate, fanout, review_loop
from dirigent.runtime import Runtime


@agent(capabilities=["plan"])
class FakePlanner:
    async def handle(self, request: DelegateRequest) -> DelegateResponse:
        return DelegateResponse(
            sender="fakeplanner",
            receiver=request.sender,
            correlation_id=request.id,
            success=True,
            result={"steps": ["step1", "step2"]},
        )


@agent(capabilities=["implement"])
class FakeDeveloper:
    async def handle(self, request: DelegateRequest) -> DelegateResponse:
        return DelegateResponse(
            sender="fakedeveloper",
            receiver=request.sender,
            correlation_id=request.id,
            success=True,
            result="code_output",
        )


@agent(name="dev2", capabilities=["implement"])
class FakeDeveloper2:
    async def handle(self, request: DelegateRequest) -> DelegateResponse:
        return DelegateResponse(
            sender="dev2",
            receiver=request.sender,
            correlation_id=request.id,
            success=True,
            result="code_output_2",
        )


@agent(capabilities=["review"])
class FakeReviewer:
    call_count: int = 0

    async def handle(self, request: DelegateRequest) -> DelegateResponse:
        self.call_count += 1
        # Accept on second round
        accepted = self.call_count >= 2
        return DelegateResponse(
            sender="fakereviewer",
            receiver=request.sender,
            correlation_id=request.id,
            success=True,
            result={"accepted": accepted, "feedback": {"hint": "try harder"}},
        )


@pytest.fixture
def runtime() -> Runtime:
    rt = Runtime()
    rt.register(FakePlanner(), FakeDeveloper(), FakeDeveloper2(), FakeReviewer())
    return rt


async def test_delegate_by_capability(runtime: Runtime) -> None:
    resp = await delegate(runtime, capability="plan", task="build auth")
    assert resp.success
    assert resp.result == {"steps": ["step1", "step2"]}


async def test_delegate_by_name(runtime: Runtime) -> None:
    resp = await delegate(runtime, to="fakedeveloper", task="write code")
    assert resp.success
    assert resp.result == "code_output"


async def test_fanout(runtime: Runtime) -> None:
    results = await fanout(
        runtime, agents=["fakedeveloper", "dev2"], task="implement feature"
    )
    assert len(results) == 2
    assert all(r.success for r in results)
    outputs = {r.result for r in results}
    assert outputs == {"code_output", "code_output_2"}


async def test_review_loop_accepts_on_second_round(runtime: Runtime) -> None:
    result = await review_loop(
        runtime,
        producer="fakedeveloper",
        reviewer="fakereviewer",
        task="implement auth",
        max_rounds=5,
    )
    assert result.accepted
    assert result.rounds == 2
    assert len(result.history) == 4  # 2 produce + 2 review


async def test_traces_recorded(runtime: Runtime) -> None:
    await delegate(runtime, capability="plan", task="test")
    assert len(runtime.traces) == 1
    assert runtime.traces[0]["success"] is True


async def test_unknown_capability_raises(runtime: Runtime) -> None:
    from dirigent.core.registry import RegistryError

    with pytest.raises(RegistryError, match="No agent provides"):
        await delegate(runtime, capability="nonexistent", task="x")
