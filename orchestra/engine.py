import asyncio
import time
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Agent:
    name: str
    instruction: str
    dependencies: tuple[str, ...] = ()


@dataclass
class Result:
    status: str
    output: str = ""
    attempts: int = 0
    error: str = ""


class Provider(Protocol):
    async def generate(self, agent: Agent, goal: str, context: dict[str, str]) -> str: ...


@dataclass
class Run:
    results: dict[str, Result] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)


def validate(agents: list[Agent]) -> None:
    names = {a.name for a in agents}
    if not agents or len(names) != len(agents) or any(not n.strip() for n in names):
        raise ValueError("Agent names must be nonempty and unique; provide at least one agent")
    for a in agents:
        if not set(a.dependencies) <= names:
            raise ValueError(f"Unknown dependency for {a.name}")
    resolved = set()
    while len(resolved) < len(names):
        ready = {a.name for a in agents if set(a.dependencies) <= resolved} - resolved
        if not ready:
            raise ValueError("Dependency cycle detected")
        resolved |= ready


async def orchestrate(agents: list[Agent], goal: str, provider: Provider, *,
                      concurrency: int = 2, timeout: float = 120, retries: int = 1) -> Run:
    validate(agents)
    if not goal.strip() or concurrency < 1 or timeout <= 0 or retries < 0:
        raise ValueError("Invalid goal or execution limits")
    run = Run()
    slots = asyncio.Semaphore(concurrency)
    tasks = {}

    def event(agent, state, **details):
        run.events.append(dict(agent=agent, state=state, time=time.time(), **details))

    async def execute(agent):
        dependencies = {name: await tasks[name] for name in agent.dependencies}
        if any(r.status != "succeeded" for r in dependencies.values()):
            result = Result("skipped", error="Dependency failed or was skipped")
            run.results[agent.name] = result
            event(agent.name, "skipped")
            return result
        context = {name: r.output for name, r in dependencies.items()}
        for attempt in range(1, retries + 2):
            async with slots:
                event(agent.name, "running", attempt=attempt)
                try:
                    output = await asyncio.wait_for(provider.generate(agent, goal, context), timeout)
                    if not isinstance(output, str) or not output.strip():
                        raise ValueError("Provider returned empty or invalid output")
                    result = Result("succeeded", output=output, attempts=attempt)
                except Exception as exc:
                    result = Result("failed", attempts=attempt, error=type(exc).__name__)
            if result.status == "succeeded":
                break
            if attempt <= retries:
                event(agent.name, "retrying", attempt=attempt, error=result.error)
                await asyncio.sleep(min(0.1 * 2 ** (attempt - 1), 2))
        run.results[agent.name] = result
        event(agent.name, result.status, attempts=result.attempts, error=result.error)
        return result

    # All tasks are registered before the event loop runs them, including forward dependencies.
    tasks = {a.name: asyncio.create_task(execute(a)) for a in agents}
    try:
        await asyncio.gather(*tasks.values())
    finally:
        for task in tasks.values():
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)
    return run
