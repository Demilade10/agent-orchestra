import argparse
import asyncio
import json
import uuid
from dataclasses import asdict
from pathlib import Path
from .engine import Agent, orchestrate
from .providers import DemoProvider, OllamaProvider


def team():
    return [
        Agent("planner", "You are a planning specialist. Turn the goal into requirements, constraints, and acceptance criteria. Be concise. State unknowns."),
        Agent("architect", "You are a software architect. Use the plan to propose a concrete implementation, components and tradeoffs. Do not claim to have run code.", ("planner",)),
        Agent("risk_analyst", "You are a risk analyst. Use the plan to identify failure modes, assumptions and practical validation steps. Do not invent sources.", ("planner",)),
        Agent("reviewer", "You are the final reviewer. Reconcile the plan, architecture and risk analysis into a concise implementation proposal with milestones, tests and unresolved questions. Flag conflicts. Do not claim verified facts without evidence.", ("planner", "architect", "risk_analyst")),
    ]


def main():
    parser = argparse.ArgumentParser(description="Coordinate a team of project-planning agents")
    parser.add_argument("goal")
    parser.add_argument("--provider", choices=["demo", "ollama"], default="demo")
    parser.add_argument("--model", default="llama3.2")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("runs"))
    args = parser.parse_args()
    provider = DemoProvider() if args.provider == "demo" else OllamaProvider(args.model)
    try:
        run = asyncio.run(orchestrate(team(), args.goal, provider, concurrency=args.concurrency,
                                     timeout=args.timeout, retries=args.retries))
    except ValueError as exc:
        parser.error(str(exc))
    destination = args.output / uuid.uuid4().hex
    destination.mkdir(parents=True)
    payload = dict(provider=args.provider, model=args.model if args.provider == "ollama" else None,
                   goal=args.goal, **asdict(run))
    (destination / "run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    final = run.results["reviewer"]
    report = final.output if final.status == "succeeded" else "Run failed: inspect run.json for agent statuses."
    (destination / "report.md").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nSaved: {destination}")
    return 0 if final.status == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
