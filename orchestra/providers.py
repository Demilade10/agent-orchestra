import asyncio
import json
import sys


class DemoProvider:
    """Deterministic fixtures, not AI-generated answers."""
    async def generate(self, agent, goal, context):
        await asyncio.sleep(0.05)
        sections = {
            "planner": "Define requirements, compare implementation choices, review risks, and propose milestones.",
            "architect": "Start with a Python prototype. Separate input handling, core logic, and output. Add tests before integration.",
            "risk_analyst": "Validate assumptions and input quality. Measure failure cases; document resource and cost limits.",
            "reviewer": "Milestones: 1. Define acceptance criteria. 2. Implement a small prototype. 3. Test failures. 4. Document results.",
        }
        return f"[DEMO FIXTURE] {agent.name}\nGoal: {goal}\n{sections.get(agent.name, 'Example output.')}\nInputs received: {', '.join(context) or 'none'}"


class OllamaProvider:
    def __init__(self, model):
        self.model = model

    async def generate(self, agent, goal, context):
        payload = {"model": self.model, "stream": False, "messages": [
            {"role": "system", "content": agent.instruction},
            {"role": "user", "content": json.dumps({"goal": goal, "dependency_outputs": context})},
        ]}
        # Isolate blocking HTTP in a cancellable process. No shell or generated code execution.
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "orchestra.ollama_worker",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        try:
            stdout, _ = await proc.communicate(json.dumps(payload).encode())
            if proc.returncode:
                raise RuntimeError("Ollama request failed; verify server and model")
            return json.loads(stdout)["message"]["content"]
        finally:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
