import asyncio
import unittest
from orchestra.engine import Agent, orchestrate, validate


class EngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_execution_and_dependency_context(self):
        active = 0
        peak = 0
        seen = {}
        both_started = asyncio.Event()
        class Provider:
            async def generate(self, agent, goal, context):
                nonlocal active, peak
                seen[agent.name] = context
                if agent.name in ("a", "b"):
                    active += 1
                    peak = max(peak, active)
                    if active == 2:
                        both_started.set()
                    await asyncio.wait_for(both_started.wait(), 1)
                    active -= 1
                return agent.name + " output"
        agents = [Agent("final", "", ("a", "b")), Agent("a", ""), Agent("b", "")]
        run = await orchestrate(agents, "goal", Provider(), retries=0)
        self.assertEqual(peak, 2)
        self.assertEqual(seen["final"], {"a": "a output", "b": "b output"})
        self.assertEqual(run.results["final"].status, "succeeded")

    async def test_retry_recovers(self):
        class Provider:
            count = 0
            async def generate(self, *args):
                self.count += 1
                if self.count == 1:
                    raise ConnectionError()
                return "ok"
        run = await orchestrate([Agent("a", "")], "goal", Provider())
        self.assertEqual(run.results["a"].attempts, 2)
        self.assertEqual(run.results["a"].output, "ok")

    async def test_timeout_skips_descendants_but_other_branch_succeeds(self):
        class Provider:
            async def generate(self, agent, *args):
                if agent.name == "slow":
                    await asyncio.sleep(10)
                return "ok"
        run = await orchestrate([Agent("slow", ""), Agent("child", "", ("slow",)),
                                 Agent("other", "")], "goal", Provider(), timeout=0.01, retries=0)
        self.assertEqual(run.results["slow"].error, "TimeoutError")
        self.assertEqual(run.results["child"].status, "skipped")
        self.assertEqual(run.results["other"].status, "succeeded")

    async def test_empty_output_is_failure(self):
        class Provider:
            async def generate(self, *args):
                return " "
        run = await orchestrate([Agent("a", "")], "goal", Provider(), retries=0)
        self.assertEqual(run.results["a"].status, "failed")

    async def test_concurrency_limit(self):
        active = peak = 0
        class Provider:
            async def generate(self, *args):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.01)
                active -= 1
                return "ok"
        await orchestrate([Agent(str(i), "") for i in range(4)], "goal", Provider(), concurrency=1)
        self.assertEqual(peak, 1)

    def test_invalid_graphs(self):
        for agents in [[], [Agent("a", ""), Agent("a", "")],
                       [Agent("a", "", ("missing",))],
                       [Agent("a", "", ("b",)), Agent("b", "", ("a",))]]:
            with self.subTest(agents=agents), self.assertRaises(ValueError):
                validate(agents)

    async def test_invalid_limits(self):
        for kwargs in [{"concurrency": 0}, {"timeout": 0}, {"retries": -1}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                await orchestrate([Agent("a", "")], "goal", None, **kwargs)
