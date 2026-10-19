# ruff: noqa: E501
"""A/B бенчмарк: главная модель (planner+synthesizer) Terra vs Opus, воркеры одинаковые.

Задача: 15 задач AIME 2025 II (ответы — целые 0..999, сверяются автоматически).
Запуск: uv run python scripts/bench_planner_ab.py  (нужны ROI_PROVIDER_BASE_URL / ROI_PROVIDER_API_KEY)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from roi_swarm.gateway import ModelGateway
from roi_swarm.graph import build_graph
from roi_swarm.runner import SwarmRunner

ROOT = Path(__file__).resolve().parents[1]

PROBLEMS = [
    "Six points A,B,C,D,E,F lie on a straight line in that order. G is a point not on the line, AC=26, BD=22, CE=31, DF=33, AF=73, CG=40, DG=30. Find the area of triangle BGE.",
    "Find the sum of all positive integers n such that n+2 divides the product 3(n+3)(n^2+9).",
    "Four unit squares form a 2x2 grid. Each of the 12 unit line segments forming the sides of the squares is colored red or blue so that each unit square has exactly 2 red sides and 2 blue sides. Find the number of such colorings.",
    "The product prod_{k=4}^{63} [log_k(5^(k^2-1)) / log_{k+1}(5^(k^2-4))] equals m/n with m,n coprime positive integers. Find m+n.",
    "Triangle ABC has angles BAC=84°, ABC=60°, ACB=36°. D,E,F are midpoints of BC, AC, AB. The circumcircle of DEF meets segments BD, AE, AF again at G, H, J. Points G,D,E,H,J,F divide the circle into six minor arcs (in the order G,D,E,H,J,F around the circle). Find arc DE + 2*arc HJ + 3*arc FG in degrees.",
    "Circle w1 (radius 6, center A) is internally tangent at B to circle w2 (radius 15). C, D lie on w2 with BC a diameter of w2 and BC ⟂ AD. Rectangle EFGH is inscribed in w1 with EF ⟂ BC, C closer to GH than to EF, D closer to FG than to EH. Triangles DGF and CHG have equal areas. The area of EFGH is m/n (coprime). Find m+n.",
    "Let A be the set of positive divisors of 2025. B is a uniformly random subset of A. The probability that B is nonempty and lcm of its elements is 2025 is m/n (coprime). Find m+n.",
    "With 1-, 10-, 25-cent coins, the greedy algorithm (repeatedly take the largest coin not exceeding the remaining value) succeeds for N if no other collection reaching N uses strictly fewer coins. Find the number of N in 1..1000 for which greedy succeeds.",
    "There are n values of x in 0<x<2π where f(x)=sin(7π·sin(5x))=0. For t of them the graph y=f(x) is tangent to the x-axis. Find n+t.",
    "Sixteen chairs in a row. Eight people each select a chair so that no person sits next to two other people. Let N be the number of 8-chair subsets that could be selected. Find N mod 1000.",
    "S is the vertex set of a regular 24-gon. Find the number of ways to draw 12 segments of equal lengths so that each vertex is an endpoint of exactly one segment.",
    "A1A2...A11 is a non-convex simple 11-gon. For every 2<=i<=10 the area of triangle A_i A_1 A_{i+1} is 1 and cos(angle A_i A_1 A_{i+1})=12/13. The perimeter is 20. Then A1A2+A1A11 = (m√n − p)/q with n squarefree and gcd(m,p,q)=1. Find m+n+p+q.",
    "x1=25/11 and x_{k+1}=(1/3)(x_k + 1/x_k − 1). x_2025 = m/n (coprime positive). Find (m+n) mod 1000.",
    "Right triangle ABC with angle A=90°, BC=38. Points K, L inside satisfy AK=AL=BK=CL=KL=14. Area of quadrilateral BKLC is n√3. Find n.",
    "There are exactly three positive real k such that f(x)=(x−18)(x−72)(x−98)(x−k)/x over positive reals attains its minimum at exactly two positive x. Find the sum of these three k.",
]
ANSWERS = [468, 49, 82, 106, 336, 293, 237, 610, 149, 907, 113, 19, 248, 104, 240]

REQUEST = (
    "Solve all 15 AIME problems below. Each answer is an integer 0..999. "
    "Each problem is independent. Verify every answer carefully.\n\n"
    + "\n".join(f"Problem {i}: {p}" for i, p in enumerate(PROBLEMS, 1))
    + "\n\nEnd the final answer with exactly one line in this format:\n"
    "ANSWERS: 1=<int>, 2=<int>, ..., 15=<int>"
)


class CountingGateway(ModelGateway):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls: Counter[str] = Counter()
        self.seconds: Counter[str] = Counter()

    async def generate_text(self, *, model: str, **kwargs) -> str:
        start = time.perf_counter()
        try:
            return await super().generate_text(model=model, **kwargs)
        finally:
            self.calls[model] += 1
            self.seconds[model] += time.perf_counter() - start


def parse_answers(text: str) -> dict[int, int]:
    line = [ln for ln in text.splitlines() if "ANSWERS:" in ln.upper()]
    src = line[-1] if line else text
    return {int(a): int(b) for a, b in re.findall(r"(\d{1,2})\s*=\s*\**(\d{1,4})", src)}


async def run_variant(name: str, main_model: str, worker_model: str) -> dict:
    runner = SwarmRunner()
    settings = runner.settings
    api_mode = os.environ.get("BENCH_API_MODE", "chat_completions")
    timeout_env = os.environ.get("BENCH_TIMEOUT")
    timeout = float(timeout_env) if timeout_env else None
    runner.client = runner.client.with_options(timeout=timeout)
    for role in ("planner", "synthesizer"):
        settings.models[role].model = main_model
    for role in ("router", "worker", "critic"):
        settings.models[role].model = worker_model
    # Opus через прокси работает только в chat_completions; Terra можно и в responses
    gateway = CountingGateway(runner.client, api_mode=api_mode)
    runner.gateway = gateway
    runner.graph = build_graph(settings, gateway, runner.context_pipeline)
    start = time.perf_counter()
    error = None
    state: dict = {}
    try:
        state = await runner.run(REQUEST, run_id=f"bench-{name}-{api_mode}-{int(time.time())}")
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        await runner.close()
    wall = time.perf_counter() - start
    final = state.get("final_answer", "")
    got = parse_answers(final)
    correct = [i for i, a in enumerate(ANSWERS, 1) if got.get(i) == a]
    results = state.get("worker_results", [])
    return {
        "variant": name,
        "main_model": main_model,
        "worker_model": worker_model,
        "wall_seconds": round(wall, 1),
        "score": len(correct),
        "correct": correct,
        "answers": got,
        "worker_target": state.get("worker_target"),
        "tasks": len(state.get("tasks", [])),
        "worker_results": len(results),
        "calls": dict(gateway.calls),
        "model_seconds": {k: round(v, 1) for k, v in gateway.seconds.items()},
        "api_mode": api_mode,
        "timeout": timeout,
        "error": error,
        "final_tail": final[-1500:],
    }


async def main() -> None:
    worker = sys.argv[1] if len(sys.argv) > 1 else "gpt-6-luna"
    variants = sys.argv[2:] or ["gpt-5.6-terra", "claude-opus-5-5"]
    out_dir = ROOT / "data" / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    for main_model in variants:
        res = await run_variant(main_model, main_model, worker)
        path = out_dir / f"ab_{main_model}_{os.environ.get('BENCH_API_MODE', 'chat')}.json"
        path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in res.items() if k != "final_tail"}, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())