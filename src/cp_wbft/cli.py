"""
[INPUT]:  依赖 simulation.run_simulation，依赖 Python 标准库 argparse
[OUTPUT]: 对外提供 main() 入口函数，注册为 cp-wbft 命令行工具
[POS]:    cp_wbft 的 CLI 层，将仿真参数映射为命令行标志，打印结构化结果
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""
from __future__ import annotations

import argparse

from .simulation import run_simulation


def parse_nodes(value: str) -> set[int]:
    if not value:
        return set()
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a CP-WBFT consensus simulation.")
    parser.add_argument("--topology", default="complete")
    parser.add_argument("--nodes", type=int, default=7)
    parser.add_argument("--byzantine", default="0,1,2,3,4,5")
    parser.add_argument("--honest-answer", default="6")
    parser.add_argument("--byzantine-answer", default="18")
    parser.add_argument("--honest-confidence", type=float, default=0.656)
    parser.add_argument("--byzantine-confidence", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    result = run_simulation(
        args.topology,
        args.nodes,
        parse_nodes(args.byzantine),
        honest_answer=args.honest_answer,
        byzantine_answer=args.byzantine_answer,
        honest_confidence=args.honest_confidence,
        byzantine_confidence=args.byzantine_confidence,
        seed=args.seed,
    )

    print(f"consensus_answer={result.answer}")
    print(f"average_confidence={result.average_confidence:.3f}")
    print(f"supporter_count={result.supporter_count}")
    print("scores=" + ", ".join(f"{k}:{v[0]:.3f}/{v[1]}" for k, v in sorted(result.scores.items())))


if __name__ == "__main__":
    main()
