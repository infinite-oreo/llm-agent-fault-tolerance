"""
[INPUT]:  依赖 consensus.CPWBFT / ConsensusResult，models.AgentState，topologies.build_topology
[OUTPUT]: 对外提供 make_states（初始状态构造）、run_simulation（仿真门面）
[POS]:    cp_wbft 的仿真层，组合拓扑+状态+共识三步为单一调用，被 cli 和 examples 驱动
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""
from __future__ import annotations

from .consensus import CPWBFT, ConsensusResult
from .models import AgentState
from .topologies import build_topology


def make_states(
    n: int,
    byzantine_nodes: set[int],
    *,
    honest_answer: str,
    byzantine_answer: str,
    honest_confidence: float = 0.9,
    byzantine_confidence: float = 0.05,
) -> list[AgentState]:
    return [
        AgentState(
            agent_id=i,
            answer=byzantine_answer if i in byzantine_nodes else honest_answer,
            confidence=byzantine_confidence if i in byzantine_nodes else honest_confidence,
            is_byzantine=i in byzantine_nodes,
        )
        for i in range(n)
    ]


def run_simulation(
    topology: str,
    n: int,
    byzantine_nodes: set[int],
    *,
    honest_answer: str,
    byzantine_answer: str,
    honest_confidence: float,
    byzantine_confidence: float,
    seed: int = 7,
) -> ConsensusResult:
    graph = build_topology(topology, n, seed=seed)
    states = make_states(
        n,
        byzantine_nodes,
        honest_answer=honest_answer,
        byzantine_answer=byzantine_answer,
        honest_confidence=honest_confidence,
        byzantine_confidence=byzantine_confidence,
    )
    return CPWBFT(graph).decide(states)
