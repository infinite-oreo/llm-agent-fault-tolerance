"""
[INPUT]:  依赖 models.AgentState，依赖 topologies.Adjacency / sorted_neighbors
[OUTPUT]: 对外提供 ConsensusResult 数据类，CPWBFT 共识引擎（refine + decide）
[POS]:    cp_wbft 的算法核心，被 simulation 驱动，被 __init__ 对外暴露
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .models import AgentState
from .topologies import Adjacency, sorted_neighbors


@dataclass(frozen=True)
class ConsensusResult:
    answer: str
    average_confidence: float
    supporter_count: int
    refined_states: list[AgentState]
    scores: dict[str, tuple[float, int]]


class CPWBFT:
    """Confidence probing-based weighted Byzantine fault-tolerant consensus."""

    def __init__(self, graph: Adjacency) -> None:
        self.graph = graph

    def refine(self, states: list[AgentState]) -> list[AgentState]:
        """Let each agent adopt the most confident visible neighbor decision."""

        by_id = {state.agent_id: state for state in states}
        refined: list[AgentState] = []

        for state in states:
            visible = [by_id[node] for node in sorted_neighbors(self.graph, state.agent_id)]
            strongest = max([state, *visible], key=lambda item: (item.confidence, -item.agent_id))
            refined.append(state.with_decision(strongest.answer, strongest.confidence))

        return refined

    def decide(self, states: list[AgentState]) -> ConsensusResult:
        """Run local refinement, then choose the answer with max average confidence.

        Ties follow the paper's supporter-count tie-breaker and then a stable
        lexical fallback for reproducibility.
        """

        refined = self.refine(states)
        buckets: dict[str, list[float]] = defaultdict(list)
        for state in refined:
            buckets[state.answer].append(state.confidence)

        scores = {
            answer: (sum(confidences) / len(confidences), len(confidences))
            for answer, confidences in buckets.items()
        }
        answer, (avg_confidence, supporter_count) = max(
            scores.items(),
            key=lambda item: (item[1][0], item[1][1], item[0]),
        )
        return ConsensusResult(
            answer=answer,
            average_confidence=avg_confidence,
            supporter_count=supporter_count,
            refined_states=refined,
            scores=scores,
        )
