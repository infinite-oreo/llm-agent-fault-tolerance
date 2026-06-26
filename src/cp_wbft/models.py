"""
[INPUT]:  依赖 Python 标准库 dataclasses
[OUTPUT]: 对外提供 AgentState 冻结数据类
[POS]:    cp_wbft 的数据层，被 consensus / simulation / probes 消费
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentState:
    """One agent's answer and confidence for a single task."""

    agent_id: int
    answer: str
    confidence: float
    is_byzantine: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0.0, 1.0]")

    def with_decision(self, answer: str, confidence: float) -> "AgentState":
        return AgentState(
            agent_id=self.agent_id,
            answer=answer,
            confidence=confidence,
            is_byzantine=self.is_byzantine,
        )
