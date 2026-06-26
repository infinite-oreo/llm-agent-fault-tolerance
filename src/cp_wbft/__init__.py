"""
[INPUT]:  依赖 consensus / models / topologies 三个内部模块
[OUTPUT]: 对外提供 AgentState, CPWBFT, ConsensusResult, build_topology
[POS]:    cp_wbft 包入口，聚合公开 API，屏蔽内部模块路径
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""

from .consensus import CPWBFT, ConsensusResult
from .models import AgentState
from .topologies import build_topology

__all__ = ["AgentState", "CPWBFT", "ConsensusResult", "build_topology"]
