"""
[INPUT]:  依赖 cp_wbft.simulation.run_simulation
[OUTPUT]: 无导出，直接打印 ConsensusResult，供快速验证使用
[POS]:    examples 层的唯一入口，展示 6/7 拜占庭完全图场景的共识恢复过程
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""
from cp_wbft.simulation import run_simulation


if __name__ == "__main__":
    result = run_simulation(
        "complete",
        7,
        {0, 1, 2, 3, 4, 5},
        honest_answer="6",
        byzantine_answer="18",
        honest_confidence=0.656,
        byzantine_confidence=0.005,
    )
    print(result)
