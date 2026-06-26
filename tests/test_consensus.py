"""
[INPUT]:  依赖 cp_wbft.consensus / models / probes / topologies 四个模块
[OUTPUT]: 无导出，unittest 套件覆盖核心场景
[POS]:    tests 层唯一测试文件，验证完全图恢复 / 链图传播 / 两种探针的正确性
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""
import unittest

from cp_wbft.consensus import CPWBFT
from cp_wbft.models import AgentState
from cp_wbft.probes import LinearHiddenConfidenceProbe, PromptConfidenceProbe
from cp_wbft.topologies import build_topology


class CPWBFTTests(unittest.TestCase):
    def test_complete_graph_recovers_honest_answer_with_one_confident_agent(self) -> None:
        graph = build_topology("complete", 7)
        states = [
            AgentState(i, "18", 0.005, is_byzantine=True)
            for i in range(6)
        ] + [AgentState(6, "6", 0.656)]

        result = CPWBFT(graph).decide(states)

        self.assertEqual(result.answer, "6")
        self.assertEqual(result.supporter_count, 7)

    def test_chain_restricts_supporter_spread(self) -> None:
        graph = build_topology("chain", 7)
        states = [
            AgentState(i, "18", 0.005, is_byzantine=True)
            for i in range(6)
        ] + [AgentState(6, "6", 0.656)]

        result = CPWBFT(graph).decide(states)

        self.assertEqual(result.answer, "6")
        self.assertEqual(result.supporter_count, 2)
        self.assertEqual(result.scores["6"], (0.656, 2))

    def test_prompt_probe_parses_confidence(self) -> None:
        response = "Solution: ... Answer: 6; Confidence: 0.90"
        self.assertEqual(PromptConfidenceProbe().score(response), 0.9)

    def test_linear_hidden_probe_scores_features(self) -> None:
        probe = LinearHiddenConfidenceProbe([1.0, -1.0], bias=0.0)
        self.assertGreater(probe.score([2.0, 0.0]), 0.8)


if __name__ == "__main__":
    unittest.main()
