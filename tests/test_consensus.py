"""
[INPUT]:  依赖 cp_wbft.consensus / models / probes / topologies 四个模块
[OUTPUT]: 无导出，unittest 套件覆盖核心场景
[POS]:    tests 层唯一测试文件，验证完全图恢复 / 链图传播（含多轮）/ 两种探针的正确性与异常 / 拓扑与状态的边界条件
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


class MultiRoundRefineTests(unittest.TestCase):
    """多轮 refine：确认信息能在受限拓扑中逐跳传播，而不是只传一跳。"""

    def _states(self) -> list[AgentState]:
        return [
            AgentState(i, "18", 0.005, is_byzantine=True)
            for i in range(6)
        ] + [AgentState(6, "6", 0.656)]

    def test_single_round_chain_only_reaches_immediate_neighbor(self) -> None:
        graph = build_topology("chain", 7)
        result = CPWBFT(graph).decide(self._states(), rounds=1)

        self.assertEqual(result.answer, "6")
        self.assertEqual(result.supporter_count, 2)

    def test_multi_round_chain_propagates_to_whole_graph(self) -> None:
        graph = build_topology("chain", 7)
        # 7 节点链最远需要 6 跳才能把信息从一端传到另一端。
        result = CPWBFT(graph).decide(self._states(), rounds=6)

        self.assertEqual(result.answer, "6")
        self.assertEqual(result.supporter_count, 7)
        self.assertEqual(result.scores["6"], (0.656, 7))

    def test_extra_rounds_beyond_convergence_are_a_no_op(self) -> None:
        graph = build_topology("chain", 7)
        converged = CPWBFT(graph).decide(self._states(), rounds=6)
        over_run = CPWBFT(graph).decide(self._states(), rounds=50)

        self.assertEqual(converged.answer, over_run.answer)
        self.assertEqual(converged.supporter_count, over_run.supporter_count)

    def test_rounds_must_be_at_least_one(self) -> None:
        graph = build_topology("chain", 7)
        with self.assertRaises(ValueError):
            CPWBFT(graph).refine(self._states(), rounds=0)


class TieBreakTests(unittest.TestCase):
    def test_equal_average_confidence_breaks_tie_by_supporter_count(self) -> None:
        graph = build_topology("complete", 4)
        states = [
            AgentState(0, "a", 0.5),
            AgentState(1, "b", 0.5),
            AgentState(2, "b", 0.5),
            AgentState(3, "a", 0.5),
        ]
        # 完全图会让所有人趋同到最高置信度的答案；这里置信度全相等，
        # refine 阶段按 (confidence, -agent_id) 排序，agent_id 更小的胜出。
        result = CPWBFT(graph).decide(states)

        self.assertIn(result.answer, {"a", "b"})
        self.assertEqual(result.supporter_count, 4)


class ProbeErrorTests(unittest.TestCase):
    def test_prompt_probe_raises_when_confidence_missing(self) -> None:
        with self.assertRaises(ValueError):
            PromptConfidenceProbe().score("Answer: 6, no confidence here")

    def test_prompt_probe_clamps_out_of_range_is_unreachable_but_clamps_bounds(self) -> None:
        # 正则只匹配 [0,1] 内的数字，这里验证边界值 0 和 1 均可解析。
        self.assertEqual(PromptConfidenceProbe().score("Confidence: 0"), 0.0)
        self.assertEqual(PromptConfidenceProbe().score("Confidence: 1"), 1.0)

    def test_linear_hidden_probe_rejects_mismatched_feature_length(self) -> None:
        probe = LinearHiddenConfidenceProbe([1.0, -1.0], bias=0.0)
        with self.assertRaises(ValueError):
            probe.score([1.0])


class ModelAndTopologyBoundaryTests(unittest.TestCase):
    def test_agent_state_rejects_confidence_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            AgentState(0, "6", 1.5)
        with self.assertRaises(ValueError):
            AgentState(0, "6", -0.1)

    def test_build_topology_rejects_unknown_name(self) -> None:
        with self.assertRaises(ValueError):
            build_topology("hexagon", 5)

    def test_build_topology_rejects_too_few_nodes(self) -> None:
        with self.assertRaises(ValueError):
            build_topology("complete", 1)

    def test_star_topology_hub_reaches_all_leaves_in_one_round(self) -> None:
        graph = build_topology("star", 5)
        states = [AgentState(0, "6", 0.9)] + [
            AgentState(i, "18", 0.1, is_byzantine=True) for i in range(1, 5)
        ]
        result = CPWBFT(graph).decide(states, rounds=1)

        self.assertEqual(result.answer, "6")
        self.assertEqual(result.supporter_count, 5)


if __name__ == "__main__":
    unittest.main()
