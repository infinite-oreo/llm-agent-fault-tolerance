"""
[INPUT]:  依赖 Python 标准库 math / re / collections.abc
[OUTPUT]: 对外提供 PromptConfidenceProbe（文本解析）、LinearHiddenConfidenceProbe（线性打分）
[POS]:    cp_wbft 的置信探针层，独立于共识逻辑，可单独接入真实 LLM 输出流水线
[PROTOCOL]: 变更时更新此头部，然后检查 CLAUDE.md
"""
from __future__ import annotations

import math
import re
from collections.abc import Sequence


class PromptConfidenceProbe:
    """Parse prompt-level confidence emitted by an LLM response."""

    _pattern = re.compile(r"confidence\s*[:=]\s*([01](?:\.\d+)?)", re.IGNORECASE)

    def score(self, response: str) -> float:
        match = self._pattern.search(response)
        if not match:
            raise ValueError("response does not contain a Confidence: <0-1> field")
        return min(1.0, max(0.0, float(match.group(1))))


class LinearHiddenConfidenceProbe:
    """Small HCP-compatible linear probe for pre-extracted hidden features.

    The paper trains logistic regression on PCA-reduced hidden states. This
    class keeps that deployment interface without tying the project to a
    specific model runtime.
    """

    def __init__(self, weights: Sequence[float], bias: float = 0.0) -> None:
        self.weights = tuple(float(w) for w in weights)
        self.bias = float(bias)

    def score(self, features: Sequence[float]) -> float:
        if len(features) != len(self.weights):
            raise ValueError("features and weights must have the same length")
        logit = sum(w * x for w, x in zip(self.weights, features)) + self.bias
        return 1.0 / (1.0 + math.exp(-logit))
