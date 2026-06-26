# LLM Agent Fault Tolerance

Reference implementation and simulation sandbox for **CP-WBFT: Confidence Probing-based Weighted Byzantine Fault Tolerant consensus** in LLM multi-agent systems.

This project is based on the paper *Rethinking the Reliability of Multi-agent System: A Perspective from Byzantine Fault Tolerance*. It focuses on the paper's engineering core:

- multiple MAS communication topologies: complete graph, chain, star, tree, random graph, layered graph
- prompt-level confidence probing (PCP) via explicit `Confidence: <0-1>` parsing
- hidden-level confidence probing (HCP) deployment interface for logistic probes over extracted hidden features
- confidence-weighted Byzantine consensus under high fault rates
- reproducible CLI simulations and unit tests

This is not the official paper repository. It is a clean project scaffold intended for experimentation, demos, and extension.

## Why CP-WBFT?

Classic Byzantine fault tolerance usually assumes that all nodes carry equal voting weight and tolerates fewer than one third faulty nodes. The paper observes that LLM agents can be more skeptical of wrong messages and can expose useful confidence signals. CP-WBFT uses those confidence signals to make consensus confidence-weighted instead of count-weighted.

The protocol implemented here follows the paper's two-stage shape:

1. **Local refinement**: each agent compares its decision against neighbor decisions and adopts a neighbor response when the neighbor has higher confidence.
2. **Weighted consensus**: final answers are grouped, scored by average confidence, and tie-broken by supporter count.

## Install

```bash
cd llm-agent-fault-tolerance
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run A Demo

The default demo mirrors the paper's motivating case: 7 agents, 6 Byzantine agents, one honest high-confidence answer.

```bash
cp-wbft --topology complete \
  --nodes 7 \
  --byzantine 0,1,2,3,4,5 \
  --honest-answer 6 \
  --byzantine-answer 18 \
  --honest-confidence 0.656 \
  --byzantine-confidence 0.005
```

Expected output:

```text
consensus_answer=6
average_confidence=0.656
supporter_count=7
scores=6:0.656/7
```

Try a constrained topology:

```bash
cp-wbft --topology chain
```

In a chain, the honest answer only reaches nearby nodes in one refinement round. The consensus can still pick the high-confidence answer, but the supporter count exposes the topology sensitivity discussed in the paper.

## Python API

```python
from cp_wbft.simulation import run_simulation

result = run_simulation(
    "complete",
    7,
    {0, 1, 2, 3, 4, 5},
    honest_answer="6",
    byzantine_answer="18",
    honest_confidence=0.656,
    byzantine_confidence=0.005,
)

print(result.answer)
```

## Probes

Prompt-level confidence probing:

```python
from cp_wbft.probes import PromptConfidenceProbe

response = "Answer: 6; Confidence: 0.90"
confidence = PromptConfidenceProbe().score(response)
```

Hidden-level confidence probing:

```python
from cp_wbft.probes import LinearHiddenConfidenceProbe

probe = LinearHiddenConfidenceProbe(weights=[0.3, -0.1, 0.8], bias=-0.2)
confidence = probe.score([1.0, 0.4, 0.7])
```

The HCP class expects features that have already been extracted from a model runtime, for example pooled answer-token hidden states after PCA reduction. This keeps the package lightweight while matching the deployment interface described by the paper.

## Test

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Project Layout

```text
src/cp_wbft/
  consensus.py    # CP-WBFT local refinement and weighted consensus
  topologies.py   # graph builders for MAS communication patterns
  probes.py       # PCP parser and HCP-compatible linear scorer
  simulation.py   # convenience simulation helpers
  cli.py          # command-line interface
tests/
  test_consensus.py
examples/
  run_demo.py
```

## Next Steps

- connect real LLM providers for PCP response generation
- add hidden-state extraction adapters for LLaMA-family models
- reproduce the GSM8K and XSTest tables with dataset loaders
- add multi-round refinement experiments
- export topology diagrams for README and papers
