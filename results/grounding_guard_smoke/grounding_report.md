# Factual Grounding & Hallucination Reduction Report

- **Provider**: `DeterministicOfflineModel`
- **Scenarios**: 24
- **Total Trials**: 120

## Primary Grounding & Hallucination Metrics

| System | Grounded Acc | Supported Rate | Unsupported Rate | Contradiction Rate | Hallucination Rate | False Certainty | Abstention Acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline` | **41.7%** | 41.7% | 58.3% | 58.3% | 8.3% | 58.3% | 33.3% |
| `full_context` | **75.0%** | 75.0% | 25.0% | 25.0% | 25.0% | 25.0% | 33.3% |
| `naive_retrieval` | **75.0%** | 75.0% | 25.0% | 25.0% | 25.0% | 25.0% | 33.3% |
| `raymember` | **66.7%** | 66.7% | 33.3% | 25.0% | 33.3% | 33.3% | 25.0% |
| `raymember_grounded` | **58.3%** | 58.3% | 41.7% | 33.3% | 41.7% | 33.3% | 25.0% |

## Secondary Engineering Metrics (Context & Latency)

| System | Avg Latency (ms) | Avg Input Tokens | Avg Context Chars |
| --- | --- | --- | --- |
| `baseline` | 0.0ms | 88 | 49 |
| `full_context` | 0.3ms | 2072 | 7899 |
| `naive_retrieval` | 0.8ms | 189 | 363 |
| `raymember` | 684.0ms | 217 | 470 |
| `raymember_grounded` | 742.9ms | 16 | 101 |

## Grounding Guard Metrics

- **Deterministic Answer Rate**: 100.0%
- **LLM Call Avoidance Rate**: 100.0%
- **Validation Failure Rate**: 0.0%
- **Fallback Rate**: 0.0%
- **False Premise Correction Rate**: 8.3%
- **Entity Isolation Accuracy**: 8.3%
- **Temporal Gap Abstention Rate**: 8.3%

## Grounded Accuracy by Task Category

| Category | `baseline` | `full_context` | `naive_retrieval` | `raymember` | `raymember_grounded` |
| --- | --- | --- | --- | --- | --- |
| CONFLICT_RESOLUTION | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| CONTRADICTORY_EVIDENCE | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| CROSS_SESSION_PERSISTENCE | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| CURRENT_STATE | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| DISTRACTOR_HALLUCINATION | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| DISTRACTOR_RESISTANCE | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| ENTITY_CONFUSION | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| FALSE_PREMISE | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| MISSING_INFO | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| TEMPORAL_HALLUCINATION | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| TEMPORAL_REASONING | 0.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| UNCERTAIN_EVIDENCE | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% |
