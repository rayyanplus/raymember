# Factual Grounding & Hallucination Reduction Report

- **Provider**: `Antigravity(Antigravity-Gemini-3.6-Flash)`
- **Scenarios**: 120
- **Total Trials**: 1800

## Primary Grounding & Hallucination Metrics

| System | Grounded Acc | Supported Rate | Unsupported Rate | Contradiction Rate | Hallucination Rate | False Certainty | Abstention Acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline` | **33.3%** | 33.3% | 66.7% | 58.3% | 8.3% | 0.0% | 33.3% |
| `full_context` | **33.3%** | 33.3% | 66.7% | 25.0% | 58.3% | 66.7% | 0.0% |
| `naive_retrieval` | **33.3%** | 33.3% | 66.7% | 25.0% | 58.3% | 66.7% | 0.0% |
| `raymember` | **50.0%** | 50.0% | 50.0% | 25.0% | 37.5% | 50.0% | 16.7% |
| `raymember_grounded` | **58.3%** | 58.3% | 41.7% | 33.3% | 41.7% | 33.3% | 25.0% |

## Secondary Engineering Metrics (Context & Latency)

| System | Avg Latency (ms) | Avg Input Tokens | Avg Context Chars |
| --- | --- | --- | --- |
| `baseline` | 0.0ms | 88 | 49 |
| `full_context` | 0.4ms | 2804 | 10827 |
| `naive_retrieval` | 1.0ms | 189 | 363 |
| `raymember` | 1291.5ms | 217 | 470 |
| `raymember_grounded` | 1315.8ms | 17 | 101 |

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
| DISTRACTOR_HALLUCINATION | 100.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| DISTRACTOR_RESISTANCE | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| ENTITY_CONFUSION | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| FALSE_PREMISE | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| MISSING_INFO | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| TEMPORAL_HALLUCINATION | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% |
| TEMPORAL_REASONING | 0.0% | 100.0% | 100.0% | 100.0% | 0.0% |
| UNCERTAIN_EVIDENCE | 100.0% | 0.0% | 0.0% | 100.0% | 0.0% |
