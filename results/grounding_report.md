# Factual Grounding & Hallucination Reduction Report

- **Provider**: `Antigravity(Antigravity-Gemini-3.6-Flash)`
- **Scenarios**: 12
- **Total Trials**: 48

## Primary Grounding & Hallucination Metrics

| System | Grounded Acc | Supported Rate | Unsupported Rate | Contradiction Rate | Hallucination Rate | False Certainty | Abstention Acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `baseline` | **33.3%** | 33.3% | 66.7% | 58.3% | 8.3% | 0.0% | 33.3% |
| `full_context` | **33.3%** | 33.3% | 66.7% | 25.0% | 58.3% | 66.7% | 0.0% |
| `naive_retrieval` | **33.3%** | 33.3% | 66.7% | 25.0% | 58.3% | 66.7% | 0.0% |
| `raymember` | **50.0%** | 50.0% | 50.0% | 25.0% | 41.7% | 50.0% | 16.7% |

## Secondary Engineering Metrics (Context & Latency)

| System | Avg Latency (ms) | Avg Input Tokens | Avg Context Chars |
| --- | --- | --- | --- |
| `baseline` | 0.0ms | 88 | 49 |
| `full_context` | 0.8ms | 3639 | 14167 |
| `naive_retrieval` | 3.0ms | 189 | 363 |
| `raymember` | 1265.1ms | 217 | 470 |

## Grounded Accuracy by Task Category

| Category | `baseline` | `full_context` | `naive_retrieval` | `raymember` |
| --- | --- | --- | --- | --- |
| CONFLICT_RESOLUTION | 0.0% | 100.0% | 100.0% | 100.0% |
| CONTRADICTORY_EVIDENCE | 0.0% | 0.0% | 0.0% | 0.0% |
| CROSS_SESSION_PERSISTENCE | 0.0% | 0.0% | 0.0% | 0.0% |
| CURRENT_STATE | 0.0% | 0.0% | 0.0% | 0.0% |
| DISTRACTOR_HALLUCINATION | 100.0% | 0.0% | 0.0% | 100.0% |
| DISTRACTOR_RESISTANCE | 0.0% | 100.0% | 100.0% | 100.0% |
| ENTITY_CONFUSION | 0.0% | 100.0% | 100.0% | 100.0% |
| FALSE_PREMISE | 0.0% | 0.0% | 0.0% | 0.0% |
| MISSING_INFO | 100.0% | 0.0% | 0.0% | 0.0% |
| TEMPORAL_HALLUCINATION | 100.0% | 0.0% | 0.0% | 0.0% |
| TEMPORAL_REASONING | 0.0% | 100.0% | 100.0% | 100.0% |
| UNCERTAIN_EVIDENCE | 100.0% | 0.0% | 0.0% | 100.0% |
