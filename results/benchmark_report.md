# Comparative Memory Systems Benchmark Report

- **Provider**: `Antigravity(Antigravity-Gemini-3.6-Flash)`
- **Scenarios**: 10
- **Total Trials**: 40

## Overall System Performance

| System | Accuracy | Confidence Acc | Provenance Acc | Hallucination Rate | Avg Latency (ms) | Avg Tokens |
| --- | --- | --- | --- | --- | --- | --- |
| `baseline` | 0.0% | 0.0% | 20.0% | 100.0% | 0.0ms | 81 |
| `full_context` | 40.0% | 100.0% | 40.0% | 20.0% | 3.2ms | 4008 |
| `naive_retrieval` | 40.0% | 100.0% | 40.0% | 20.0% | 5.4ms | 190 |
| `raymember` | 60.0% | 100.0% | 40.0% | 20.0% | 3609.6ms | 224 |

## Accuracy by Category

| Category | `baseline` | `full_context` | `naive_retrieval` | `raymember` |
| --- | --- | --- | --- | --- |
| CONFLICT_RESOLUTION | 0.0% | 100.0% | 100.0% | 100.0% |
| CROSS_SESSION_PERSISTENCE | 0.0% | 0.0% | 0.0% | 0.0% |
| CURRENT_STATE | 0.0% | 50.0% | 50.0% | 50.0% |
| DISTRACTOR_RESISTANCE | 0.0% | 0.0% | 0.0% | 100.0% |
| TEMPORAL_REASONING | 0.0% | 50.0% | 50.0% | 50.0% |

## Accuracy vs Distractor Memory Scale

| Scale | `baseline` | `full_context` | `naive_retrieval` | `raymember` |
| --- | --- | --- | --- | --- |
| 500 obs | 0.0% | 0.0% | 0.0% | 100.0% |
