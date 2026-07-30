# Raymember Multi-Scale Context Efficiency & Retrieval Benchmark Report

Evaluates 3 retrieval strategies across Small (10 entities), Medium (100 entities), and Large (500 entities) memory scale worlds.

### Small Scale (10 Entities, 50 Observations)
| Strategy | Accuracy (%) | Avg Chars | Median Chars | Context Reduction (%) | Precision | Recall | Latency (ms) |
|---|---|---|---|---|---|---|---|
| **1. No Memory** | **0.0%** | 0.0 | 0 | **100.0%** | 0.0 | 0.0 | 0.0ms |
| **2. Naive Full Memory** | **100.0%** | 5,303.0 | 5,303 | **0.0%** | 0.0167 | 1.0 | 32.82ms |
| **3. Raymember Ranked Memory** | **100.0%** | 1,223.6 | 1,225 | **76.93%** | 0.1 | 1.0 | 32.91ms |

### Medium Scale (100 Entities, 500 Observations)
| Strategy | Accuracy (%) | Avg Chars | Median Chars | Context Reduction (%) | Precision | Recall | Latency (ms) |
|---|---|---|---|---|---|---|---|
| **1. No Memory** | **0.0%** | 0.0 | 0 | **100.0%** | 0.0 | 0.0 | 0.0ms |
| **2. Naive Full Memory** | **100.0%** | 52,103.0 | 52,103 | **0.0%** | 0.0017 | 1.0 | 356.73ms |
| **3. Raymember Ranked Memory** | **100.0%** | 1,223.8 | 1,225 | **97.65%** | 0.1 | 1.0 | 321.95ms |

### Large Scale (500 Entities, 2,500 Observations)
| Strategy | Accuracy (%) | Avg Chars | Median Chars | Context Reduction (%) | Precision | Recall | Latency (ms) |
|---|---|---|---|---|---|---|---|
| **1. No Memory** | **0.0%** | 0.0 | 0 | **100.0%** | 0.0 | 0.0 | 0.0ms |
| **2. Naive Full Memory** | **100.0%** | 953,603.0 | 953,603 | **0.0%** | 0.0003 | 1.0 | 10140.08ms |
| **3. Raymember Ranked Memory** | **42.5%** | 1,224.9 | 1,225 | **99.87%** | 0.0425 | 0.425 | 24842.49ms |

---
### Context Mode Comparison (`compact` vs `standard` vs `evidence`)
- **`compact`**: Ultra-concise current state & material uncertainty summary.
- **`standard`**: Balanced evidence-aware context with history & update explanation.
- **`evidence`**: Full observation trajectory with provenance tags and timestamps.

> **Validation Disclaimer**: This benchmark evaluates deterministic context retrieval precision, recall, and character reduction across memory database scales. It does not fabricate claims regarding third-party LLM parameter performance.