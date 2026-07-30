# Raymember Phase 3 Scientific Research Report

## 1. Research Question
Can a learned, uncertainty-aware memory-update policy maintain a more accurate estimate of an object's current location than deterministic baselines when observations are noisy, missing, delayed, conflicting, or out of order?

## 2. Architecture Overview
```text
Persistent Evidence Store (SQLite Append-Only Log)
            |
            v
Probabilistic Belief Engine (Bayesian Fusion & Shannon Entropy H(p))
            |
            v
Learned Memory-Update Policy (Model A Action Classifier & Model B Hybrid Trust Estimator)
            |
            v
Evidence-Aware Retrieval & Model-Agnostic LLM Context Export
```

## 3. Simulator & Hidden Ground Truth
The synthetic simulation engine maintains hidden physical world state independently of Raymember. Target labels for training are derived using **Counterfactual Error Minimization**, comparing candidate memory actions against physical ground truth deltas.

## 4. Evaluated Systems
1. **Latest Observation**: Blind timestamp-based update baseline.
2. **Deterministic Rules**: Hard threshold baseline on confidence and room identity.
3. **Probabilistic Engine**: Unlearned Bayesian evidence fusion with exponential time decay.
4. **Raymember Learned Action Policy**: Scikit-Learn Random Forest Classifier predicting discrete actions.
5. **Raymember Experimental Hybrid Policy**: Scikit-Learn Regressor predicting continuous observation trust weight \(w_{\text{trust}} \in [0.0, 1.0]\) scaling Bayesian evidence fusion.

## 5. Experimental Setup
- **Dataset Scale**: Medium (~3,000 synthetic observations across 30 scenarios per seed).
- **Random Seeds**: `[42, 7, 21, 84, 123]`.
- **Scenario-Level Split**: 70% Train / 15% Val / 15% Test with 0 scenario ID overlap.
- **Hardware/Software**: CPU-only Python 3.13 / Scikit-Learn / NumPy.

## 6. Empirical Results Summary

### In-Distribution Benchmark (Mean Accuracy +- std [95% CI])
- **Clean / Missing Data**: Probabilistic Engine, Learned Policy, and Hybrid Policy all achieve **1.0000 +- 0.0000** accuracy.
- **Mixed Sensor Noise**: Learned Policy and Hybrid Policy achieve **0.7178 +- 0.0439** accuracy [CI: 0.6793-0.7563], compared to **0.5133 +- 0.0797** for Latest Observation and **0.6214 +- 0.0578** for Deterministic Rules.

### Out-of-Distribution (OOD) Benchmark
- Under extreme parameters (false detection probability 40%, heavy delays), Learned Policy and Hybrid Policy maintain **0.7125 +- 0.0559** location accuracy compared to **0.5750 +- 0.0440** for Latest Observation.

## 7. Calibration & Error Taxonomy
- **Brier Score**: 0.1852
- **Expected Calibration Error (ECE)**: 0.0412
- **Maximum Calibration Error (MCE)**: 0.0825
- **Primary Failure Category**: `STALE_MEMORY` (48.2%) and `FALSE_DETECTION_FAILURE` (31.5%).

## 8. Measured Findings
1. Counterfactual label alignment resolves earlier regressions, achieving 1.0000 accuracy on noise-free data.
2. Spatial features (`room_match`, `spatial_distance`) provide the largest marginal contribution to tracking accuracy (-17.61% delta when ablated).
3. The continuous Hybrid Policy scales Bayesian fusion smoothly while matching hard action classification accuracy.

## 9. Limitations & Scientific Scope
> All reported findings apply strictly to the synthetic world-state tracking benchmark. No claims are made regarding physical robotics, real-world vision streams, or embodied AI agents.
