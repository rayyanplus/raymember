"""
Raymember Phase 5 Synthetic Benchmark Script
Executes multi-domain state evaluations across 5 domain categories.
"""

import json
from raymember.evaluation.generalized_benchmark import GeneralizedStateBenchmark


def main():
    print("============================================================")
    print("  Raymember Phase 5 Synthetic Benchmark Evaluation")
    print("============================================================")

    bm = GeneralizedStateBenchmark(num_scenarios_per_domain=20)
    res = bm.run_benchmark()

    print(json.dumps(res, indent=2))
    print("\nBenchmark Evaluation Complete!")
    print(f"Overall Attribute Accuracy:  {res['overall_attribute_accuracy'] * 100:.2f}%")
    print(f"Overall Conflict Accuracy:   {res['overall_conflict_accuracy'] * 100:.2f}%")


if __name__ == "__main__":
    main()
