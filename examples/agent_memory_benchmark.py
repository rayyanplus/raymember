"""Executable Multi-Scale Agent Memory Benchmark Script."""

import sys
from raymember.evaluation.agent_benchmark import MultiScaleAgentBenchmark


def main():
    print("=== Running Raymember Multi-Scale Agent Context & Retrieval Benchmark ===")
    benchmark = MultiScaleAgentBenchmark(random_seed=42)
    results = benchmark.run_multi_scale_benchmark(
        output_json="results/context_scale_benchmark.json",
        output_md="results/context_scale_benchmark_report.md",
    )

    print("\nBenchmark Summary across Scales:")
    for scale_name, rep in results["scales"].items():
        print(f"\n--- {scale_name} Scale ({rep['entities']} Entities, {rep['observations']} Obs) ---")
        for strat, r in rep["results"].items():
            print(f"  {strat}:")
            print(f"    Accuracy: {r['answer_accuracy_pct']}%")
            print(f"    Avg Context Chars: {r['avg_context_characters']}")
            print(f"    Context Reduction: {r['context_reduction_pct']}%")
            print(f"    Precision: {r['precision']} | Recall: {r['recall']}")
            print(f"    Retrieval Latency: {r['avg_retrieval_latency_ms']} ms")

    print("\nBenchmark reports saved to:")
    print("  - results/context_scale_benchmark.json")
    print("  - results/context_scale_benchmark_report.md")


if __name__ == "__main__":
    main()
