"""
CLI Entry Point for Factual Grounding & Hallucination Comparative Benchmark.

Executes benchmark comparing 4 systems:
  1. BASELINE_LLM (No memory)
  2. FULL_CONTEXT_LLM (Unfiltered observation history)
  3. NAIVE_RETRIEVAL_LLM (Top-k lexical retrieval over raw observations)
  4. RAYMEMBER_LLM (Relevance-ranked, belief-resolved persistent memory)

Primary Research Objective:
  Evaluates whether Raymember reduces hallucinated, unsupported, contradictory,
  and unjustifiably confident world-state claims across 12 scenario categories.
"""

import argparse
import json
import os
import sys

# Ensure src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from raymember.evaluation.comparative_benchmark import (
    ComparativeScenarioGenerator,
    MemoryBenchmarkRunner,
)


def main():
    parser = argparse.ArgumentParser(description="Raymember Grounding & Hallucination Benchmark")
    parser.add_argument("--provider", type=str, default="mock", help="Model provider: mock, antigravity, gemini, openai, ollama, anthropic")
    parser.add_argument("--model", type=str, default=None, help="Model name identifier")
    parser.add_argument("--systems", type=str, default="baseline,full_context,naive_retrieval,raymember,raymember_grounded", help="Comma-separated list of systems to evaluate")
    parser.add_argument("--scenarios", type=int, default=120, help="Number of scenarios to generate")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per scenario")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic scenario generation")
    parser.add_argument("--output", type=str, default="results", help="Output directory path")

    args = parser.parse_args()

    systems_list = [s.strip().lower() for s in args.systems.split(",") if s.strip()]

    print("=======================================================================")
    print("  Raymember Factual Grounding & Hallucination Comparative Benchmark")
    print("=======================================================================")
    print(f"  Provider:       {args.provider}")
    print(f"  Model:          {args.model or 'Default'}")
    print(f"  Systems:        {', '.join(systems_list)}")
    print(f"  Scenarios:      {args.scenarios}")
    print(f"  Runs:           {args.runs}")
    print(f"  Random Seed:    {args.seed}")
    print(f"  Output Dir:     {args.output}")
    print("=======================================================================")

    # 1. Generate scenarios
    generator = ComparativeScenarioGenerator(seed=args.seed)
    scenarios = generator.generate_scenarios(num_scenarios=args.scenarios)
    print(f"\n[STEP 1] Generated {len(scenarios)} scenarios across 12 grounding categories.")

    # 2. Run benchmark
    runner = MemoryBenchmarkRunner(
        provider=args.provider,
        model_name=args.model,
        output_dir=args.output,
    )

    print("\n[STEP 2] Running benchmark evaluations...")
    summary = runner.run_benchmark(
        scenarios=scenarios,
        systems=systems_list,
        num_runs=args.runs,
    )

    # 3. Print Primary Grounding & Hallucination Summary Table
    print("\n=========================================================================================================")
    print("  Primary Evaluation Objective: Factual Grounding & Hallucination Metrics")
    print("=========================================================================================================")
    print(f"{'System':<18} | {'Grounded Acc':<12} | {'Supported %':<12} | {'Unsupported %':<14} | {'Contradiction %':<15} | {'Hallucination %':<15} | {'False Certainty %':<16}")
    print("-" * 115)

    for sys_name, m in summary["systems"].items():
        print(
            f"{sys_name:<18} | {m['grounded_answer_accuracy']*100:>10.1f}% | "
            f"{m['supported_claim_rate']*100:>10.1f}% | {m['unsupported_claim_rate']*100:>12.1f}% | "
            f"{m['contradiction_rate']*100:>13.1f}% | {m['hallucination_rate']*100:>13.1f}% | "
            f"{m['false_certainty_rate']*100:>14.1f}%"
        )

    print("\n=========================================================================================================")
    print("  Secondary Engineering Metrics: Context Size & Latency")
    print("=========================================================================================================")
    print(f"{'System':<18} | {'Avg Latency (ms)':<18} | {'Avg Input Tokens':<18} | {'Avg Context Chars':<18}")
    print("-" * 78)
    for sys_name, m in summary["systems"].items():
        print(
            f"{sys_name:<18} | {m['avg_latency_ms']:>16.1f}ms | "
            f"{m['avg_input_tokens']:>16.0f} | {m['avg_context_chars']:>16.0f}"
        )

    if "raymember_grounded" in summary["systems"]:
        print("\n=========================================================================================================")
        print("  Grounding Guard Metrics (raymember_grounded)")
        print("=========================================================================================================")
        m = summary["systems"]["raymember_grounded"]
        print(f"  Deterministic Answer Rate:   {m['deterministic_answer_rate']*100:>6.1f}%")
        print(f"  LLM Call Avoidance Rate:     {m['llm_call_avoidance_rate']*100:>6.1f}%")
        print(f"  Validation Failure Rate:     {m['validation_failure_rate']*100:>6.1f}%")
        print(f"  Fallback Rate:               {m['fallback_rate']*100:>6.1f}%")

    print("=========================================================================================================")
    print(f"Trial logs exported to:           {os.path.join(args.output, 'benchmark_trials.csv')}")
    print(f"JSON summary saved to:             {os.path.join(args.output, 'benchmark_summary.json')}")
    print(f"Grounding Markdown report at:      {os.path.join(args.output, 'grounding_report.md')}")
    print("=========================================================================================================")


if __name__ == "__main__":
    main()
