"""
Raymember Phase 7 Agent Comparison Benchmark Script
Executes 32 scenarios across Strategy A (No Memory), Strategy B (Naive History), and Strategy C (Raymember).

Supports CLI flags for opting into real LLM models (--provider openai/ollama/anthropic), repeated runs (--runs N), and saving audited JSON results.
Runs 100% offline deterministically by default.
"""

import argparse
import json
import sys

from raymember.evaluation.agent_comparison import AgentComparisonBenchmark
from raymember.evaluation.harness import ModelHarness


def main():
    parser = argparse.ArgumentParser(description="Raymember Phase 7 Agent Comparison Benchmark")
    parser.add_argument("--provider", type=str, default="mock", help="Model provider: mock (default), openai, ollama, anthropic")
    parser.add_argument("--model", type=str, default=None, help="Model name (e.g. gpt-4o, llama3, claude-3-5-sonnet-20241022)")
    parser.add_argument("--api-key", type=str, default=None, help="API key for cloud provider")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL for OpenAI-compatible or Ollama endpoints")
    parser.add_argument("--runs", type=int, default=1, help="Number of evaluation runs for statistical confidence intervals")
    parser.add_argument("--out", type=str, default=None, help="Output JSON results filename")
    parser.add_argument("--validate-config", action="store_true", help="Validate model provider configuration offline without making API calls")

    args = parser.parse_args()

    if args.validate_config:
        cfg = ModelHarness.validate_provider_config(
            provider=args.provider,
            model_name=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
        )
        print("=======================================================================")
        print("  Raymember Provider Configuration Offline Validation")
        print("=======================================================================")
        print(f"  Valid:            {cfg['valid']}")
        print(f"  Provider:         {cfg['provider']}")
        print(f"  Real LLM Opt-in:  {cfg['is_real_model']}")
        print(f"  API Key Status:   {cfg['api_key_status']}")
        print(f"  Base URL:         {cfg['base_url']}")
        print(f"  Notes:            {cfg['notes']}")
        print("=======================================================================")
        sys.exit(0 if cfg["valid"] else 1)

    model_fn, model_name, is_real_model = ModelHarness.get_model(
        provider=args.provider,
        model_name=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )

    print("=======================================================================")
    print("  Raymember Phase 7: Real LLM Evaluation & Behavioral Benchmark")
    print("=======================================================================")
    print(f"  Model Provider:    {model_name}")
    print(f"  Real LLM Opt-in:   {is_real_model}")
    print(f"  Evaluation Runs:   {args.runs}")
    print("  Comparing 3 Context Strategies across 32 Evaluation Scenarios:")
    print("    - Strategy A: No Memory (Zero Context)")
    print("    - Strategy B: Naive History (Unfiltered Raw Observation Stream)")
    print("    - Strategy C: Raymember (Relevance-Ranked, Conflict-Aware State Context)")
    print("=======================================================================\n")

    bm = AgentComparisonBenchmark()
    summary = bm.run_benchmark(
        model=model_fn,
        model_name=model_name,
        is_real_model=is_real_model,
        num_runs=args.runs,
    )

    print(json.dumps(summary, indent=2))

    print("\n-----------------------------------------------------------------------")
    print("  BENCHMARK SUMMARY RESULTS")
    print("-----------------------------------------------------------------------")
    for strat, metrics in summary["strategies"].items():
        print(f"\n{strat}:")
        print(f"  Accepted State Accuracy:            {metrics['accepted_state_accuracy']['mean'] * 100:.1f}% (±{metrics['accepted_state_accuracy']['ci_95_margin'] * 100:.1f}%)")
        print(f"  Conflict Interpretation Accuracy:   {metrics['conflict_interpretation_accuracy']['mean'] * 100:.1f}% (±{metrics['conflict_interpretation_accuracy']['ci_95_margin'] * 100:.1f}%)")
        print(f"  Unsupported Fact Rate:             {metrics['unsupported_fact_rate']['mean'] * 100:.1f}% (±{metrics['unsupported_fact_rate']['ci_95_margin'] * 100:.1f}%)")
        print(f"  Contradiction Rate:                 {metrics['contradiction_rate']['mean'] * 100:.1f}%")
        print(f"  Provenance Citation Accuracy:       {metrics['provenance_citation_accuracy']['mean'] * 100:.1f}%")
        print(f"  Average Context Size:               {metrics['avg_context_char_size']} chars (~{metrics['avg_context_tokens']} tokens)")
        print(f"  Average Response Latency:           {metrics['avg_response_latency_ms']} ms")

    out_file = args.out or ("benchmark_results_real_llm.json" if is_real_model else "benchmark_results_deterministic.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to '{out_file}'.")


if __name__ == "__main__":
    main()
