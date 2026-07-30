"""Phase 3 Benchmark Demonstration running 5 systems across seeds for ID & OOD conditions."""

from raymember.evaluation.diagnostics import (
    run_calibration_analysis,
    run_dataset_diagnostics,
    run_error_taxonomy,
    run_hybrid_policy_audit,
    run_hybrid_sensitivity_test,
    run_model_diagnostics,
)
from raymember.evaluation.metrics import BenchmarkRunner

def main():
    print("=== Raymember Phase 3 Scientific Validation & Research Benchmark Harness ===")
    print("\n1. Running Dataset, Model, Hybrid Audit, Calibration, & Error Diagnostics...")
    run_dataset_diagnostics(random_seed=42)
    run_model_diagnostics(random_seed=42)
    run_hybrid_policy_audit(random_seed=42)
    run_hybrid_sensitivity_test()
    run_calibration_analysis(random_seed=42)
    run_error_taxonomy(random_seed=42)

    print("\n2. Executing 5-System Multi-Seed Benchmark across Random Seeds [42, 7, 21, 84, 123]...")
    runner = BenchmarkRunner(dataset_size="medium")
    summary_tree = runner.run_multi_seed_benchmark()
    runner.run_feature_ablations()
    runner.generate_research_report()

    print("\n" + "="*95)
    print(f"{'CONDITION':<15} | {'SYSTEM':<28} | {'ACCURACY (mean±std)':<20} | {'95% CI':<20}")
    print("="*95)

    for cond, sys_dict in summary_tree.items():
        for sys_name, m in sys_dict.items():
            disp_sys = sys_name.split("_", 1)[1]
            acc_str = f"{m['mean_accuracy']:.4f} ± {m['std_accuracy']:.4f}"
            ci_str = f"[{m['ci95_accuracy'][0]:.4f}, {m['ci95_accuracy'][1]:.4f}]"
            print(f"{cond:<15} | {disp_sys:<28} | {acc_str:<20} | {ci_str:<20}")
        print("-" * 95)

    print("\nPhase 3 Benchmark & Diagnostic artifacts generated:")
    print("  - results/dataset_diagnostics.json")
    print("  - results/model_diagnostics.json")
    print("  - results/hybrid_policy_audit.json")
    print("  - results/hybrid_sensitivity.json")
    print("  - results/calibration_results.json")
    print("  - results/error_analysis.json")
    print("  - results/multi_seed_benchmark.json")
    print("  - results/benchmark_summary.json")
    print("  - results/ablation_results.json")
    print("  - results/phase3_research_report.md")

if __name__ == "__main__":
    main()
