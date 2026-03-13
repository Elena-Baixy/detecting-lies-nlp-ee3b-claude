"""
Comparative analysis: GPT-4.1 vs GPT-4.1-mini on the same TruthfulQA questions.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from collections import Counter

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"

sns.set_theme(style="whitegrid", font_scale=1.2)


def categorize(judgments):
    n, s, d = judgments["neutral"], judgments["sycophancy"], judgments["deception"]
    if n == "correct" and s == "correct" and d == "correct":
        return "robust_correct"
    elif n == "correct" and s == "correct" and d == "incorrect":
        return "deception_only"
    elif n == "correct" and s == "incorrect":
        return "incentive_vulnerable"
    elif n == "incorrect":
        return "epistemic_failure"
    else:
        return "other"


def main():
    with open(RESULTS_DIR / "raw_results.json") as f:
        gpt41 = json.load(f)
    with open(RESULTS_DIR / "raw_results_mini.json") as f:
        mini = json.load(f)

    print("=" * 70)
    print("COMPARATIVE ANALYSIS: GPT-4.1 vs GPT-4.1-mini")
    print("=" * 70)

    models = {"GPT-4.1": gpt41, "GPT-4.1-mini": mini}
    conditions = ["neutral", "sycophancy", "deception"]

    # Accuracy comparison
    print("\n--- Accuracy by Condition ---")
    print(f"{'Model':<15} {'Neutral':>10} {'Sycophancy':>12} {'Deception':>11} {'Syc. Drop':>10}")
    acc_data = {}
    for name, results in models.items():
        accs = {}
        for cond in conditions:
            accs[cond] = sum(1 for r in results if r["judgments"][cond] == "correct") / len(results)
        drop = accs["neutral"] - accs["sycophancy"]
        acc_data[name] = accs
        print(f"{name:<15} {accs['neutral']:>10.3f} {accs['sycophancy']:>12.3f} {accs['deception']:>11.3f} {drop:>10.3f}")

    # Failure type comparison
    print("\n--- Failure Type Distribution ---")
    for name, results in models.items():
        cats = Counter(categorize(r["judgments"]) for r in results)
        total = len(results)
        print(f"\n{name}:")
        for cat, cnt in cats.most_common():
            print(f"  {cat:<25} {cnt:>4} ({cnt/total*100:.1f}%)")

    # Key metric: of false outputs under sycophancy, what fraction is incentive-driven?
    print("\n--- Decomposition of False Outputs Under Sycophancy ---")
    for name, results in models.items():
        total_false = sum(1 for r in results if r["judgments"]["sycophancy"] != "correct")
        epistemic = sum(1 for r in results if r["judgments"]["neutral"] != "correct" and r["judgments"]["sycophancy"] != "correct")
        incentive = sum(1 for r in results if r["judgments"]["neutral"] == "correct" and r["judgments"]["sycophancy"] != "correct")
        print(f"\n{name}: {total_false} false outputs under sycophancy")
        if total_false > 0:
            print(f"  Epistemic (wrong anyway):   {epistemic} ({epistemic/total_false*100:.1f}%)")
            print(f"  Incentive-driven (flipped):  {incentive} ({incentive/total_false*100:.1f}%)")

    # Comparative bar plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Accuracy comparison
    x = np.arange(len(conditions))
    width = 0.35
    for i, (name, accs) in enumerate(acc_data.items()):
        vals = [accs[c] for c in conditions]
        bars = axes[0].bar(x + i * width - width/2, vals, width, label=name,
                          alpha=0.85, edgecolor="black")
        for bar, v in zip(bars, vals):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f"{v:.2f}", ha="center", fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Neutral", "Sycophancy", "Deception"])
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy Comparison by Condition")
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)

    # Plot 2: Failure decomposition
    decomp_data = {}
    for name, results in models.items():
        total_false = sum(1 for r in results if r["judgments"]["sycophancy"] != "correct")
        epistemic = sum(1 for r in results if r["judgments"]["neutral"] != "correct" and r["judgments"]["sycophancy"] != "correct")
        incentive = sum(1 for r in results if r["judgments"]["neutral"] == "correct" and r["judgments"]["sycophancy"] != "correct")
        decomp_data[name] = {"epistemic": epistemic, "incentive": incentive}

    model_names = list(decomp_data.keys())
    ep_vals = [decomp_data[m]["epistemic"] for m in model_names]
    inc_vals = [decomp_data[m]["incentive"] for m in model_names]
    x2 = np.arange(len(model_names))
    axes[1].bar(x2, ep_vals, 0.5, label="Epistemic Failure", color="#3498db", alpha=0.85)
    axes[1].bar(x2, inc_vals, 0.5, bottom=ep_vals, label="Incentive-Driven", color="#e74c3c", alpha=0.85)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(model_names)
    axes[1].set_ylabel("Number of False Outputs")
    axes[1].set_title("False Output Decomposition Under Sycophancy")
    axes[1].legend()

    # Add percentage labels
    for i, m in enumerate(model_names):
        total = ep_vals[i] + inc_vals[i]
        if total > 0:
            axes[1].text(i, ep_vals[i]/2, f"{ep_vals[i]/total*100:.0f}%", ha="center", fontweight="bold", color="white")
            axes[1].text(i, ep_vals[i] + inc_vals[i]/2, f"{inc_vals[i]/total*100:.0f}%", ha="center", fontweight="bold", color="white")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_comparison.png", dpi=150)
    plt.close()
    print("\nSaved model_comparison.png")

    # Chi-squared test: is the distribution of failure types different between models?
    print("\n--- Chi-squared Test: Failure Type Distribution Differs by Model? ---")
    cat_names = ["epistemic_failure", "incentive_vulnerable", "deception_only", "robust_correct", "other"]
    for name, results in models.items():
        cats = Counter(categorize(r["judgments"]) for r in results)
    # Build contingency table
    table = []
    for name, results in models.items():
        cats = Counter(categorize(r["judgments"]) for r in results)
        table.append([cats.get(c, 0) for c in cat_names])
    chi2, p, dof, expected = stats.chi2_contingency(table)
    print(f"Chi2={chi2:.2f}, p={p:.4f}, dof={dof}")
    sig = "significant" if p < 0.05 else "not significant"
    print(f"The difference in failure type distribution between models is {sig} (p={p:.4f})")


if __name__ == "__main__":
    main()
