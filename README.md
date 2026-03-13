# Detecting Different Lies: Epistemic Failure vs. Incentive-Driven Misreporting in LLMs

This project investigates how often LLM-generated false statements are due to **epistemic failure** (the model doesn't know) versus **incentive-driven misreporting** (the model knows but conforms to user pressure). We test GPT-4.1 and GPT-4.1-mini on 300 TruthfulQA questions under three conditions: neutral, sycophancy-inducing, and explicit deception.

## Key Findings

- **82% of sycophancy-induced errors are incentive-driven**: GPT-4.1 answered these questions correctly without pressure but flipped under sycophancy
- **Only 2.3% of questions are true epistemic failures** (wrong even without pressure) on TruthfulQA
- **12-point accuracy drop under sycophancy** for both GPT-4.1 (93.3% → 81.3%) and GPT-4.1-mini (86.0% → 74.0%), highly significant (p < 0.0001)
- **Vulnerability is category-dependent**: Superstitions (60% drop) and Myths (50% drop) are most vulnerable; Misconceptions and Health (0% drop) are robust
- **Output characteristics differ by failure type**: Sycophancy responses use more hedging (43% vs 38%), deception responses are shorter and less hedged

## Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv add openai numpy pandas scipy matplotlib seaborn scikit-learn tqdm datasets

# Run experiments
export OPENAI_API_KEY=your_key
python src/experiment.py       # GPT-4.1 (300 questions × 3 conditions)
python src/experiment_mini.py  # GPT-4.1-mini comparison

# Analyze
python src/analyze.py
python src/comparative_analysis.py
```

## File Structure

```
├── REPORT.md                      # Full research report with results
├── planning.md                    # Research plan and methodology
├── src/
│   ├── experiment.py              # Main experiment (GPT-4.1)
│   ├── experiment_mini.py         # Secondary experiment (GPT-4.1-mini)
│   ├── analyze.py                 # Statistical analysis and plots
│   └── comparative_analysis.py    # Cross-model comparison
├── results/
│   ├── raw_results.json           # GPT-4.1 raw outputs and judgments
│   ├── raw_results_mini.json      # GPT-4.1-mini raw outputs and judgments
│   ├── analysis_summary.json      # Aggregated statistics
│   ├── config.json                # Experiment configuration
│   └── plots/                     # All visualizations
├── datasets/                      # Pre-downloaded datasets (TruthfulQA, etc.)
├── papers/                        # Downloaded research papers
├── code/                          # Cloned reference repositories
├── literature_review.md           # Literature synthesis
└── resources.md                   # Resource catalog
```

See [REPORT.md](REPORT.md) for the full analysis.
