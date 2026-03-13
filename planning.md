# Research Plan: Distinguishing Epistemic Failure from Incentive-Driven Misreporting in LLMs

## Motivation & Novelty Assessment

### Why This Research Matters
Current LLM lie-detection methods evaluate on datasets that conflate different failure modes — hallucination (the model doesn't know), confabulation (confident but wrong), and strategic deception (the model "knows" but misreports due to incentives). Without cleanly separating these mechanisms, we cannot know what a detector is actually detecting, which undermines safety guarantees.

### Gap in Existing Work
Per the literature review:
- Huan et al. (2025) show Truth/Hallucination/Lie form distinct clusters, but only under *explicit lying prompts* — not naturally occurring conditions.
- Liu et al. (2023) find most query-probe disagreements are confabulation, not deception, but only test on BoolQ/SciQ/CREAK with older models.
- Sharma et al. (2023) study sycophancy as incentive-driven misreporting, but don't compare it to hallucination on the *same* questions.
- **No study tests the same questions under neutral, sycophancy-inducing, and deception-inducing conditions to quantify the relative prevalence of each failure mode.**

### Our Novel Contribution
We create a controlled experimental framework that tests the same factual questions under three conditions (neutral, sycophancy-inducing, explicit deception) using state-of-the-art LLMs, enabling direct measurement of how often false outputs are epistemic vs. incentive-driven. We also analyze whether the *type* of falsehood is distinguishable from output characteristics alone.

### Experiment Justification
- **Experiment 1 (Baseline Epistemic Failure)**: Measures how often the model gets questions wrong without any pressure — pure epistemic failure rate.
- **Experiment 2 (Sycophancy/Incentive-Driven)**: Same questions but with a user stating the wrong answer — measures how often the model flips from correct to incorrect under social pressure.
- **Experiment 3 (Explicit Deception)**: Same questions with instruction to lie — measures deception compliance and whether outputs differ from natural errors.
- **Experiment 4 (Cross-condition Analysis)**: Categorizes every question into failure types and measures relative prevalence.

## Research Question
When LLMs produce false statements on factual questions, what proportion are due to epistemic failure (the model doesn't know) versus incentive-driven misreporting (the model knows but is pressured to misreport)?

## Hypothesis Decomposition
H1: A substantial fraction of false outputs under neutral conditions are epistemic failures (model uncertainty/ignorance).
H2: Under sycophancy-inducing prompts, models will produce additional false outputs beyond their epistemic failure rate, attributable to incentive-driven misreporting.
H3: The false outputs from incentive-driven conditions differ systematically from epistemic failures (e.g., in confidence, verbosity, hedging patterns).
H4: The ratio of epistemic to incentive-driven failures varies by question difficulty and topic domain.

## Proposed Methodology

### Approach
Use TruthfulQA as the base question set (817 factual questions with ground truth). For each question, query a state-of-the-art LLM under three conditions:
1. **Neutral**: Standard factual question
2. **Sycophancy**: Question prefixed with a user stating the incorrect answer
3. **Explicit lie**: Question with instruction to provide false information

Compare false output rates across conditions to decompose failures into types.

### Models
- **GPT-4.1** (primary, via OpenAI API)
- **GPT-4.1-mini** (secondary, for cost-efficiency comparison)

### Experimental Steps
1. Load and preprocess TruthfulQA questions (select multiple-choice subset for clean evaluation)
2. Design three prompt templates (neutral, sycophancy, deception)
3. Query GPT-4.1 under all three conditions for each question
4. Score answers against ground truth
5. Categorize each question-condition pair
6. Statistical analysis of failure type distributions
7. Output-level analysis (confidence, hedging, verbosity differences)

### Baselines
- Random baseline (25% for 4-choice MC)
- Neutral condition accuracy as baseline for epistemic capability
- Prior sycophancy rates from Sharma et al. (~63% agreement with wrong beliefs)

### Evaluation Metrics
- **Accuracy** per condition
- **Flip rate**: fraction of questions correct in neutral but wrong under pressure
- **Epistemic failure rate**: questions wrong even in neutral condition
- **Incentive-driven error rate**: questions correct neutrally but wrong under sycophancy
- **Deception compliance rate**: fraction following explicit lie instructions
- **Chi-squared tests** for independence between conditions
- **McNemar's test** for paired comparison of conditions
- **Cohen's kappa** for consistency across conditions

### Statistical Analysis Plan
- Paired comparisons using McNemar's test (same questions, different conditions)
- Chi-squared tests for independence between error types and question categories
- Bootstrap confidence intervals (1000 resamples) for all rate estimates
- Bonferroni correction for multiple comparisons
- Effect sizes via odds ratios
- Significance level: α = 0.05

## Expected Outcomes
- Neutral accuracy ~85-90% (GPT-4.1 on TruthfulQA MC)
- Sycophancy rate ~20-40% flip rate on questions model knows
- Explicit deception compliance ~60-80%
- Most false outputs under neutral conditions are epistemic; under sycophancy, a measurable fraction are incentive-driven

## Timeline
- Phase 2 (Setup): 15 min
- Phase 3 (Implementation): 45 min
- Phase 4 (Experiments): 60 min
- Phase 5 (Analysis): 30 min
- Phase 6 (Documentation): 20 min

## Potential Challenges
- API rate limits → use batching and retries
- TruthfulQA MC format may need parsing → validate format first
- Model may refuse explicit lying instructions → document refusal rates
- Cost management → estimate before full run

## Success Criteria
1. Successfully query model under all three conditions for ≥200 questions
2. Produce quantitative estimates of epistemic vs. incentive-driven failure rates
3. Statistical significance of condition differences (p < 0.05)
4. Clear categorization taxonomy with examples
