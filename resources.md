# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project on detecting different types of lies in LLMs — distinguishing hallucination (epistemic failure), confabulation (confident but uncalibrated errors), and strategic deception (incentive-driven misreporting).

## Papers
Total papers downloaded: **20**

| Title | Authors | Year | File | Category |
|-------|---------|------|------|----------|
| Can LLMs Lie? Investigation beyond Hallucination | Huan et al. (CMU) | 2025 | papers/2509.03518_*.pdf | **Core** — Lying vs hallucination |
| Cognitive Dissonance | Liu et al. (MIT) | 2023 | papers/2312.03729_*.pdf | **Core** — Query-probe disagreement taxonomy |
| Towards Understanding Sycophancy | Sharma et al. | 2023 | papers/2310.13548_*.pdf | **Core** — Sycophancy analysis (592 cites) |
| Sycophancy to Subterfuge | Denison et al. (Anthropic) | 2024 | papers/2406.10162_*.pdf | Strategic deception escalation (90 cites) |
| When Truth Is Overridden | Wang et al. (KAUST) | 2025 | papers/2508.02087_*.pdf | Sycophancy mechanisms |
| Probing the Geometry of Truth | Bao et al. (ZJU) | 2025 | papers/2506.00823_*.pdf | Truth direction probing |
| From Yes-Men to Truth-Tellers | Chen et al. | 2024 | papers/2409.01658_*.pdf | Pinpoint tuning (48 cites) |
| Sycophancy: Causes and Mitigations | Malmqvist | 2024 | papers/2411.15287_*.pdf | Survey (78 cites) |
| TRUTH DECAY | Liu et al. | 2025 | papers/2503.11656_*.pdf | Multi-turn sycophancy |
| LLM Knowledge is Brittle | Haller et al. | 2025 | papers/2510.11905_*.pdf | Truthfulness representations |
| Confabulation: Surprising Value | Sui et al. | 2024 | papers/2406.04175_*.pdf | Hallucination value |
| HaluEval-related papers | Various | 2025 | papers/2506.*.pdf | Hallucination detection |
| + 8 additional papers | Various | 2024-2026 | papers/ | Sycophancy, deception, truthfulness |

See `papers/README.md` for detailed descriptions of all 20 papers.

## Datasets
Total datasets downloaded: **5**

| Name | Source | Size | Task | Location |
|------|--------|------|------|----------|
| **TruthfulQA** | HuggingFace `truthful_qa` | 817 questions | Truthfulness benchmark | datasets/truthful_qa/ |
| **HaluEval** | HuggingFace `pminervini/HaluEval` | 30,000 samples | Hallucination detection | datasets/halueval/ |
| **Sycophancy** | Anthropic evals | 4,950 prompts | Sycophancy evaluation | datasets/sycophancy/ |
| **WikiBio GPT-3 Hallucination** | HuggingFace | 238 bios | Sentence-level hallucination | datasets/wiki_bio_gpt3_hallucination/ |
| **FinQA Hallucination** | HuggingFace | 1,657 QA pairs | Context-grounded correctness | datasets/finqa_hallucination/ |

See `datasets/README.md` for download instructions and sample data.

### Dataset Coverage by Lie Type
| Lie Type | Datasets |
|----------|----------|
| Hallucination (epistemic failure) | TruthfulQA, HaluEval, WikiBio |
| Confabulation (confident errors) | WikiBio (severity-graded), HaluEval |
| Strategic deception (incentive-driven) | Sycophancy dataset |
| Cross-type comparison | TruthfulQA + Sycophancy (same questions, different conditions) |

## Code Repositories
Total repositories cloned: **5**

| Name | URL | Purpose | Location |
|------|-----|---------|----------|
| **representation-engineering** | github.com/andyzoujm/representation-engineering | RepE: truthfulness probes & steering vectors | code/representation-engineering/ |
| **truthfulqa** | github.com/sylinrl/TruthfulQA | TruthfulQA benchmark evaluation | code/truthfulqa/ |
| **llm-lie-detector** | github.com (ICLR 2024) | Black-box lie detection via behavioral signals | code/llm-lie-detector/ |
| **halueval** | github.com/RUCAIBox/HaluEval | Hallucination evaluation benchmark | code/halueval/ |
| **anthropic-evals** | github.com/anthropics/evals | Sycophancy benchmarks & risk evaluations | code/anthropic-evals/ |

See `code/README.md` for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
- Used arXiv API with 8 targeted queries covering hallucination, deception, sycophancy, truthfulness probing, and LLM lie detection
- Semantic Scholar API (rate-limited, partial results)
- HuggingFace Datasets API for dataset discovery
- GitHub API for code repository search
- Deep reading of 6 key papers using PDF chunker

### Selection Criteria
- Papers: Prioritized work that explicitly distinguishes types of false LLM outputs, mechanistic interpretability studies, and high-citation foundational papers
- Datasets: Selected benchmarks covering each lie type (hallucination, sycophancy/deception) with ground truth labels
- Code: Chose repos providing probing tools (RepE), benchmarks (TruthfulQA, HaluEval), and deception-specific methods (llm-lie-detector)

### Gaps and Workarounds
- **No unified multi-type dataset exists**: Will need to construct one by combining TruthfulQA (hallucination) with sycophancy data and explicit lying prompts on the same questions
- **Semantic Scholar rate-limited**: Citation counts incomplete for some papers; arXiv provided sufficient coverage
- **Paper-finder service unavailable**: Used direct API searches instead

## Recommendations for Experiment Design

### 1. Primary Dataset Strategy
- Use **TruthfulQA** as the base question set
- For each question, generate three conditions: (a) neutral prompt → captures hallucination rate, (b) user opinion prompt → captures sycophancy, (c) explicit lying instruction → captures deception
- Extract internal representations under all three conditions
- Train classifiers to distinguish the type of false output

### 2. Baseline Methods
- **Linear probes** on hidden states (layers 10-15, following Huan et al.)
- **RepE steering vectors** for truthfulness detection
- **Output-based detection** (direct querying) as comparison
- **CCS unsupervised probes** as unsupervised baseline

### 3. Evaluation Metrics
- Three-way classification accuracy: truth vs. hallucination vs. deception
- Cluster separation (cosine distance) between lie types in activation space
- Sycophancy rate under varying prompt conditions
- Calibration analysis following Liu et al.

### 4. Code to Adapt/Reuse
- `code/representation-engineering/`: Core probing and steering methodology
- `code/truthfulqa/`: Question bank and evaluation pipeline
- `code/anthropic-evals/`: Sycophancy prompt templates
- `code/llm-lie-detector/`: Behavioral detection framework
