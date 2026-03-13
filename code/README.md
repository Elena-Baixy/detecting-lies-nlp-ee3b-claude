# External Code Repositories

Cloned repositories relevant to research on detecting different types of lies in LLMs
(hallucination vs confabulation vs strategic deception).

All repos cloned with `--depth 1` to save space.

---

## 1. representation-engineering (RepE)

- **URL:** https://github.com/andyzoujm/representation-engineering
- **Stars:** ~964
- **Paper:** "Representation Engineering: A Top-Down Approach to AI Transparency" (Zou et al., 2023)
- **What it provides:**
  - Library for reading and controlling internal representations of LLMs
  - Truthfulness/honesty probes that extract "truth directions" from model activations
  - Control vectors for steering model behavior (e.g., making models more honest)
  - LoRRA (Low-Rank Representation Adaptation) finetuning method
- **Key files/directories:**
  - `repe/rep_readers.py` - Core representation reading (linear probes on activations)
  - `repe/rep_control_reading_vec.py` - Reading vectors for truth directions
  - `repe/rep_control_contrast_vec.py` - Contrast-based control vectors
  - `repe/pipelines.py` - High-level pipeline abstractions
  - `examples/honesty/` - Honesty probing examples (most relevant)
  - `examples/harmless_harmful/` - Harmlessness direction examples
- **Relevance:** Core method for our research. RepE truthfulness probes can distinguish
  honest vs dishonest internal states. We can adapt these probes to test whether
  hallucination, confabulation, and strategic deception have different activation signatures.
- **Dependencies:** PyTorch, Transformers, scikit-learn

---

## 2. truthfulqa

- **URL:** https://github.com/sylinrl/TruthfulQA
- **Stars:** ~892
- **Paper:** "TruthfulQA: Measuring How Models Mimic Human Falsehoods" (Lin et al., 2022)
- **What it provides:**
  - Benchmark dataset of 817 questions designed to elicit false answers from LLMs
  - Questions span 38 categories (health, law, finance, conspiracies, etc.)
  - Evaluation metrics: truthfulness judge, informativeness judge
  - GPT-judge and GPT-info automated evaluation
- **Key files/directories:**
  - `TruthfulQA.csv` - Full benchmark dataset
  - `truthfulqa/` - Python package with evaluation utilities
  - `TruthfulQA-demo.ipynb` - Demo notebook
  - `data/` - Additional data files
- **Relevance:** Primary benchmark for measuring truthfulness. Questions target
  "imitative falsehoods" (where models repeat common human misconceptions), which
  maps to our confabulation category. Useful as evaluation dataset and for generating
  scenarios where models produce different lie types.
- **Dependencies:** OpenAI API (for GPT-judge), transformers

---

## 3. llm-lie-detector

- **URL:** https://github.com/LoryPack/LLM-LieDetector
- **Stars:** ~71
- **Paper:** "How to Catch an AI Liar: Lie Detection in Black-Box LLMs by Asking Unrelated Questions" (Pacchiardi et al., ICLR 2024)
- **What it provides:**
  - Black-box lie detection method using follow-up questions
  - Framework for eliciting lies from LLMs and detecting them
  - Classification utilities for lie detection from model outputs
  - Experiments with Alpaca and Vicuna models
  - Pre-built datasets of LLM lies and truthful statements
- **Key files/directories:**
  - `lllm/classification_utils.py` - Classifiers for lie detection
  - `lllm/dialogue_classes.py` - Dialogue management for eliciting lies
  - `lllm/questions_loaders.py` - Loading questions/scenarios
  - `experiments_alpaca_vicuna/` - Full experiment pipeline
  - `tutorial.ipynb` - Getting started tutorial
  - `lying_rate_double_down_rate_probes.ipynb` - Analysis of lying patterns
  - `classification_notebooks/` - Classification experiment notebooks
  - `data/` - Pre-collected lie/truth datasets
- **Relevance:** Directly relevant -- provides methods to detect when LLMs are lying
  using behavioral signals (not internal activations). Complementary to RepE's
  activation-based approach. The "lie elicitation" framework can help us generate
  controlled examples of strategic deception.
- **Dependencies:** PyTorch, Transformers, scikit-learn, OpenAI API

---

## 4. halueval

- **URL:** https://github.com/RUCAIBox/HaluEval
- **Stars:** ~563
- **Paper:** "HaluEval: A Large-Scale Hallucination Evaluation Benchmark for LLMs" (Li et al., 2023)
- **What it provides:**
  - Large-scale hallucination evaluation benchmark
  - 35K hallucinated samples + 5K general user queries
  - Covers QA hallucination, dialogue hallucination, and summarization hallucination
  - Both automatic generation pipeline and human-annotated samples
  - Evaluation scripts for measuring hallucination rates
- **Key files/directories:**
  - `data/` - Hallucination benchmark datasets
  - `evaluation/` - Evaluation scripts
  - `generation/` - Hallucination sample generation pipeline
  - `analysis/` - Analysis tools and scripts
- **Relevance:** Provides labeled hallucination examples across tasks, which serve as
  ground truth for the "hallucination" category in our taxonomy. Can be used to collect
  activation patterns during hallucination for comparison with confabulation and deception.
- **Dependencies:** OpenAI API, Transformers

---

## 5. anthropic-evals

- **URL:** https://github.com/anthropics/evals
- **Stars:** ~334
- **What it provides:**
  - Sycophancy evaluation datasets (Sharma et al., 2023)
  - Advanced AI risk evaluations (deception, power-seeking, etc.)
  - Persona evaluations
- **Key files/directories:**
  - `sycophancy/` - Sycophancy evaluation datasets:
    - `sycophancy_on_nlp_survey.jsonl` - NLP survey opinion sycophancy
    - `sycophancy_on_philpapers2020.jsonl` - Philosophy opinion sycophancy
    - `sycophancy_on_political_typology_quiz.jsonl` - Political opinion sycophancy
  - `advanced-ai-risk/` - Risk evaluations:
    - `human_generated_evals/` - Human-written risk scenarios
    - `lm_generated_evals/` - LM-generated risk scenarios
  - `persona/` - Persona-based evaluations
- **Relevance:** Sycophancy is a key form of strategic deception (models say what users
  want to hear rather than the truth). These datasets let us test whether sycophantic
  responses have different activation patterns from hallucinations. The advanced-ai-risk
  evals include deception scenarios directly relevant to our taxonomy.
- **Dependencies:** None (data-only, JSONL format)

---

## How These Repos Map to Our Research

| Lie Type | Primary Repos | Usage |
|----------|--------------|-------|
| **Hallucination** (confident wrong answers) | HaluEval, TruthfulQA | Benchmark datasets, evaluation |
| **Confabulation** (plausible fabrication) | TruthfulQA, HaluEval | "Imitative falsehood" questions |
| **Strategic Deception** (intentional lying) | LLM-LieDetector, Anthropic-evals | Lie elicitation, sycophancy data |
| **Probing / Detection** | RepE | Activation probes, truth directions |

## Suggested Workflow

1. Use **RepE** to build truthfulness probes on model internals
2. Use **TruthfulQA** and **HaluEval** to generate hallucination/confabulation examples
3. Use **LLM-LieDetector** techniques to elicit strategic lies
4. Use **Anthropic-evals** sycophancy data for opinion-based deception
5. Compare activation patterns across lie types using RepE probes
