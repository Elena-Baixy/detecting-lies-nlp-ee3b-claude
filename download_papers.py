#!/usr/bin/env python3
"""Download the selected papers using curl."""
import subprocess
import os
import time
import json
import re

PAPERS_DIR = "papers"

papers = [
    {"arxiv_id": "2508.19432", "title": "Quantized but Deceptive? A Multi-Dimensional Truthfulness Evaluation of Quantized LLMs", "authors": [], "year": "2025", "citations": 0, "relevance": 45},
    {"arxiv_id": "2312.03729", "title": "Cognitive Dissonance: Why Do Language Model Outputs Disagree with Internal Representations of Truthfulness?", "authors": [], "year": "2023", "citations": 0, "relevance": 42},
    {"arxiv_id": "2406.04175", "title": "Confabulation: The Surprising Value of Large Language Model Hallucinations", "authors": [], "year": "2024", "citations": 0, "relevance": 39},
    {"arxiv_id": "2403.09676", "title": "Unmasking the Shadows of AI: Investigating Deceptive Capabilities in Large Language Models", "authors": [], "year": "2024", "citations": 0, "relevance": 38},
    {"arxiv_id": "2411.15287", "title": "Sycophancy in Large Language Models: Causes and Mitigations", "authors": [], "year": "2024", "citations": 78, "relevance": 37},
    {"arxiv_id": "2310.13548", "title": "Towards Understanding Sycophancy in Language Models", "authors": [], "year": "2023", "citations": 592, "relevance": 34},
    {"arxiv_id": "2509.03518", "title": "Can LLMs Lie? Investigation beyond Hallucination", "authors": [], "year": "2025", "citations": 0, "relevance": 32},
    {"arxiv_id": "2408.11261", "title": "Sycophancy in Vision-Language Models: A Systematic Analysis and an Inference-Time Mitigation Method", "authors": [], "year": "2024", "citations": 0, "relevance": 32},
    {"arxiv_id": "2506.00823", "title": "Probing the Geometry of Truth: Consistency and Generalization of Truth Directions in LLMs", "authors": [], "year": "2025", "citations": 0, "relevance": 32},
    {"arxiv_id": "2510.11905", "title": "LLM Knowledge is Brittle: Truthfulness Representations Rely on Superficial Resemblance", "authors": [], "year": "2025", "citations": 0, "relevance": 32},
    {"arxiv_id": "2505.23840", "title": "Measuring Sycophancy of Language Models in Multi-turn Dialogues", "authors": [], "year": "2025", "citations": 32, "relevance": 32},
    {"arxiv_id": "2406.10162", "title": "Sycophancy to Subterfuge: Investigating Reward-Tampering in Large Language Models", "authors": [], "year": "2024", "citations": 90, "relevance": 31},
    {"arxiv_id": "2508.02087", "title": "When Truth Is Overridden: Uncovering the Internal Origins of Sycophancy in Large Language Models", "authors": [], "year": "2025", "citations": 16, "relevance": 30},
    {"arxiv_id": "2506.00448", "title": "Fact-Controlled Diagnosis of Hallucinations in Medical Text Summarization", "authors": [], "year": "2025", "citations": 0, "relevance": 29},
    {"arxiv_id": "2409.01658", "title": "From Yes-Men to Truth-Tellers: Addressing Sycophancy in Large Language Models with Pinpoint Tuning", "authors": [], "year": "2024", "citations": 48, "relevance": 29},
    {"arxiv_id": "2512.00656", "title": "Sycophancy Claims about Language Models: The Missing Human-in-the-Loop", "authors": [], "year": "2025", "citations": 4, "relevance": 29},
    {"arxiv_id": "2503.11656", "title": "TRUTH DECAY: Quantifying Multi-Turn Sycophancy in Language Models", "authors": [], "year": "2025", "citations": 13, "relevance": 27},
    {"arxiv_id": "2506.09886", "title": "Probabilistic distances-based hallucination detection in LLMs with RAG", "authors": [], "year": "2025", "citations": 0, "relevance": 26},
    {"arxiv_id": "2506.22486", "title": "Hallucination Detection with Small Language Models", "authors": [], "year": "2025", "citations": 0, "relevance": 26},
]


def clean_filename(s):
    s = re.sub(r'[^a-zA-Z0-9\s]', '', s)
    s = re.sub(r'\s+', '_', s.strip())
    return s[:60]


downloaded = []
for p in papers:
    short_title = clean_filename(p['title'])
    filename = f"{p['arxiv_id']}_{short_title}.pdf"
    filepath = os.path.join(PAPERS_DIR, filename)
    p['filename'] = filename

    if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
        print(f"  Already exists: {filename}")
        downloaded.append(p)
        continue

    url = f"https://arxiv.org/pdf/{p['arxiv_id']}.pdf"
    print(f"  Downloading: {p['arxiv_id']} - {p['title'][:60]}...")
    try:
        result = subprocess.run(
            ['curl', '-sL', '-o', filepath, url],
            timeout=60, capture_output=True
        )
        if result.returncode == 0 and os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
            size_kb = os.path.getsize(filepath) // 1024
            print(f"    OK ({size_kb} KB)")
            downloaded.append(p)
        else:
            print(f"    FAILED")
            if os.path.exists(filepath):
                os.remove(filepath)
    except Exception as e:
        print(f"    Error: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
    time.sleep(3)

print(f"\nDownloaded {len(downloaded)}/{len(papers)} papers")

# Save metadata
with open(os.path.join(PAPERS_DIR, "papers_metadata.json"), "w") as f:
    json.dump(downloaded, f, indent=2)

print("Metadata saved.")
