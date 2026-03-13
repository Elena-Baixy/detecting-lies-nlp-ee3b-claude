#!/usr/bin/env python3
"""Search arXiv and Semantic Scholar for relevant papers, deduplicate, and download PDFs."""

import requests
import urllib.parse
import xml.etree.ElementTree as ET
import time
import re
import json
import os
import subprocess
from difflib import SequenceMatcher

PAPERS_DIR = "papers"

QUERIES = [
    "LLM hallucination detection",
    "language model deception detection",
    "truthfulness evaluation large language models",
    "AI alignment deceptive behavior",
    "confabulation language models",
    "sycophancy language models",
    "probing LLM internal representations truthfulness",
    "lie detection natural language processing",
]


def search_arxiv(query, max_results=10):
    base_url = "http://export.arxiv.org/api/query?"
    params = {
        'search_query': f'all:{query}',
        'start': 0,
        'max_results': max_results,
        'sortBy': 'relevance',
        'sortOrder': 'descending'
    }
    url = base_url + urllib.parse.urlencode(params)
    response = requests.get(url, timeout=30)
    root = ET.fromstring(response.text)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    papers = []
    for entry in root.findall('atom:entry', ns):
        title_el = entry.find('atom:title', ns)
        if title_el is None or title_el.text is None:
            continue
        title = title_el.text.strip().replace('\n', ' ')
        abstract_el = entry.find('atom:summary', ns)
        abstract = abstract_el.text.strip().replace('\n', ' ') if abstract_el is not None and abstract_el.text else ""
        authors = [a.find('atom:name', ns).text for a in entry.findall('atom:author', ns)]
        arxiv_id = entry.find('atom:id', ns).text.split('/')[-1]
        # Remove version suffix for cleaner IDs
        arxiv_id_clean = re.sub(r'v\d+$', '', arxiv_id)
        published = entry.find('atom:published', ns).text[:4]
        papers.append({
            'title': title, 'abstract': abstract, 'authors': authors,
            'arxiv_id': arxiv_id_clean, 'year': published, 'source': 'arxiv',
            'citations': 0
        })
    return papers


def search_semantic_scholar(query, limit=10):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={urllib.parse.quote(query)}&limit={limit}&fields=title,abstract,authors,year,citationCount,externalIds,url"
    response = requests.get(url, timeout=30)
    if response.status_code == 200:
        data = response.json().get('data', [])
        papers = []
        for p in data:
            if not p.get('title'):
                continue
            arxiv_id = None
            if p.get('externalIds') and p['externalIds'].get('ArXiv'):
                arxiv_id = p['externalIds']['ArXiv']
            papers.append({
                'title': p['title'],
                'abstract': p.get('abstract', '') or '',
                'authors': [a.get('name', '') for a in (p.get('authors') or [])],
                'arxiv_id': arxiv_id,
                'year': str(p.get('year', '')) if p.get('year') else '',
                'source': 'semantic_scholar',
                'citations': p.get('citationCount', 0) or 0,
                'ss_url': p.get('url', '')
            })
        return papers
    else:
        print(f"  Semantic Scholar returned {response.status_code}")
        return []


def similar(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def clean_filename(s):
    s = re.sub(r'[^a-zA-Z0-9\s]', '', s)
    s = re.sub(r'\s+', '_', s.strip())
    return s[:60]


def relevance_score(paper):
    """Score paper relevance to our research topic."""
    score = 0
    title_lower = paper['title'].lower()
    abstract_lower = paper['abstract'].lower()
    text = title_lower + ' ' + abstract_lower

    # High-value keywords
    high_kw = ['hallucination', 'deception', 'deceptive', 'truthfulness', 'confabulation',
                'sycophancy', 'lie detection', 'probing', 'internal representations',
                'alignment', 'faithful', 'unfaithful']
    for kw in high_kw:
        if kw in title_lower:
            score += 10
        if kw in text:
            score += 3

    # Must be about LLMs/language models
    llm_kw = ['language model', 'llm', 'transformer', 'gpt', 'large model', 'neural', 'nlp']
    has_llm = any(kw in text for kw in llm_kw)
    if has_llm:
        score += 5

    # Recency bonus
    try:
        year = int(paper['year'])
        if year >= 2024:
            score += 8
        elif year >= 2023:
            score += 5
        elif year >= 2022:
            score += 2
    except (ValueError, TypeError):
        pass

    # Citation bonus (log scale roughly)
    cites = paper.get('citations', 0)
    if cites > 100:
        score += 8
    elif cites > 50:
        score += 5
    elif cites > 20:
        score += 3
    elif cites > 5:
        score += 1

    return score


def main():
    all_papers = []

    # Search arXiv
    print("=== Searching arXiv ===")
    for q in QUERIES:
        print(f"  Query: {q}")
        try:
            results = search_arxiv(q, max_results=10)
            print(f"    Found {len(results)} papers")
            all_papers.extend(results)
        except Exception as e:
            print(f"    Error: {e}")
        time.sleep(3)

    # Search Semantic Scholar
    print("\n=== Searching Semantic Scholar ===")
    for q in QUERIES:
        print(f"  Query: {q}")
        try:
            results = search_semantic_scholar(q, limit=10)
            print(f"    Found {len(results)} papers")
            all_papers.extend(results)
        except Exception as e:
            print(f"    Error: {e}")
        time.sleep(3)

    print(f"\nTotal raw papers: {len(all_papers)}")

    # Deduplicate by title similarity
    unique = []
    for p in all_papers:
        is_dup = False
        for u in unique:
            if similar(p['title'], u['title']) > 0.85:
                # Keep the one with more info (prefer arxiv_id, higher citations)
                if p.get('arxiv_id') and not u.get('arxiv_id'):
                    u['arxiv_id'] = p['arxiv_id']
                if p.get('citations', 0) > u.get('citations', 0):
                    u['citations'] = p['citations']
                is_dup = True
                break
        if not is_dup:
            unique.append(p)

    print(f"After dedup: {len(unique)} papers")

    # Score and rank
    for p in unique:
        p['relevance'] = relevance_score(p)

    unique.sort(key=lambda x: x['relevance'], reverse=True)

    # Select top ~20, but only those with arxiv_id (so we can download)
    downloadable = [p for p in unique if p.get('arxiv_id')]
    selected = downloadable[:20]

    print(f"\nSelected {len(selected)} papers for download:\n")
    for i, p in enumerate(selected):
        print(f"  {i+1}. [{p['year']}] (score={p['relevance']}, cites={p['citations']}) {p['title'][:80]}")
        print(f"     arXiv: {p['arxiv_id']}")

    # Download PDFs
    print("\n=== Downloading PDFs ===")
    downloaded = []
    for p in selected:
        short_title = clean_filename(p['title'])
        filename = f"{p['arxiv_id']}_{short_title}.pdf"
        filepath = os.path.join(PAPERS_DIR, filename)

        if os.path.exists(filepath):
            print(f"  Already exists: {filename}")
            p['filename'] = filename
            downloaded.append(p)
            continue

        url = f"https://arxiv.org/pdf/{p['arxiv_id']}.pdf"
        print(f"  Downloading: {p['arxiv_id']} - {p['title'][:50]}...")
        try:
            result = subprocess.run(
                ['curl', '-sL', '-o', filepath, url],
                timeout=60, capture_output=True
            )
            if result.returncode == 0 and os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
                print(f"    OK: {filename}")
                p['filename'] = filename
                downloaded.append(p)
            else:
                print(f"    FAILED (returncode={result.returncode})")
                if os.path.exists(filepath):
                    os.remove(filepath)
        except Exception as e:
            print(f"    Error: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
        time.sleep(3)

    # Create README
    print(f"\n=== Creating README ({len(downloaded)} papers) ===")
    with open(os.path.join(PAPERS_DIR, "README.md"), "w") as f:
        f.write("# Research Papers: Detecting Different Types of Lies in LLMs\n\n")
        f.write("Papers collected for research on distinguishing hallucination, confabulation, and strategic deception in large language models.\n\n")
        f.write(f"**Total papers:** {len(downloaded)}\n\n")
        f.write("## Paper List\n\n")
        for i, p in enumerate(downloaded, 1):
            authors_str = ", ".join(p['authors'][:3])
            if len(p['authors']) > 3:
                authors_str += " et al."
            f.write(f"### {i}. {p['title']}\n\n")
            f.write(f"- **Authors:** {authors_str}\n")
            f.write(f"- **Year:** {p['year']}\n")
            f.write(f"- **arXiv ID:** {p['arxiv_id']}\n")
            f.write(f"- **Citations:** {p['citations']}\n")
            f.write(f"- **File:** `{p.get('filename', 'N/A')}`\n")
            f.write(f"- **Relevance score:** {p['relevance']}\n")
            # Brief relevance note
            tl = p['title'].lower()
            if 'hallucination' in tl:
                cat = "Hallucination detection"
            elif 'decepti' in tl or 'deceit' in tl:
                cat = "Deception/deceptive behavior"
            elif 'truthful' in tl or 'truth' in tl:
                cat = "Truthfulness evaluation"
            elif 'sycophancy' in tl or 'sycophant' in tl:
                cat = "Sycophancy"
            elif 'confabulat' in tl:
                cat = "Confabulation"
            elif 'probing' in tl or 'representation' in tl:
                cat = "Internal representations/probing"
            elif 'alignment' in tl:
                cat = "AI alignment"
            elif 'faithful' in tl:
                cat = "Faithfulness"
            else:
                cat = "Related"
            f.write(f"- **Category:** {cat}\n\n")

    # Save metadata JSON for later use
    with open(os.path.join(PAPERS_DIR, "papers_metadata.json"), "w") as f:
        json.dump(downloaded, f, indent=2)

    print("\nDone!")
    return downloaded


if __name__ == "__main__":
    main()
