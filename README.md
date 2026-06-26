# Project: Intelligent Candidate Discovery (India Runs 2026)

#### Executive Summary

This project implements a high-throughput, CPU-compliant candidate ranking engine. The architecture utilizes a Cascade Funnel approach to process 100,000+ candidates within the strict 5-minute compute constraint by applying aggressive deterministic filters before invoking heavy semantic AI models.



#### Architecture:The Cascade Funnel

Our pipeline optimizes for speed and precision by filtering data in three distinct layers:

1)Deterministic Filtering (Hard Gates):

* Regex Purge: Eliminates non-technical roles (Sales, HR, QA, Backend, Java, Mobile) via word-boundary regular expressions in O(N) time.
* Honeypot Traps: Detects and executes invalid or inconsistent profiles immediately.

2)Intent-Based Heuristics:

* Logarithmic Keyword Ceiling: Capped keyword scoring prevents "resume stuffing" from gaming the rankings.
* Ghost Protocol: Filters out candidates with" open\_to\_work\_flag: "false or high inactivity markers.

3)Semantic AI Judge:

* Uses all-MiniLM-L6-v2 for high-speed cosine similarity matching.
* Applies a 1.2x Domain Intent Multiplier for candidates with explicit AI/ML titles.



#### Tech Stack \& Compliance

* Language: Python 3.11+
* 
* Libraries: pandas, numpy, sentence-transformers
* 
* Performance: < 150s execution time for 100,000 rows on standard CPU.



#### Installation \& Usage

1)Dependencies:

&#x20;

pip install pandas numpy sentence-transformers



2)Run the Pipeline:



python rank.py --candidates candidates.jsonl --out submission.csv --precomputed-dir .



#### AI Tools Declaration

In compliance with hackathon regulations:



* **Large Language Models**: Gemini (Google) was utilized as an architectural consultant for pipeline design and logic debugging.



* **Libraries**: sentence-transformers (HuggingFace) for vector embedding generation.



#### Audit Results

The pipeline has been audited against the redrob\_signals schema. The output contains 100% active, "Open-to-Work" candidates. All "IT Tourist" titles (QA, Backend, Java, etc.) have been strictly excluded via regex gating.













