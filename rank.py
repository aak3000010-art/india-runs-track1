"""
rank.py -- India Runs Hackathon (Hard-Gated ML Architect Discovery)

COMPLIANCE:
  - 100% CPU Compliant.
  - Hard-Gated: Backend, Java, Mobile, and non-active candidates are now 
    mathematically impossible to rank.
"""

import argparse
import json
import time
import re
import numpy as np
import pandas as pd
from datetime import datetime

# ----------------------------------------------------------------------------
# CONSTANTS -- LOCKDOWN MODE
# ----------------------------------------------------------------------------

TOP_N = 100
PRE_FILTER_LIMIT = 1500  
CURRENT_HACKATHON_DATE = datetime(2026, 6, 26)

# HARD BAN LIST: Any title containing these words is killed at the gate
NON_TECH_TITLES = {
    "hr", "marketing", "civil", "mechanical", "accountant", "project manager", 
    "operations", "sales", "business analyst", "support", "designer", "admin", 
    "recruiter", "talent", "finance", "content writer", "scrum master", "product manager",
    "qa", "tester", "sdet", "frontend", "front-end", "cloud", "devops", "full stack", 
    "full-stack", "backend", "java", ".net", "mobile", "android", "ios", "analyst"
}
NON_TECH_PATTERN = re.compile(r'\b(?:' + '|'.join(re.escape(w) for w in NON_TECH_TITLES) + r')\b')

# DOMAIN ENFORCEMENT: Candidates must be natively aligned to ML/AI
DOMAIN_TITLE_KEYWORDS = [
    "ml", "machine learning", "ai", "artificial intelligence", "nlp", 
    "data scientist", "data engineer", "search", "recommendation", "vision", "applied scientist"
]

JD_SKILL_TERMS = {
    "search", "ranking", "retrieval", "recommendation", "recommender",
    "embedding", "vector", "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "nlp", "bert", "transformer", "llm",
    "rag", "fine-tun", "lora", "qlora", "peft", "pytorch", "tensorflow",
    "machine learning", "deep learning", "a/b test", "ndcg", "mrr",
    "python", "spark", "airflow", "mlflow", "xgboost",
}

def log(t_start, msg):
    print(f"[{time.time() - t_start:6.1f}s] {msg}")

def build_candidate_text(c):
    profile = c.get("profile", {})
    text_parts = [profile.get("current_title", ""), profile.get("headline", ""), profile.get("summary", "")]
    for job in c.get("career_history", []):
        text_parts.append(job.get("title", ""))
    for skill in c.get("skills", []):
        text_parts.append(skill.get("name", ""))
    return " ".join([p for p in text_parts if p and isinstance(p, str)]).lower()

def notice_period_multiplier(notice_days):
    if notice_days <= 30: return 1.08
    if notice_days >= 90: return 0.50
    span = 90 - 30
    frac = (notice_days - 30) / span
    return 1.08 - frac * (1.08 - 0.50)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--precomputed-dir", required=True)
    args = parser.parse_args()
    t_start = time.time()

    jd_embedding = np.load(f"{args.precomputed_dir}/jd_embedding.npy")
    survivors = []

    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            profile = c.get("profile", {})
            signals = c.get("redrob_signals", {})
            title = str(profile.get("current_title", "")).lower()

            # GATE 1: The "Tourist" Purge
            if NON_TECH_PATTERN.search(title): continue
            
            # GATE 2: Ghost Protocol (Active only)
            if signals.get("open_to_work_flag") is False: continue
            
            # GATE 3: Experience
            yoe = profile.get("years_of_experience", 0)
            if yoe < 2 or yoe > 14: continue

            # HEURISTIC
            full_text = build_candidate_text(c)
            k_count = sum(1 for term in JD_SKILL_TERMS if term in full_text)
            if k_count < 2: continue
            
            # SCORING
            mult = notice_period_multiplier(signals.get("notice_period_days", 90))
            # Boost for AI/ML Titles
            boost = 1.2 if any(kw in title for kw in DOMAIN_TITLE_KEYWORDS) else 1.0
            
            survivors.append({
                "candidate_id": c["candidate_id"],
                "text": full_text,
                "pre_score": min(k_count, 5) * mult,
                "boost": boost,
                "raw_data": c
            })

    # AI RANKING
    survivors.sort(key=lambda x: x["pre_score"], reverse=True)
    elite = survivors[:PRE_FILTER_LIMIT]
    
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([c["text"] for c in elite], convert_to_numpy=True)
    
    jd_norm = jd_embedding.flatten() / np.linalg.norm(jd_embedding)
    sims = (embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)) @ jd_norm

    results = []
    for i, cand in enumerate(elite):
        final_score = float(sims[i]) * cand["boost"]
        c_raw = cand["raw_data"]
        results.append({
            "candidate_id": cand["candidate_id"],
            "score": round(final_score, 4),
            "reasoning": f"{c_raw['profile'].get('years_of_experience',0):.1f}yr experience {c_raw['profile'].get('current_title')} in {c_raw['profile'].get('location')}."
        })

    df = pd.DataFrame(results).sort_values(by="score", ascending=False).head(TOP_N)
    df.insert(1, "rank", range(1, len(df) + 1))
    df.to_csv(args.out, index=False)
    log(t_start, "Pipeline Finished.")

if __name__ == "__main__":
    main()
