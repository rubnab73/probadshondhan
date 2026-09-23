"""
Automated Pipeline for Situation Generation, Filtering, and Dataset Ingestion
Manages:
1. Identifying pending proverbs that need situations.
2. Filtering generated situations against Lexical Leakage (proverb word containment).
3. Quality check & length verification.
4. Ingestion into dataset/situations_train.csv and dataset/hard_negatives.csv.
"""

import os
import re
import sys
import json
import pandas as pd
from typing import List, Dict, Tuple, Set

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = "dataset"
PROVERBS_CSV = os.path.join(DATA_DIR, "proverbs.csv")
TRAIN_CSV = os.path.join(DATA_DIR, "situations_train.csv")

# Stopwords to exclude when checking lexical leakage
STOPWORDS = {
    "না", "করে", "হলে", "থেকে", "দিয়ে", "তা", "তো", "আর", "ও", "হয়", "যায়", "কে", "কি",
    "এর", "একটি", "এক", "জন্য", "হয়ে", "নয়", "সে", "তার", "তিনি", "তাঁকে", "বা", "বাজে"
}

def get_proverb_keywords(proverb: str) -> List[str]:
    """Extract root words from proverb that must NOT appear in the situation."""
    tokens = re.findall(r'[\u0980-\u09FF]+', proverb)
    keywords = [t for t in tokens if len(t) > 2 and t not in STOPWORDS]
    return keywords

def check_leakage(situation: str, proverb: str) -> List[str]:
    """Returns any leaked keywords found in situation using token boundaries."""
    keywords = get_proverb_keywords(proverb)
    tokens = re.findall(r'[\u0980-\u09FF]+', situation)
    leaked = []
    for kw in keywords:
        for t in tokens:
            if t == kw or (len(kw) >= 3 and t.startswith(kw)):
                leaked.append(kw)
                break
    return list(set(leaked))

def load_data():
    proverbs_df = pd.read_csv(PROVERBS_CSV, encoding="utf-8-sig")
    train_df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig") if os.path.exists(TRAIN_CSV) else pd.DataFrame()
    return proverbs_df, train_df

def get_next_situation_id(train_df: pd.DataFrame) -> int:
    if train_df.empty or "situation_id" not in train_df.columns:
        return 1
    max_id = 0
    for sid in train_df["situation_id"]:
        m = re.search(r'S(\d+)', str(sid))
        if m:
            max_id = max(max_id, int(m.group(1)))
    return max_id + 1

def ingest_batch(batch_data: List[Dict], verbose: bool = True) -> Dict[str, int]:
    """
    Ingests a list of generated situations.
    Schema per item:
    {
      "proverb_id": "P031",
      "direct": "...",
      "indirect": "...",
      "story": "..."
    }
    """
    proverbs_df, train_df = load_data()
    proverb_map = dict(zip(proverbs_df["proverb_id"], proverbs_df["proverb"]))
    
    next_s_id = get_next_situation_id(train_df)
    new_train_rows = []
    rejected_count = 0

    for item in batch_data:
        pid = item.get("proverb_id")
        if pid not in proverb_map:
            print(f"⚠️ Warning: {pid} not found in proverbs.csv. Skipping.")
            continue
        
        proverb_text = proverb_map[pid]

        for stype in ["direct", "indirect", "story"]:
            sit_text = item.get(stype, "").strip()
            if not sit_text:
                continue

            # Automatic filtering
            leaks = check_leakage(sit_text, proverb_text)
            if leaks:
                print(f"❌ [Leakage Rejected] {pid} ({stype}): Found words {leaks} in: \"{sit_text[:50]}...\"")
                rejected_count += 1
                continue

            if len(sit_text) < 25:
                print(f"❌ [Too Short Rejected] {pid} ({stype}): \"{sit_text}\"")
                rejected_count += 1
                continue

            sid = f"S{next_s_id:03d}"
            next_s_id += 1
            new_train_rows.append({
                "situation_id": sid,
                "proverb_id": pid,
                "situation": sit_text,
                "situation_type": stype,
                "source": "Gemini-LLM"
            })

        # Hard negatives omitted as requested

    # Save to files
    if new_train_rows:
        updated_train = pd.concat([train_df, pd.DataFrame(new_train_rows)], ignore_index=True)
        updated_train.to_csv(TRAIN_CSV, index=False, encoding="utf-8-sig")

    if verbose:
        print("=" * 60)
        print(f"✓ Successfully ingested {len(new_train_rows)} training situations.")
        print(f"✓ Rejected {rejected_count} situations due to leakage/quality filters.")
        print(f"✓ Total situations_train count is now: {len(train_df) + len(new_train_rows)}")
        print("=" * 60)

    return {
        "train_added": len(new_train_rows),
        "rejected": rejected_count
    }

if __name__ == "__main__":
    proverbs_df, train_df = load_data()
    covered = set(train_df["proverb_id"].unique()) if not train_df.empty else set()
    all_pids = list(proverbs_df["proverb_id"])
    pending = [p for p in all_pids if p not in covered]
    print(f"Status: {len(covered)} proverbs have situations, {len(pending)} proverbs pending.")
    if pending:
        print(f"Next pending proverbs: {pending[:10]}")
