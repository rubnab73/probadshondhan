import sys
import os
import time
sys.stdout.reconfigure(encoding="utf-8")
sys.path.append(os.path.abspath("."))

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "csebuetnlp/banglabert"
CACHE_DIR = "dataset/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

proverbs_df = pd.read_csv("dataset/proverbs.csv", encoding="utf-8-sig")
train_df = pd.read_csv("dataset/situations_train.csv", encoding="utf-8-sig")

print(f"Loading {MODEL_NAME}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def encode_texts(texts, batch_size=32):
    all_embeddings = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            emb = mean_pooling(outputs, inputs["attention_mask"])
            emb = F.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb.cpu().numpy())
        if (i + batch_size) % 256 == 0 or (i + batch_size) >= total:
            print(f"  Encoded {min(i+batch_size, total)}/{total} texts...")
    return np.vstack(all_embeddings)

# 1. Proverbs encoding (proverb + meaning)
proverb_cache_path = os.path.join(CACHE_DIR, "banglabert_proverbs.npy")
if not os.path.exists(proverb_cache_path):
    print("Encoding 500 proverbs...")
    t0 = time.time()
    proverb_texts = [f"{row['proverb']}: {row['meaning']}" for _, row in proverbs_df.iterrows()]
    proverb_embs = encode_texts(proverb_texts, batch_size=32)
    np.save(proverb_cache_path, proverb_embs)
    print(f"Saved proverb embeddings ({proverb_embs.shape}) in {time.time()-t0:.1f}s")
else:
    print("Loading cached proverb embeddings...")
    proverb_embs = np.load(proverb_cache_path)

# 2. Training situations encoding
situation_cache_path = os.path.join(CACHE_DIR, "banglabert_situations.npy")
needs_sit_compute = True
if os.path.exists(situation_cache_path):
    try:
        sit_embs = np.load(situation_cache_path)
        if len(sit_embs) == len(train_df):
            print(f"Loading cached situation embeddings ({sit_embs.shape})...")
            needs_sit_compute = False
        else:
            print(f"Cache size ({len(sit_embs)}) does not match dataset situations ({len(train_df)}). Recomputing...")
    except Exception as e:
        print(f"Error loading cache: {e}. Recomputing...")

if needs_sit_compute:
    print(f"Encoding {len(train_df)} training situations...")
    t0 = time.time()
    situations = train_df["situation"].tolist()
    sit_embs = encode_texts(situations, batch_size=32)
    np.save(situation_cache_path, sit_embs)
    print(f"Saved situation embeddings ({sit_embs.shape}) in {time.time()-t0:.1f}s")

print("Precomputation complete! Both proverb and situation embeddings ready.")
