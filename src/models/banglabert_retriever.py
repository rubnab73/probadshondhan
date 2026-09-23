"""
BanglaBERT Contextual Transformer Retriever for Situation-to-Proverb Semantic Retrieval.
Uses pre-trained BanglaBERT (csebuetnlp/banglabert)
to extract 768-dimensional contextual embeddings with dual-layer (proverb + situation kNN) scoring.

DIMENSIONS OVERVIEW:
- Hidden / Embedding Dimension: D_bert = 768
- Token Sequence Length: L_seq <= 128
- Batch Size: B = 16 (or 32)
- Input Token IDs: (B, L_seq)
- Token Hidden Representations: (B, L_seq, 768)
- Mean-Pooled Sentence Embeddings: (B, 768)
- Proverb Knowledgebase Embeddings: (500, 768)
- Training Situations Embeddings: (1510, 768)
- Query Embedding: (1, 768)
- Similarity Dot-Product Output: (500,)
"""

from typing import List, Dict, Any, Optional
import os
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class BanglaBERTRetriever:
    """Transformer-based dense semantic retriever using BanglaBERT with dual-layer kNN matching."""

    def __init__(self, model_name: str = "csebuetnlp/banglabert", device: Optional[str] = None):
        self.model_name = model_name
        self.proverbs_df: Optional[pd.DataFrame] = None
        self.proverb_embeddings: Optional[np.ndarray] = None  # Shape: (500, 768)
        self.sit_embeddings: Optional[np.ndarray] = None      # Shape: (1510, 768)
        self.sit_pids: List[str] = []                         # Length: 1510
        self.pid_to_idx: Dict[str, int] = {}                  # Size: 500
        
        if device is None:
            self.device = "cuda" if (TRANSFORMERS_AVAILABLE and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device
            
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        self.cache_dir = "dataset/cache"

    def _load_model(self):
        """Lazy loader for transformer model."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "PyTorch or Transformers is not available in the current environment. "
                "Install them via 'pip install torch transformers'."
            )
        if not self.is_loaded:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            self.is_loaded = True

    def _mean_pooling(self, model_output, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean pooling: averages token vectors taking attention mask into account.
        
        Dimensions:
            - model_output[0] (token_embeddings): (B, L_seq, 768)
            - attention_mask:                     (B, L_seq)
            - input_mask_expanded:                (B, L_seq, 768)
            - sum_embeddings:                     (B, 768)
            - sum_mask:                           (B, 768)
            - return pooled:                      (B, 768)
        """
        token_embeddings = model_output[0] # Tensor Shape: (B, L_seq, 768)
        
        # Expand mask from (B, L_seq) to (B, L_seq, 768)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum along sequence length dimension (dim=1) -> Tensor Shape: (B, 768)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) # (B, 768)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)          # (B, 768)
        
        return sum_embeddings / sum_mask # Tensor Shape: (B, 768)

    def encode(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Encodes a list of N Bangla texts into 768-dimensional normalized embeddings.
        
        Dimensions:
            - texts:             List[str] of length N
            - batch_texts:       List[str] of size B (B <= batch_size)
            - encoded_input:     Dict with tensors of shape (B, L_seq) where L_seq <= 128
            - sentence_embeds:   (B, 768)
            - return array:      (N, 768)
        """
        self._load_model()
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size] # Length: B (e.g. 16)
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(self.device) # 'input_ids' Shape: (B, L_seq), 'attention_mask' Shape: (B, L_seq)

            with torch.no_grad():
                model_output = self.model(**encoded_input) # Last hidden state: (B, L_seq, 768)
                sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask']) # (B, 768)
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1) # L2 Unit Vectors: (B, 768)
                all_embeddings.append(sentence_embeddings.cpu().numpy()) # NumPy array: (B, 768)

        return np.vstack(all_embeddings) # Output Matrix Shape: (N, 768)

    def fit(self, proverbs_df: pd.DataFrame, training_situations_df: Optional[pd.DataFrame] = None):
        """
        Load or precompute dense embeddings for proverbs and situations.
        
        Cached Matrix Dimensions:
            - self.proverb_embeddings: (500, 768)
            - self.sit_embeddings:     (1510, 768)
        """
        self.proverbs_df = proverbs_df.copy().reset_index(drop=True)
        self.pid_to_idx = {pid: i for i, pid in enumerate(self.proverbs_df["proverb_id"])} # Size: 500

        proverb_cache = os.path.join(self.cache_dir, "banglabert_proverbs.npy")
        sit_cache = os.path.join(self.cache_dir, "banglabert_situations.npy")

        # 1. Proverb embeddings | Matrix Shape: (500, 768)
        if os.path.exists(proverb_cache):
            cached_p = np.load(proverb_cache)
            if len(cached_p) == len(self.proverbs_df):
                self.proverb_embeddings = cached_p # Shape: (500, 768)
            else:
                proverb_texts = [f"{row['proverb']}: {row['meaning']}" for _, row in self.proverbs_df.iterrows()]
                self.proverb_embeddings = self.encode(proverb_texts) # Shape: (500, 768)
                np.save(proverb_cache, self.proverb_embeddings)
        else:
            proverb_texts = [f"{row['proverb']}: {row['meaning']}" for _, row in self.proverbs_df.iterrows()]
            self.proverb_embeddings = self.encode(proverb_texts) # Shape: (500, 768)
            np.save(proverb_cache, self.proverb_embeddings)

        # 2. Situation embeddings | Matrix Shape: (1510, 768)
        if training_situations_df is not None and not training_situations_df.empty:
            self.sit_pids = training_situations_df["proverb_id"].tolist() # Length: 1510
            if os.path.exists(sit_cache):
                cached_s = np.load(sit_cache)
                if len(cached_s) == len(training_situations_df):
                    self.sit_embeddings = cached_s # Shape: (1510, 768)
                else:
                    self.sit_embeddings = self.encode(training_situations_df["situation"].tolist()) # Shape: (1510, 768)
                    np.save(sit_cache, self.sit_embeddings)
            else:
                self.sit_embeddings = self.encode(training_situations_df["situation"].tolist()) # Shape: (1510, 768)
                np.save(sit_cache, self.sit_embeddings)
                
        return self

    def score_all(self, query: str) -> np.ndarray:
        """
        Compute dual-layer cosine similarity of query against proverbs and situations.
        
        Linear Algebra Dimensions:
            - query_vec:                (1, 768)
            - self.proverb_embeddings:  (500, 768)
            - doc_sims:                 (500, 768) @ (768, 1) -> (500,)
            - self.sit_embeddings:      (1510, 768)
            - sit_sims:                 (1510, 768) @ (768, 1) -> (1510,)
            - sit_agg:                  (500,)  Max-pooled per proverb
            - return combined:          (500,)
        """
        if self.proverb_embeddings is None:
            raise ValueError("Model is not fitted. Call fit() first.")
            
        query_vec = self.encode([query])  # Shape: (1, 768)
        
        # Matrix-vector multiplication (500, 768) @ (768, 1) -> Shape: (500,)
        doc_sims = np.dot(self.proverb_embeddings, query_vec.T).flatten() # Shape: (500,)
        
        if self.sit_embeddings is not None and len(self.sit_pids) > 0:
            # Matrix-vector multiplication (1510, 768) @ (768, 1) -> Shape: (1510,)
            sit_sims = np.dot(self.sit_embeddings, query_vec.T).flatten() # Shape: (1510,)
            
            # Max pooling per proverb across its situations -> Shape: (500,)
            sit_agg = np.zeros(len(self.proverbs_df)) # Shape: (500,)
            for s_idx, spid in enumerate(self.sit_pids):
                p_idx = self.pid_to_idx.get(spid)
                if p_idx is not None and sit_sims[s_idx] > sit_agg[p_idx]:
                    sit_agg[p_idx] = sit_sims[s_idx]
                    
            # 65% direct proverb meaning match + 35% nearest situation neighbor match
            return 0.65 * doc_sims + 0.35 * sit_agg # Shape: (500,)
            
        return doc_sims # Shape: (500,)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top_k proverbs using BanglaBERT contextual similarity.
        
        Dimensions:
            - sims:        (500,)
            - top_indices: (top_k,)
        """
        sims = self.score_all(query) # Shape: (500,)
        top_indices = np.argsort(sims)[::-1][:top_k] # Shape: (top_k,)

        results = []
        for rank, idx in enumerate(top_indices, 1):
            score = float(sims[idx])
            row = self.proverbs_df.iloc[idx]
            results.append({
                "rank": rank,
                "proverb_id": row["proverb_id"],
                "proverb": row["proverb"],
                "meaning": row["meaning"],
                "theme": row.get("theme", ""),
                "example": row.get("example", ""),
                "score": round(score, 4),
                "model": "BanglaBERT"
            })
        return results
