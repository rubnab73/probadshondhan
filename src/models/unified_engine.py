"""
Unified Proverb Search Engine (Flagship Hybrid Model)
Combines:
1. TF-IDF Lexical Retriever: Fast keyword precision
2. BanglaBERT Neural Retriever: Deep contextual semantics (csebuetnlp/banglabert)
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from .tfidf_retriever import TFIDFRetriever
from .banglabert_retriever import BanglaBERTRetriever, TRANSFORMERS_AVAILABLE


class UnifiedProverbEngine:
    """
    The Single, Combined Optimal Model for Bangla Situation-to-Proverb Retrieval.
    Blends BanglaBERT contextual attention (75%) with TF-IDF lexical matching (25%).
    """

    def __init__(self, use_transformer: bool = True):
        self.use_transformer = use_transformer and TRANSFORMERS_AVAILABLE
        self.tfidf = TFIDFRetriever()
        self.banglabert = BanglaBERTRetriever() if self.use_transformer else None
        self.proverbs_df: Optional[pd.DataFrame] = None

    def fit(self, proverbs_df: pd.DataFrame, training_situations_df: Optional[pd.DataFrame] = None):
        """Fit both TF-IDF and BanglaBERT engines on proverbs and training situations."""
        self.proverbs_df = proverbs_df.copy().reset_index(drop=True)
        self.tfidf.fit(self.proverbs_df, training_situations_df)
        if self.banglabert is not None:
            self.banglabert.fit(self.proverbs_df, training_situations_df)
        return self

    def retrieve(self, situation: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Unified inference call combining TF-IDF lexical and BanglaBERT neural intelligence."""
        if self.proverbs_df is None:
            raise RuntimeError("Engine not fitted. Call fit() first.")

        def normalize_scores(arr: np.ndarray) -> np.ndarray:
            ptp = np.ptp(arr)
            if ptp < 1e-9:
                return np.zeros_like(arr)
            return (arr - np.min(arr)) / ptp

        tfidf_raw = self.tfidf.score_all(situation)
        norm_tfidf = normalize_scores(tfidf_raw)

        if self.banglabert is not None:
            bert_raw = self.banglabert.score_all(situation)
            norm_bert = normalize_scores(bert_raw)
            unified_scores = 0.75 * norm_bert + 0.25 * norm_tfidf
        else:
            norm_bert = np.zeros_like(norm_tfidf)
            unified_scores = norm_tfidf

        top_indices = np.argsort(unified_scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            row = self.proverbs_df.iloc[idx]
            has_keyword_match = norm_tfidf[idx] > 0.4
            has_deep_semantic = (self.banglabert is not None and norm_bert[idx] > 0.4)

            if has_keyword_match and has_deep_semantic:
                match_type = "শব্দার্থিক ও প্রাসঙ্গিক উভয়ের মেলবন্ধন (Strong Semantic & Lexical Match)"
            elif has_deep_semantic:
                match_type = "গভীর ভাবার্থ ও রূপক অনুধাবন (Deep Contextual Match)"
            else:
                match_type = "আংশিক প্রাসঙ্গিকতা (Partial Relevance)"

            results.append({
                "rank": rank,
                "proverb_id": row["proverb_id"],
                "proverb": row["proverb"],
                "meaning": row["meaning"],
                "theme": row.get("theme", ""),
                "example": row.get("example", ""),
                "score": round(float(unified_scores[idx]), 4),
                "lexical_score": round(float(norm_tfidf[idx]), 3),
                "semantic_score": round(float(norm_bert[idx]), 3) if self.banglabert is not None else 0.0,
                "match_type": match_type,
                "model": "Unified Proverb Neural Engine"
            })

        return results
