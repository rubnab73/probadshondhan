"""
TF-IDF Lexical Retriever for Bangla Situation-to-Proverb Search.
Uses scikit-learn TfidfVectorizer with sublinear TF scaling and dual-layer
proverb + training situation matching for high lexical precision.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from ..preprocessor import BanglaPreprocessor


class TFIDFRetriever:
    """Modern, compact TF-IDF Lexical Retriever using scikit-learn."""

    def __init__(self):
        self.preprocessor = BanglaPreprocessor(remove_stopwords=False)
        self.vectorizer = TfidfVectorizer(
            tokenizer=self.preprocessor.tokenize,
            token_pattern=None,
            sublinear_tf=True,
            smooth_idf=True,
            norm="l2"
        )
        self.proverbs_df: Optional[pd.DataFrame] = None
        self.proverb_matrix = None  # Sparse matrix (N_proverbs, |V|)
        self.sit_matrix = None      # Sparse matrix (N_situations, |V|)
        self.sit_pids: List[str] = []
        self.pid_to_idx: Dict[str, int] = {}

    def fit(self, proverbs_df: pd.DataFrame, training_situations_df: Optional[pd.DataFrame] = None):
        """Fit vectorizer on proverbs and training situations."""
        self.proverbs_df = proverbs_df.copy().reset_index(drop=True)
        self.pid_to_idx = {pid: i for i, pid in enumerate(self.proverbs_df["proverb_id"])}

        # Proverb text documents: "proverb: meaning"
        proverb_docs = [
            f"{row['proverb']} {row['meaning']}"
            for _, row in self.proverbs_df.iterrows()
        ]

        # Training situation documents
        situation_docs = []
        if training_situations_df is not None and not training_situations_df.empty:
            situation_docs = training_situations_df["situation"].tolist()
            self.sit_pids = training_situations_df["proverb_id"].tolist()

        # Fit on combined corpus
        all_docs = proverb_docs + situation_docs
        self.vectorizer.fit(all_docs)

        # Precompute L2-normalized TF-IDF sparse matrices
        self.proverb_matrix = self.vectorizer.transform(proverb_docs)
        if situation_docs:
            self.sit_matrix = self.vectorizer.transform(situation_docs)

        return self

    def score_all(self, situation: str) -> np.ndarray:
        """Compute dual-layer cosine similarity scores across all proverbs."""
        if self.proverb_matrix is None:
            raise RuntimeError("TF-IDF Retriever is not fitted. Call fit() first.")

        query_vec = self.vectorizer.transform([situation])  # (1, |V|)
        proverb_scores = self.proverb_matrix.dot(query_vec.T).toarray().flatten()

        if self.sit_matrix is not None and self.sit_pids:
            sit_scores = self.sit_matrix.dot(query_vec.T).toarray().flatten()

            # Max-pool situation similarities per proverb
            sit_agg = np.zeros(len(self.proverbs_df), dtype=np.float32)
            for s_idx, spid in enumerate(self.sit_pids):
                p_idx = self.pid_to_idx.get(spid)
                if p_idx is not None and sit_scores[s_idx] > sit_agg[p_idx]:
                    sit_agg[p_idx] = sit_scores[s_idx]

            # 50% direct proverb meaning match + 50% nearest situation match
            return 0.5 * proverb_scores + 0.5 * sit_agg

        return proverb_scores

    def retrieve(self, situation: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top_k proverbs for a given situation."""
        scores = self.score_all(situation)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            row = self.proverbs_df.iloc[idx]
            results.append({
                "rank": rank,
                "proverb_id": row["proverb_id"],
                "proverb": row["proverb"],
                "meaning": row["meaning"],
                "theme": row.get("theme", ""),
                "example": row.get("example", ""),
                "score": round(float(scores[idx]), 4),
                "model": "TF-IDF Lexical Engine"
            })
        return results
