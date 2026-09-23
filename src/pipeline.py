"""
Inference & Data Pipeline for Bangla Situation-to-Proverb System.
Coordinates data loading and retrieval across Unified, BanglaBERT, and TF-IDF models.
"""

import os
from typing import List, Dict, Any, Optional
import pandas as pd

from .models.tfidf_retriever import TFIDFRetriever
from .models.unified_engine import UnifiedProverbEngine


class ProverbPipeline:
    """Unified query and inference manager for Situation-to-Proverb models."""

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, "dataset")
        self.data_dir = data_dir

        self.proverbs_df: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.dev_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

        # 1. Flagship Ensemble (BanglaBERT + TF-IDF)
        self.unified = UnifiedProverbEngine()
        # 2. Standalone TF-IDF Lexical Model
        self.tfidf = self.unified.tfidf

        self.is_ready = False
        self.load_data()
        self.train_models()

    def load_data(self):
        """Load all dataset CSV files."""
        proverbs_path = os.path.join(self.data_dir, "proverbs.csv")
        train_path = os.path.join(self.data_dir, "situations_train.csv")
        dev_path = os.path.join(self.data_dir, "situations_dev.csv")
        test_path = os.path.join(self.data_dir, "situations_human_test.csv")

        if os.path.exists(proverbs_path):
            self.proverbs_df = pd.read_csv(proverbs_path, encoding="utf-8-sig")
        if os.path.exists(train_path):
            self.train_df = pd.read_csv(train_path, encoding="utf-8-sig")
        if os.path.exists(dev_path):
            self.dev_df = pd.read_csv(dev_path, encoding="utf-8-sig")
        if os.path.exists(test_path):
            self.test_df = pd.read_csv(test_path, encoding="utf-8-sig")

    def train_models(self):
        """Fit models with proverbs and training situation context."""
        if self.proverbs_df is None:
            raise RuntimeError("Proverbs data not loaded.")

        self.unified.fit(self.proverbs_df, self.train_df)
        self.is_ready = True

    def retrieve(self, situation: str, model_name: str = "unified", top_k: int = 5) -> Dict[str, Any]:
        """Retrieve top proverbs for a situation using the specified model."""
        if not self.is_ready:
            self.train_models()

        m = model_name.lower().strip()
        if m in ("banglabert", "bert", "pretrained", "semantic"):
            if self.unified.banglabert is not None:
                results = self.unified.banglabert.retrieve(situation, top_k=top_k)
                active_model = "BanglaBERT Contextual Neural Model"
            else:
                results = self.unified.retrieve(situation, top_k=top_k)
                active_model = "Unified Proverb Neural Engine"
        elif m in ("tfidf", "tf-idf", "tf_idf", "lexical", "bm25"):
            results = self.tfidf.retrieve(situation, top_k=top_k)
            active_model = "TF-IDF Lexical Engine"
        else:
            results = self.unified.retrieve(situation, top_k=top_k)
            active_model = "Unified Proverb Neural Engine"

        return {
            "query": situation,
            "model": active_model,
            "top_k": top_k,
            "results": results
        }

    def get_proverbs(self) -> List[Dict[str, Any]]:
        """Return all proverbs as a list of dicts."""
        if self.proverbs_df is None:
            return []
        return self.proverbs_df.to_dict(orient="records")
