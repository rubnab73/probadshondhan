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
