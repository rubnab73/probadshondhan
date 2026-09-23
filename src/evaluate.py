import os
import math
from typing import Dict, List, Any
import numpy as np
import pandas as pd

from .pipeline import ProverbPipeline


def compute_ndcg_at_k(rank: int, k: int = 5) -> float:
    """Computes binary nDCG@k for a single relevant item at rank (1-indexed)."""
    if rank <= 0 or rank > k:
        return 0.0
    dcg = 1.0 / math.log2(rank + 1)
    idcg = 1.0 / math.log2(1 + 1)
    return dcg / idcg


class ProverbEvaluator:
    """Benchmark evaluation suite across all active retrieval models."""

    def __init__(self, pipeline: ProverbPipeline):
        self.pipeline = pipeline

    def evaluate_model_on_split(self, model_name: str, test_df: pd.DataFrame, top_k: int = 5) -> Dict[str, float]:
        """Evaluate a specific model on a given test dataset."""
        top1_hits = 0
        top3_hits = 0
        top5_hits = 0
        rr_sum = 0.0
        ndcg_sum = 0.0
        total = len(test_df)

        if total == 0:
            return {"top_1": 0.0, "top_3": 0.0, "top_5": 0.0, "mrr": 0.0, "ndcg_5": 0.0, "total_queries": 0}

        for _, row in test_df.iterrows():
            situation = row["situation"]
            target_id = row["proverb_id"]

            retrieval = self.pipeline.retrieve(situation, model_name=model_name, top_k=top_k)
            results = retrieval["results"]

            rank_found = 0
            for r in results:
                if r["proverb_id"] == target_id:
                    rank_found = r["rank"]
                    break

            if rank_found == 1:
                top1_hits += 1
            if 1 <= rank_found <= 3:
                top3_hits += 1
            if 1 <= rank_found <= 5:
                top5_hits += 1

            if rank_found > 0:
                rr_sum += 1.0 / rank_found
                ndcg_sum += compute_ndcg_at_k(rank_found, k=5)

        return {
            "top_1": round(top1_hits / total, 4),
            "top_3": round(top3_hits / total, 4),
            "top_5": round(top5_hits / total, 4),
            "mrr": round(rr_sum / total, 4),
            "ndcg_5": round(ndcg_sum / total, 4),
            "total_queries": total
        }

    def run_full_benchmark(self) -> Dict[str, Any]:
        """
        Runs full benchmark comparing:
        1. TF-IDF (Modern Scikit-Learn Lexical Retriever)
        2. BanglaBERT (Pretrained Contextual Transformer)
        3. Unified Proverb Engine (BanglaBERT + TF-IDF Hybrid Ensemble)
        """
        models = [
            "TF-IDF",
            "BanglaBERT",
            "Unified Proverb Engine"
        ]
        results = {
            "human_test": {},
            "dev_split": {}
        }

        # 1. Human Test Split
        if self.pipeline.test_df is not None:
            print("\nEvaluating on Human Gold Test split (25 samples)...")
            for m in models:
                print(f"  --> Benchmarking {m}...")
                results["human_test"][m] = self.evaluate_model_on_split(m, self.pipeline.test_df)

        # 2. Dev Split
        if self.pipeline.dev_df is not None:
            print("\nEvaluating on Dev Validation split (20 samples)...")
            for m in models:
                print(f"  --> Benchmarking {m}...")
                results["dev_split"][m] = self.evaluate_model_on_split(m, self.pipeline.dev_df)

        return results


def print_markdown_table(benchmark_results: Dict[str, Any], split_name: str = "human_test"):
    """Helper to print nicely formatted benchmark table without encoding errors."""
    data = benchmark_results.get(split_name, {})
    title = "Human Gold Test" if split_name == "human_test" else "Dev Validation"
    lines = [
        f"\n### Benchmark Results: {title}",
        "| Model / Algorithm | Paradigm | Top-1 | Top-3 | Top-5 | MRR | nDCG@5 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    label_map = {
        "TF-IDF": ("TF-IDF Lexical Engine", "Lexical (scikit-learn)"),
        "BanglaBERT": ("BanglaBERT Contextual", "Deep Neural Transformer"),
        "Unified Proverb Engine": ("Unified Proverb Engine", "Hybrid Ensemble (Flagship)")
    }

    for method, m in data.items():
        disp_name, origin = label_map.get(method, (method, "Custom"))
        lines.append(f"| **{disp_name}** | {origin} | {m['top_1']*100:.1f}% | {m['top_3']*100:.1f}% | {m['top_5']*100:.1f}% | {m['mrr']:.4f} | {m['ndcg_5']:.4f} |")

    print("\n".join(lines))


if __name__ == "__main__":
    pipeline = ProverbPipeline()
    evaluator = ProverbEvaluator(pipeline)
    res = evaluator.run_full_benchmark()
    print_markdown_table(res, "human_test")
    print_markdown_table(res, "dev_split")
