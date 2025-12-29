import json
import os
from typing import Any, Dict, List, Tuple

import pandas as pd

from .config import Settings


class ValidationStore:
    """
    Keeps validation inference results in memory.
    If no prediction file is present, falls back to ground-truth CSV so the UI can still render data.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.records: List[Dict[str, Any]] = []
        self.source: str = "unknown"
        self._load()

    def _load(self):
        if os.path.exists(self.settings.validation_pred_path):
            self.records = self._load_prediction_file(self.settings.validation_pred_path)
            self.source = "prediction_file"
        elif os.path.exists(self.settings.validation_pred_csv_path):
            self.records = self._load_prediction_csv_with_join(self.settings.validation_pred_csv_path)
            self.source = "prediction_csv"
        else:
            self.records = self._load_fallback_csv(self.settings.validation_csv_path)
            self.source = "fallback_ground_truth"

    def refresh(self):
        self._load()

    def _load_prediction_file(self, path: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                rows.append(
                    {
                        "title": data.get("title", ""),
                        "text": data.get("text", ""),
                        "pred_product": data.get("pred_product", ""),
                        "pred_hazard": data.get("pred_hazard", ""),
                        "topk_product": data.get("topk_product", []),
                        "topk_hazard": data.get("topk_hazard", []),
                    }
                )
        return rows

    def _load_fallback_csv(self, path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            return []
        df = pd.read_csv(path)
        rows = []
        for _, row in df.iterrows():
            rows.append(
                {
                    "title": str(row.get("title", "") or "").strip(),
                    "text": str(row.get("text", "") or "").strip(),
                    "pred_product": str(row.get("product-category", "") or ""),
                    "pred_hazard": str(row.get("hazard-category", "") or ""),
                    "topk_product": [],
                    "topk_hazard": [],
                }
            )
        return rows

    def _load_prediction_csv_with_join(self, pred_path: str) -> List[Dict[str, Any]]:
        """
        Join prediction CSV (with doc_id) to the corresponding incidents CSV to recover title/text.
        Supports top-k columns in the CSV:
          - prod_top{1..K}, prod_score{1..K}
          - haz_top{1..K},  haz_score{1..K}
        Scores can be percentages like '91.06%'.
        """
        if not os.path.exists(pred_path):
            return []
        pred_df = pd.read_csv(pred_path)
        valid_df = pd.read_csv(self.settings.validation_csv_path) if os.path.exists(self.settings.validation_csv_path) else None
        test_df = pd.read_csv(self.settings.test_csv_path) if os.path.exists(self.settings.test_csv_path) else None

        # Pick the best join dataframe by matching max doc_id / rowcount.
        join_df = valid_df
        if test_df is not None and ("doc_id" in pred_df.columns):
            try:
                max_doc_id = int(pred_df["doc_id"].max())
            except Exception:
                max_doc_id = -1
            if max_doc_id >= 0 and (valid_df is None or max_doc_id >= len(valid_df)) and max_doc_id < len(test_df):
                join_df = test_df

        rows = []
        for _, row in pred_df.iterrows():
            doc_id = int(row.get("doc_id", -1))
            title = ""
            text = ""
            if join_df is not None and 0 <= doc_id < len(join_df):
                vrow = join_df.iloc[doc_id]
                title = str(vrow.get("title", "") or "").strip()
                text = str(vrow.get("text", "") or "").strip()

            def parse_pct(v) -> float:
                if v is None:
                    return 0.0
                s = str(v).strip()
                if not s:
                    return 0.0
                s = s.replace("%", "").strip()
                try:
                    # CSV stores percent like 91.06 -> convert to 0.9106
                    return float(s) / 100.0
                except Exception:
                    return 0.0

            def parse_topk(prefix: str) -> List[Tuple[str, float]]:
                out: List[Tuple[str, float]] = []
                k = 1
                while True:
                    top_col = f"{prefix}_top{k}"
                    score_col = f"{prefix}_score{k}"
                    if top_col not in pred_df.columns or score_col not in pred_df.columns:
                        break
                    lbl = str(row.get(top_col, "") or "").strip()
                    prob = parse_pct(row.get(score_col, ""))
                    if lbl:
                        out.append((lbl, prob))
                    k += 1
                return out

            topk_product = parse_topk("prod")
            topk_hazard = parse_topk("haz")
            rows.append(
                {
                    "title": title,
                    "text": text,
                    "pred_product": str(row.get("pred_product", "") or ""),
                    "pred_hazard": str(row.get("pred_hazard", "") or ""),
                    "topk_product": topk_product,
                    "topk_hazard": topk_hazard,
                }
            )
        return rows

    def page(self, offset: int = 0, limit: int = 100) -> Tuple[List[Dict[str, Any]], int, str]:
        total = len(self.records)
        start = max(offset, 0)
        end = min(start + limit, total)
        return self.records[start:end], total, self.source

