"""
Utility script to precompute validation predictions and save to JSONL.

Usage:
    python -m backend.precompute_validation --input data/incidents_valid.csv --output data/validation_predictions.jsonl
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config import Settings, get_settings
from .pipeline import HazardPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute validation predictions")
    parser.add_argument("--input", dest="input_path", default=None, help="CSV file with validation incidents")
    parser.add_argument("--output", dest="output_path", default=None, help="Where to write JSONL predictions")
    parser.add_argument("--top-k", dest="top_k", type=int, default=None, help="Top-k labels to store")
    parser.add_argument("--max-rows", dest="max_rows", type=int, default=None, help="Limit rows (for quick dry-run)")
    return parser.parse_args()


def main():
    args = parse_args()
    settings = get_settings()

    input_path = args.input_path or settings.validation_csv_path
    output_path = args.output_path or settings.validation_pred_path

    df = pd.read_csv(input_path)
    if args.max_rows:
        df = df.head(args.max_rows)

    pipeline = HazardPipeline(settings)
    top_k = args.top_k or settings.top_k_default

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
            title = row.get("title", "")
            text = row.get("text", "")
            pred = pipeline.predict(str(title), str(text), top_k=top_k)
            record = {
                "title": title,
                "text": text,
                "pred_product": pred["pred_product"],
                "pred_hazard": pred["pred_hazard"],
                "topk_product": pred["topk_product"],
                "topk_hazard": pred["topk_hazard"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(df)} predictions to {output_path}")


if __name__ == "__main__":
    main()

