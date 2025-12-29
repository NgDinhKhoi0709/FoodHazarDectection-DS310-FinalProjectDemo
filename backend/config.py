import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Get project root (parent of backend folder)
_BACKEND_DIR = Path(__file__).parent
_PROJECT_ROOT = _BACKEND_DIR.parent


def _resolve_path(relative_path: str) -> str:
    """Resolve path relative to project root if not absolute."""
    if os.path.isabs(relative_path):
        return relative_path
    return str(_PROJECT_ROOT / relative_path)


@dataclass
class ModelSpec:
    name: str
    repo_id: str
    base_model: str


DEFAULT_MODEL_SPECS: List[ModelSpec] = [
    ModelSpec(
        name="DeBERTa",
        repo_id="tringooo/DeBERTa-FoodHazardDetection",
        base_model="microsoft/deberta-v3-large",
    ),
    ModelSpec(
        name="RoBERTa",
        repo_id="tringooo/RoBERTa-FoodHazardDetection",
        base_model="FacebookAI/roberta-large",
    ),
]


@dataclass
class Settings:
    label_map_path: str = _resolve_path(os.getenv("LABEL_MAP_PATH", "data/label_mappings.json"))
    validation_pred_path: str = _resolve_path(os.getenv("VALIDATION_PRED_PATH", "data/validation_predictions.jsonl"))
    validation_pred_csv_path: str = _resolve_path(os.getenv("VALIDATION_PRED_CSV_PATH", "data/test_predictions.csv"))
    validation_csv_path: str = _resolve_path(os.getenv("VALIDATION_CSV_PATH", "data/incidents_valid.csv"))
    test_csv_path: str = _resolve_path(os.getenv("TEST_CSV_PATH", "data/incidents_test.csv"))

    ensemble_weights_path: str = _resolve_path(os.getenv("ENSEMBLE_WEIGHTS_PATH", "data/ensemble_weights.json"))
    ensemble_weight: float = float(os.getenv("ENSEMBLE_WEIGHT", "0.55"))
    top_k_default: int = int(os.getenv("TOP_K_DEFAULT", "3"))

    max_tokens: int = int(os.getenv("MAX_TOKENS", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "64"))
    min_chars: int = int(os.getenv("MIN_CHARS", "30"))

    hf_token: Optional[str] = os.getenv("HF_TOKEN")
    auto_compute_validation: bool = os.getenv("AUTO_COMPUTE_VALIDATION", "false").lower() == "true"


def get_settings() -> Settings:
    settings = Settings()
    # If ENSEMBLE_WEIGHT not overridden, try to load from ensemble_weights.json
    if os.getenv("ENSEMBLE_WEIGHT") is None and os.path.exists(settings.ensemble_weights_path):
        try:
            import json

            with open(settings.ensemble_weights_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "tuned_w_model_a" in data:
                settings.ensemble_weight = float(data["tuned_w_model_a"])
        except Exception:
            pass
    return settings

