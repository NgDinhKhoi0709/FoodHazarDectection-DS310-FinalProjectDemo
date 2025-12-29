import difflib
import json
import os
import re
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from bs4 import BeautifulSoup
from huggingface_hub import hf_hub_download
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .config import DEFAULT_MODEL_SPECS, ModelSpec, Settings

# -----------------------------
# Text preprocessing
# -----------------------------


def extract_text_from_html(html_content: str) -> str:
    if html_content is None:
        return ""
    soup = BeautifulSoup(str(html_content), "html.parser")
    return soup.get_text(separator=" ").strip()


def basic_clean(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


NOISE_PATTERNS: List[str] = []


def remove_recall_boilerplate(text: str) -> str:
    t = text
    for p in NOISE_PATTERNS:
        t = re.sub(p, " ", t, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", t).strip()


def normalize_sentence_for_dedupe(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\d{3,}", " ", s)
    s = re.sub(r'[^a-z0-9\\s.,;:!?"-]+', " ", s)
    return re.sub(r"\s+", " ", s).strip()


def advanced_deduplicate_sentences(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned, seen, buf = [], set(), []

    for s in sentences:
        raw = s.strip()
        if not raw:
            continue
        norm = normalize_sentence_for_dedupe(raw)
        if len(norm) < 10:
            continue
        if norm in seen:
            continue

        dup = False
        for prev in buf[-50:]:
            if difflib.SequenceMatcher(None, norm, prev).ratio() >= 0.95:
                dup = True
                break
        if dup:
            continue

        cleaned.append(raw)
        seen.add(norm)
        buf.append(norm)

    return " ".join(cleaned).strip()


ENTITY_REPLACEMENTS = {
    r"\be\.?\s*coli\b": "escherichia coli",
    r"\bc\.?\s*botulinum\b": "clostridium botulinum",
    r"\blisteria\s+spp\b": "listeria monocytogenes",
    r"\bsoy\s+proteins?\b": "soybeans",
}


def normalize_entities(text: str) -> str:
    t = text
    for pat, rep in ENTITY_REPLACEMENTS.items():
        t = re.sub(pat, rep, t, flags=re.IGNORECASE)
    return t


def clean_foodhazard_text(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    t = basic_clean(text)
    t = remove_recall_boilerplate(t)
    t = advanced_deduplicate_sentences(t)
    t = normalize_entities(t)
    return re.sub(r"\s+", " ", t).strip()


# -----------------------------
# Tokenization utilities
# -----------------------------


def chunk_by_tokens(text: str, tokenizer, max_tokens: int, overlap: int, min_chars: int) -> List[str]:
    text = str(text).strip()
    if not text:
        return []
    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]

    chunks: List[str] = []
    start = 0
    while start < len(enc):
        end = start + max_tokens
        sub_ids = enc[start:end]
        sub_text = tokenizer.decode(sub_ids, skip_special_tokens=True)
        sub_text = re.sub(r"\s+", " ", sub_text).strip()
        if len(sub_text) >= min_chars:
            chunks.append(sub_text)
        start += max_tokens - overlap
    return chunks


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


# -----------------------------
# Model definitions
# -----------------------------


class MultiTaskClassifier(nn.Module):
    def __init__(self, base_model_name: str, n_product: int, n_hazard: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(base_model_name)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.product_head = nn.Linear(hidden_size, n_product)
        self.hazard_head = nn.Linear(hidden_size, n_hazard)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]
        cls = self.dropout(cls)
        product_logits = self.product_head(cls)
        hazard_logits = self.hazard_head(cls)
        return torch.cat([product_logits, hazard_logits], dim=-1)


def load_multitask_from_hub(
    spec: ModelSpec, n_product: int, n_hazard: int, device: torch.device, hf_token: Optional[str]
):
    tokenizer = AutoTokenizer.from_pretrained(spec.repo_id, trust_remote_code=True, token=hf_token)

    weights_path = None
    for fname in ["model.safetensors", "pytorch_model.bin"]:
        try:
            weights_path = hf_hub_download(repo_id=spec.repo_id, filename=fname, token=hf_token)
            break
        except Exception:
            continue
    if weights_path is None:
        raise FileNotFoundError(f"Missing weights in repo: {spec.repo_id}")

    model = MultiTaskClassifier(base_model_name=spec.base_model, n_product=n_product, n_hazard=n_hazard)

    if weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file as safe_load_file

        state_dict = safe_load_file(weights_path)
    else:
        state_dict = torch.load(weights_path, map_location="cpu")

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return tokenizer, model


# -----------------------------
# Inference pipeline
# -----------------------------


class HazardPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_specs = DEFAULT_MODEL_SPECS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.product_id2label, self.hazard_id2label = self._load_label_space(settings.label_map_path)

        self._lock = threading.Lock()
        self._models_ready = False
        self.tokenizer_a = None
        self.tokenizer_b = None
        self.model_a = None
        self.model_b = None

    @property
    def product_labels(self) -> List[str]:
        return self.product_id2label

    @property
    def hazard_labels(self) -> List[str]:
        return self.hazard_id2label

    @property
    def models_ready(self) -> bool:
        return self._models_ready

    def _load_label_space(self, path: str) -> Tuple[List[str], List[str]]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Label mapping not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        prod_map: Dict[str, int] = data.get("product_label_to_id", {})
        haz_map: Dict[str, int] = data.get("hazard_label_to_id", {})

        def invert(m: Dict[str, int]) -> List[str]:
            return [k for k, _ in sorted(m.items(), key=lambda kv: kv[1])]

        return invert(prod_map), invert(haz_map)

    def _ensure_models_loaded(self):
        if self._models_ready:
            return
        with self._lock:
            if self._models_ready:
                return
            spec_a, spec_b = self.model_specs[0], self.model_specs[1]
            self.tokenizer_a, self.model_a = load_multitask_from_hub(
                spec_a, len(self.product_id2label), len(self.hazard_id2label), self.device, self.settings.hf_token
            )
            self.tokenizer_b, self.model_b = load_multitask_from_hub(
                spec_b, len(self.product_id2label), len(self.hazard_id2label), self.device, self.settings.hf_token
            )
            self._models_ready = True

    def predict(self, title: str, content: str, top_k: Optional[int] = None):
        self._ensure_models_loaded()

        w_a = float(self.settings.ensemble_weight)
        w_b = 1.0 - w_a
        topk = top_k or self.settings.top_k_default
        topk = max(1, min(topk, len(self.product_id2label), len(self.hazard_id2label)))

        title_clean = clean_foodhazard_text(extract_text_from_html(title))
        text_clean = clean_foodhazard_text(extract_text_from_html(content))
        merged = f"{title_clean} {text_clean}".lower().strip()
        if not merged:
            merged = "[EMPTY]"

        chunks_a = chunk_by_tokens(
            merged, self.tokenizer_a, self.settings.max_tokens, self.settings.chunk_overlap, self.settings.min_chars
        )
        chunks_b = chunk_by_tokens(
            merged, self.tokenizer_b, self.settings.max_tokens, self.settings.chunk_overlap, self.settings.min_chars
        )
        if not chunks_a:
            chunks_a = ["[EMPTY]"]
        if not chunks_b:
            chunks_b = ["[EMPTY]"]

        prod_a, haz_a = self._infer_chunks(self.tokenizer_a, self.model_a, chunks_a)
        prod_b, haz_b = self._infer_chunks(self.tokenizer_b, self.model_b, chunks_b)

        prod = w_a * prod_a + w_b * prod_b
        haz = w_a * haz_a + w_b * haz_b

        pred_prod = int(np.argmax(prod))
        pred_haz = int(np.argmax(haz))

        topk_prod_idx = np.argsort(-prod)[:topk]
        topk_haz_idx = np.argsort(-haz)[:topk]

        return {
            "pred_product": self.product_id2label[pred_prod],
            "pred_hazard": self.hazard_id2label[pred_haz],
            "topk_product": [(self.product_id2label[i], float(prod[i])) for i in topk_prod_idx],
            "topk_hazard": [(self.hazard_id2label[i], float(haz[i])) for i in topk_haz_idx],
        }

    def _infer_chunks(self, tokenizer, model, chunks: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        enc = tokenizer(
            chunks,
            truncation=True,
            padding="max_length",
            max_length=self.settings.max_tokens,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).detach().cpu().numpy()
        prod_logits = logits[:, : len(self.product_id2label)]
        haz_logits = logits[:, len(self.product_id2label) :]

        prod_probs = softmax_np(prod_logits).mean(axis=0)
        haz_probs = softmax_np(haz_logits).mean(axis=0)

        prod_probs = prod_probs / max(prod_probs.sum(), 1e-12)
        haz_probs = haz_probs / max(haz_probs.sum(), 1e-12)
        return prod_probs, haz_probs

