from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import get_settings
from .pipeline import HazardPipeline
from .validation_store import ValidationStore

settings = get_settings()
pipeline = HazardPipeline(settings)
validation_store = ValidationStore(settings)

app = FastAPI(title="Food Hazard Detection Demo", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TopKItem(BaseModel):
    label: str
    prob: float


class InferRequest(BaseModel):
    title: str
    content: str
    top_k: Optional[int] = None


class InferResponse(BaseModel):
    pred_product: str
    pred_hazard: str
    topk_product: List[TopKItem]
    topk_hazard: List[TopKItem]


class ValidationRecord(BaseModel):
    title: str
    text: str
    pred_product: str
    pred_hazard: str
    topk_product: List[TopKItem] = Field(default_factory=list)
    topk_hazard: List[TopKItem] = Field(default_factory=list)


class ValidationResponse(BaseModel):
    total: int
    items: List[ValidationRecord]
    source: str


class HealthResponse(BaseModel):
    status: str
    device: str
    models_ready: bool
    validation_source: str
    labels: dict


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        device=str(pipeline.device),
        models_ready=pipeline.models_ready,
        validation_source=validation_store.source,
        labels={"product": len(pipeline.product_labels), "hazard": len(pipeline.hazard_labels)},
    )


@app.get("/validation", response_model=ValidationResponse)
async def get_validation(
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
):
    items_raw, total, source = validation_store.page(offset=offset, limit=limit)

    def to_topk(arr):
        return [TopKItem(label=lbl, prob=float(prob)) for lbl, prob in arr] if arr else []

    records = [
        ValidationRecord(
            title=item.get("title", ""),
            text=item.get("text", ""),
            pred_product=item.get("pred_product", ""),
            pred_hazard=item.get("pred_hazard", ""),
            topk_product=to_topk(item.get("topk_product", [])),
            topk_hazard=to_topk(item.get("topk_hazard", [])),
        )
        for item in items_raw
    ]
    return ValidationResponse(total=total, items=records, source=source)


@app.post("/infer", response_model=InferResponse)
async def infer(payload: InferRequest):
    try:
        pred = pipeline.predict(payload.title, payload.content, top_k=payload.top_k)
    except Exception as exc:  # pragma: no cover - surfaced to client
        raise HTTPException(status_code=500, detail=str(exc))

    def to_topk(arr):
        return [TopKItem(label=lbl, prob=float(prob)) for lbl, prob in arr]

    return InferResponse(
        pred_product=pred["pred_product"],
        pred_hazard=pred["pred_hazard"],
        topk_product=to_topk(pred["topk_product"]),
        topk_hazard=to_topk(pred["topk_hazard"]),
    )

