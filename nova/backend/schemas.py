"""Pydantic request/response schemas for the NOVA AI HTTP API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"


class ModelInfo(BaseModel):
    name: str
    model_type: str
    architecture: str
    file_path: Optional[str] = None
    quantization: Optional[str] = None
    context_length: Optional[int] = None
    model_format: Optional[str] = None
    description: str = ""
    loaded: bool = False


class ModelListResponse(BaseModel):
    models: List[ModelInfo]


class ModelLoadRequest(BaseModel):
    name: str = Field(..., description="Registered model name to load")


class ModelUnloadRequest(BaseModel):
    name: Optional[str] = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    model: Optional[str] = None
    conversation_id: Optional[str] = None
    rag_mode: str = Field(default="auto", pattern="^(auto|enabled|disabled)$")
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: int = Field(default=50, ge=0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_new_tokens: int = Field(default=64, ge=1, le=512)


class ChatResponse(BaseModel):
    text: str
    model: str
    sources: List[dict] = []
    conversation_id: str
    rag_used: bool = False


class ConversationItem(BaseModel):
    id: str
    title: str
    model_name: str
    rag_enabled: bool
    created_at: str
    updated_at: str


class MessageItem(BaseModel):
    id: str
    role: str
    content: str
    sources: List[str] = []
    created_at: str


class KnowledgeIngestRequest(BaseModel):
    path: str


class KnowledgeDocument(BaseModel):
    id: str
    title: str
    source_type: str
    source_path: Optional[str] = None
    source_url: Optional[str] = None
    chunk_count: int
    indexed: bool
    created_at: str


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class KnowledgeSearchResult(BaseModel):
    id: str
    score: float
    text: str
    source: str


class TrainingStartRequest(BaseModel):
    preset: str = "tiny"
    epochs: int = Field(default=1, ge=1)
    batch_size: int = Field(default=2, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    dataset_path: Optional[str] = None


class TrainingStatusResponse(BaseModel):
    running: bool
    step: int
    epoch: int
    model: str
    loss: Optional[float] = None


class TrainingHistoryItem(BaseModel):
    model_name: str
    preset: str
    epochs: int
    batch_size: int
    learning_rate: float
    steps: int
    final_train_loss: Optional[float]
    final_val_loss: Optional[float]
    best_val_loss: Optional[float]
    created_at: str
