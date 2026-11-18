"""NOVA AI FastAPI backend.

Localhost-only by default. Long-running training runs in a background thread
so chat requests are never blocked.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from ..core.config import CHECKPOINTS_DIR, DATABASES_DIR, MODELS_DIR, ensure_directories
from ..core.logging import get_logger
from ..database.db import DatabaseManager
from ..database.models import (
    ConversationStore,
    MessageStore,
    DocumentStore,
    TrainingHistoryStore,
    init_schema,
)
from ..knowledge.embeddings import TfidfEmbedding
from ..knowledge.vector_store import create_vector_store, VectorItem
from ..knowledge.retriever import Retriever
from ..knowledge.rag import RAGPipeline
from ..orchestrator import Orchestrator
from ..runtime.model_manager import ModelManager
from .schemas import (  # noqa: E501
    ChatRequest,
    ChatResponse,
    ConversationItem,
    HealthResponse,
    KnowledgeDocument,
    KnowledgeIngestRequest,
    KnowledgeSearchRequest,
    KnowledgeSearchResult,
    MessageItem,
    ModelInfo,
    ModelListResponse,
    ModelLoadRequest,
    ModelUnloadRequest,
    TrainingHistoryItem,
    TrainingStartRequest,
    TrainingStatusResponse,
)

logger = get_logger("backend.api")

app = FastAPI(title="NOVA AI", version="0.1.0", docs_url="/docs", redoc_url="/redoc")


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
class AppState:
    def __init__(self) -> None:
        ensure_directories()
        self.databases_dir = DATABASES_DIR
        self.models_dir = MODELS_DIR

        conv_db = DATABASES_DIR / "conversations.db"
        init_schema(conv_db)
        self.conv_conn = DatabaseManager(conv_db).connect()
        self.conversation_store = ConversationStore(self.conv_conn)
        self.message_store = MessageStore(self.conv_conn)

        self.model_manager = ModelManager(databases_dir=DATABASES_DIR, models_dir=MODELS_DIR)

        # RAG (lazy embeddings; TF-IDF fallback works offline without downloads).
        kb_db = DATABASES_DIR / "knowledge.db"
        init_schema(kb_db)
        self.kb_conn = DatabaseManager(kb_db).connect()
        self.document_store = DocumentStore(self.kb_conn)
        self._vector_store = None
        self._retriever = None
        self._rag = None

        self.training_thread = None
        self.training_status = {
            "running": False,
            "step": 0,
            "epoch": 0,
            "model": "",
            "loss": None,
        }
        # In-memory chunk catalog for the knowledge base (rebuilt on each ingest).
        self._chunks: list = []  # list of {id, text, metadata}
        self._embedding_model: Optional[TfidfEmbedding] = None

        th_db = DATABASES_DIR / "training.db"
        init_schema(th_db)
        self.training_history_store = TrainingHistoryStore(DatabaseManager(th_db).connect())

    @property
    def embedding_model(self) -> TfidfEmbedding:
        if self._embedding_model is None:
            self._embedding_model = TfidfEmbedding(dimension=384)
        return self._embedding_model

    @property
    def vector_store(self):
        if self._vector_store is None:
            self._vector_store = create_vector_store(dimension=384, prefer_faiss=False)
        return self._vector_store

    @property
    def rag(self) -> Optional[RAGPipeline]:
        if self._rag is None and self._retriever is None:
            self._retriever = Retriever(self.embedding_model, self.vector_store)
            self._rag = RAGPipeline(self._retriever)
        return self._rag

    def rebuild_vector_store(self) -> None:
        """Re-fit the embedding model on all chunks and rebuild the store."""
        self._vector_store = create_vector_store(dimension=384, prefer_faiss=False)
        if not self._chunks:
            self._retriever = Retriever(self.embedding_model, self._vector_store)
            self._rag = RAGPipeline(self._retriever)
            return
        texts = [c["text"] for c in self._chunks]
        self.embedding_model.fit(texts)
        items = [
            VectorItem(
                id=c["id"],
                vector=self.embedding_model.embed_query(c["text"]),
                metadata=c["metadata"],
            )
            for c in self._chunks
        ]
        self._vector_store.add(items)
        self._retriever = Retriever(self.embedding_model, self._vector_store)
        self._rag = RAGPipeline(self._retriever)
        logger.info("Rebuilt vector store with %d chunks", len(self._chunks))


state = AppState()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
@app.get("/models", response_model=ModelListResponse)
def list_models() -> ModelListResponse:
    entries = state.model_manager.list_models()
    models = [ModelInfo(**e) for e in entries]
    # Mark the currently loaded model.
    active = state.model_manager.active_model_name()
    for m in models:
        if m.name == active:
            m.loaded = True
    return ModelListResponse(models=models)


@app.post("/models/load", response_model=ModelInfo)
def load_model(req: ModelLoadRequest) -> ModelInfo:
    try:
        backend = state.model_manager.load_model(req.name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    info = backend.get_model_info()
    return ModelInfo(**asdict(info))


@app.post("/models/unload", response_model=dict)
def unload_model(req: ModelUnloadRequest) -> dict:
    state.model_manager.unload_model()
    return {"status": "unloaded"}



# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    orchestrator = Orchestrator(
        state.model_manager,
        state.conversation_store,
        state.message_store,
        rag_pipeline=state.rag,
    )
    try:
        response = orchestrator.chat(
            user_input=req.message,
            model_name=req.model,
            conversation_id=req.conversation_id,
            rag_mode=req.rag_mode,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return ChatResponse(
        text=response.text,
        model=response.model,
        sources=response.sources,
        conversation_id=response.conversation_id,
        rag_used=response.rag_used,
    )


@app.post("/chat/stream", response_model=ChatResponse)
def chat_stream(req: ChatRequest) -> ChatResponse:
    """Streaming endpoint. The PyTorch backend yields the full completion;
    the HTTP response returns it as a single message."""
    return chat(req)


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------
@app.get("/conversations", response_model=list[ConversationItem])
def list_conversations():
    rows = state.conversation_store.list()
    return [ConversationItem(**dict(r)) for r in rows]


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    conv = state.conversation_store.get(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    messages = state.message_store.list_for_conversation(conversation_id)
    return {
        "conversation": dict(conv),
        "messages": [
            {
                "id": m["id"],
                "role": m["role"],
                "content": m["content"],
                "sources": json.loads(m["sources"] or "[]"),
                "created_at": m["created_at"],
            }
            for m in messages
        ],
    }



# ---------------------------------------------------------------------------
# Knowledge
# ---------------------------------------------------------------------------
@app.post("/knowledge/ingest")
def ingest_document(req: KnowledgeIngestRequest):
    from ..collection.document_loader import load_document
    from ..knowledge.chunker import chunk_text
    from ..knowledge.duplicate_detector import DuplicateDetector

    path = Path(req.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.path}")

    doc = load_document(path)
    if not doc.text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from the document")

    doc_id = doc.metadata.get("name", path.name)
    detector = DuplicateDetector()
    if detector.is_duplicate(doc.text, source=doc_id):
        return {"status": "duplicate", "doc_id": doc_id}

    chunks = chunk_text(doc.text, chunk_size=500, overlap=100)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document produced no chunks")

    for i, ch in enumerate(chunks):
        chunk_id = f"{doc_id}::chunk{i}"
        state._chunks.append(
            {
                "id": chunk_id,
                "text": ch.text,
                "metadata": {
                    "text": ch.text,
                    "source": doc.metadata.get("name", path.name),
                    "source_type": doc.metadata.get("format", path.suffix.lstrip(".")),
                    "doc_id": doc_id,
                    "chunk_index": ch.index,
                    "path": str(path),
                },
            }
        )

    state.document_store.add(
        {
            "id": doc_id,
            "title": doc.metadata.get("name", path.name),
            "source_type": doc.metadata.get("format", path.suffix.lstrip(".")),
            "source_path": str(path),
            "chunk_count": len(chunks),
            "indexed": True,
        }
    )

    state.rebuild_vector_store()
    return {
        "status": "ingested",
        "doc_id": doc_id,
        "chunks": len(chunks),
        "source_type": doc.metadata.get("format", path.suffix.lstrip(".")),
    }


@app.get("/knowledge/documents", response_model=List[KnowledgeDocument])
def list_documents():
    rows = state.document_store.list()
    return [
        KnowledgeDocument(
            id=r["id"],
            title=r["title"],
            source_type=r["source_type"],
            source_path=r["source_path"],
            source_url=r["source_url"],
            chunk_count=int(r["chunk_count"] or 0),
            indexed=bool(r["indexed"]),
            created_at=r["created_at"],
        )
        for r in rows
    ]


@app.delete("/knowledge/documents/{doc_id}")
def delete_document(doc_id: str):
    state.document_store.delete(doc_id)
    state._chunks = [c for c in state._chunks if c["metadata"].get("doc_id") != doc_id]
    state.rebuild_vector_store()
    return {"status": "deleted", "doc_id": doc_id}


@app.post("/knowledge/search", response_model=List[KnowledgeSearchResult])
def search_knowledge(req: KnowledgeSearchRequest):
    rag = state.rag
    if rag is None:
        return []
    chunks = rag.retriever.retrieve(req.query, top_k=req.top_k)
    return [
        KnowledgeSearchResult(
            id=c.doc_id or c.source,
            score=c.score,
            text=c.text,
            source=c.source,
        )
        for c in chunks
    ]




# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _run_training(preset, epochs, batch_size, learning_rate, dataset_path):
    from ..core.config import get_preset
    from ..core.tokenizer import build_tokenizer
    from ..core.model import NOVATinyModel
    from ..training.dataset import TextDataset, make_dataloader
    from ..training.preprocessing import clean_text, tokenize_corpus
    from ..training.trainer import TrainConfig, Trainer

    state.training_status = {
        "running": True,
        "step": 0,
        "epoch": 0,
        "model": preset,
        "loss": None,
    }
    try:
        config = get_preset(preset)
        texts = []
        if dataset_path and Path(dataset_path).exists():
            p = Path(dataset_path)
            if p.is_dir():
                for f in p.iterdir():
                    if f.suffix.lower() in (".txt", ".md"):
                        texts.append(clean_text(f.read_text(encoding="utf-8", errors="replace")))
            else:
                texts.append(clean_text(p.read_text(encoding="utf-8", errors="replace")))
        if not texts:
            texts = ["hello world", "nova ai", "local first chatbot"]

        token_ids, tokenizer = tokenize_corpus(texts, tokenizer_type="char")
        tok_path = CHECKPOINTS_DIR / f"tokenizer_{preset}.json"
        tokenizer.save(tok_path)

        ds = TextDataset(token_ids, block_size=config.context_length, stride=config.context_length // 2, shuffle=True)
        train_dl = make_dataloader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_ds = TextDataset(token_ids, block_size=config.context_length, stride=config.context_length // 2, shuffle=False)
        val_ds._block_starts = val_ds._block_starts[: max(1, len(val_ds) // 10)]
        val_dl = make_dataloader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        model = NOVATinyModel(config)
        train_cfg = TrainConfig(
            model_name=preset,
            model_preset=preset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            context_length=config.context_length,
            save_every_steps=max(1, len(train_dl) // 4),
        )

        def on_step(step, loss):
            state.training_status["step"] = step
            state.training_status["loss"] = loss

        def on_epoch(em):
            state.training_status["epoch"] = em.epoch
            state.training_status["loss"] = em.train_loss

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_dl,
            val_dataloader=val_dl,
            config=train_cfg,
            model_config=config,
            tokenizer_config=tokenizer.config,
            on_step=on_step,
            on_epoch=on_epoch,
        )
        trainer.train()
        ckpt = trainer.save_checkpoint()
        state.training_history_store.add(
            {
                "model_name": preset,
                "preset": preset,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "steps": trainer.step,
                "final_train_loss": trainer.history.steps[-1].train_loss if trainer.history.steps else None,
                "final_val_loss": trainer.history.epochs[-1].val_loss if trainer.history.epochs else None,
                "best_val_loss": trainer.history.best_val_loss,
                "best_epoch": trainer.history.best_epoch,
                "checkpoint_path": str(ckpt),
            }
        )
    except Exception as exc:
        logger.exception("Training failed: %s", exc)
    finally:
        state.training_status["running"] = False
        state.training_thread = None


@app.post("/training/start")
def training_start(req: TrainingStartRequest):
    if state.training_status["running"]:
        raise HTTPException(status_code=409, detail="Training already running")
    import threading

    state.training_thread = threading.Thread(
        target=_run_training,
        args=(req.preset, req.epochs, req.batch_size, req.learning_rate, req.dataset_path),
        daemon=True,
    )
    state.training_thread.start()
    return {"status": "started", "preset": req.preset}


@app.post("/training/stop")
def training_stop():
    if not state.training_status["running"]:
        return {"status": "not running"}
    return {"status": "stop requested"}


@app.get("/training/status", response_model=TrainingStatusResponse)
def training_status():
    return TrainingStatusResponse(**state.training_status)


@app.get("/training/history", response_model=List[TrainingHistoryItem])
def training_history():
    rows = state.training_history_store.list()
    return [TrainingHistoryItem(**dict(r)) for r in rows]
