import json
import logging
import math
import hashlib
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .config import ContinuumConfig

logger = logging.getLogger(__name__)

_DECAY_BATCH_SIZE = 500  # Max memories to process per decay pass


@dataclass
class MemoryResult:
    """Single result returned from memory retrieval."""
    id: str
    text: str
    score: float
    strength: float
    age_days: float
    metadata: Dict[str, Any]


class RAGMemory:
    """
    Persistent memory system using ChromaDB + sentence-transformers.
    Supports Ebbinghaus decay, reinforcement, and pruning.
    """

    def __init__(self, config: ContinuumConfig) -> None:
        self.config = config

        self.data_path = Path(config.data_dir)
        self.chroma_path = self.data_path / "db"
        self.exports_path = self.data_path / "exports"
        self.config_path = self.data_path / "config"

        for d in (self.chroma_path, self.exports_path, self.config_path):
            d.mkdir(parents=True, exist_ok=True)

        logger.info("Loading embedding model: %s", config.embed_model)
        self.embedder = SentenceTransformer(config.embed_model, device="cpu")
        self.embedding_dim: int = self.embedder.get_sentence_embedding_dimension()

        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=config.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

        self.session_added: int = 0
        self.session_pruned: int = 0
        self.last_decay_time: float = 0.0
        self._load_stats()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Embed and store a new memory, returning its ID."""
        now = time.time()
        mem_meta: Dict[str, Any] = {
            "timestamp": now,
            "strength": 1.0,
            "last_reinforced": now,
            "source": (metadata or {}).get("source", "conversation"),
        }
        if metadata:
            mem_meta.update({k: v for k, v in metadata.items() if k != "source"})

        embedding = self.embedder.encode(text).tolist()
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        memory_id = f"mem_{int(now * 1000)}_{text_hash}"

        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            metadatas=[mem_meta],
            documents=[text],
        )
        self.session_added += 1
        self._save_stats()
        logger.debug("Added memory %s", memory_id)
        return memory_id

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_strength: Optional[float] = None,
    ) -> List[MemoryResult]:
        """Retrieve the most relevant memories for a query."""
        if self.collection.count() == 0:
            return []

        k = top_k or self.config.top_k
        threshold = min_strength if min_strength is not None else self.config.min_strength
        query_embedding = self.embedder.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k * 2, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        memories: List[MemoryResult] = []
        for doc_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            strength = float(meta.get("strength", 0.0))
            if strength < threshold:
                continue

            memories.append(MemoryResult(
                id=doc_id,
                text=doc,
                score=max(0.0, 1.0 - dist),
                strength=strength,
                age_days=(time.time() - float(meta.get("timestamp", time.time()))) / 86400,
                metadata=meta,
            ))
            if len(memories) >= k:
                break

        return memories

    def reinforce(self, memory_id: str, boost: Optional[float] = None) -> None:
        """
        Boost a memory's strength on re-access (capped at 1.0).
        Uses config.reinforce_boost if boost is not specified.
        """
        effective_boost = boost if boost is not None else self.config.reinforce_boost

        result = self.collection.get(ids=[memory_id], include=["metadatas"])
        if not result["ids"]:
            logger.debug("reinforce: memory %s not found", memory_id)
            return

        meta = result["metadatas"][0]
        meta["strength"] = round(min(1.0, float(meta.get("strength", 0.5)) + effective_boost), 6)
        meta["last_reinforced"] = time.time()
        self.collection.update(ids=[memory_id], metadatas=[meta])

    def decay_all(self) -> int:
        """
        Apply Ebbinghaus decay to all memories in batches.
        Prunes memories that fall below min_strength.
        Skips if called within the last hour.
        Returns number of pruned memories.
        """
        if time.time() - self.last_decay_time <= 3600:
            return 0
        if self.collection.count() == 0:
            return 0

        pruned = 0
        offset = 0

        while True:
            batch = self.collection.get(
                include=["metadatas"],
                limit=_DECAY_BATCH_SIZE,
                offset=offset,
            )
            if not batch["ids"]:
                break

            to_delete: List[str] = []
            to_update_ids: List[str] = []
            to_update_metas: List[Dict] = []

            for mem_id, meta in zip(batch["ids"], batch["metadatas"]):
                last_reinforced = float(
                    meta.get("last_reinforced", meta.get("timestamp", time.time()))
                )
                days = (time.time() - last_reinforced) / 86400
                new_strength = float(meta.get("strength", 1.0)) * math.exp(
                    -self.config.decay_lambda * days
                )

                if new_strength < self.config.min_strength:
                    to_delete.append(mem_id)
                else:
                    meta["strength"] = round(new_strength, 6)
                    to_update_ids.append(mem_id)
                    to_update_metas.append(meta)

            if to_delete:
                self.collection.delete(ids=to_delete)
                pruned += len(to_delete)

            if to_update_ids:
                self.collection.update(ids=to_update_ids, metadatas=to_update_metas)

            offset += len(batch["ids"])

        self.session_pruned += pruned
        self.last_decay_time = time.time()
        self._save_stats()
        logger.info("Decay pass complete: pruned=%d", pruned)
        return pruned

    # ------------------------------------------------------------------
    # Stats, inspection, export
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Return current memory system statistics."""
        total = self.collection.count()
        avg_strength = 0.0

        if total > 0:
            all_mem = self.collection.get(include=["metadatas"])
            strengths = [float(m.get("strength", 0.0)) for m in all_mem["metadatas"]]
            avg_strength = float(np.mean(strengths)) if strengths else 0.0

        size_kb = (
            sum(f.stat().st_size for f in self.chroma_path.rglob("*") if f.is_file()) / 1024
            if self.chroma_path.exists()
            else 0.0
        )

        return {
            "total": total,
            "avg_strength": round(avg_strength, 3),
            "session_added": self.session_added,
            "session_pruned": self.session_pruned,
            "size_kb": round(size_kb, 2),
        }

    def get_top_facts(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return the n strongest memories."""
        if self.collection.count() == 0:
            return []

        all_mem = self.collection.get(include=["documents", "metadatas"])
        pairs = sorted(
            zip(all_mem["documents"], all_mem["metadatas"]),
            key=lambda x: float(x[1].get("strength", 0.0)),
            reverse=True,
        )
        return [
            {"text": doc, "strength": round(float(meta.get("strength", 0.0)), 4)}
            for doc, meta in pairs[:n]
        ]

    def export_json(self) -> Path:
        """Serialize all memories to a timestamped JSON export file."""
        filename = self.exports_path / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        all_mem = self.collection.get(include=["documents", "metadatas"])

        payload = {
            "export_timestamp": time.time(),
            "total_memories": len(all_mem["ids"]),
            "config": {
                "embed_model": self.config.embed_model,
                "collection": self.config.chroma_collection,
                "decay_lambda": self.config.decay_lambda,
                "min_strength": self.config.min_strength,
            },
            "memories": [
                {"id": mid, "text": doc, "metadata": meta}
                for mid, doc, meta in zip(
                    all_mem["ids"], all_mem["documents"], all_mem["metadatas"]
                )
            ],
        }
        filename.write_text(json.dumps(payload, indent=2))
        logger.info("Exported %d memories to %s", len(all_mem["ids"]), filename)
        return filename

    def reset(self) -> None:
        """Delete and recreate the ChromaDB collection."""
        try:
            self.chroma_client.delete_collection(self.config.chroma_collection)
        except Exception:
            logger.warning("reset: collection delete failed, continuing")

        self.collection = self.chroma_client.create_collection(
            name=self.config.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        self.session_added = 0
        self.session_pruned = 0
        self.last_decay_time = time.time()
        self._save_stats()
        logger.info("Memory collection reset.")

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load_stats(self) -> None:
        stats_path = self.config_path / "stats.json"
        if not stats_path.exists():
            return
        try:
            data = json.loads(stats_path.read_text())
            self.session_added = int(data.get("session_added", 0))
            self.session_pruned = int(data.get("session_pruned", 0))
            self.last_decay_time = float(data.get("last_decay_time", 0.0))
        except (json.JSONDecodeError, OSError, ValueError) as e:
            logger.warning("Failed to load stats: %s", e)

    def _save_stats(self) -> None:
        stats_path = self.config_path / "stats.json"
        try:
            stats_path.write_text(json.dumps({
                "session_added": self.session_added,
                "session_pruned": self.session_pruned,
                "last_decay_time": self.last_decay_time,
                "last_updated": time.time(),
            }, indent=2))
        except OSError as e:
            logger.warning("Failed to save stats: %s", e)
