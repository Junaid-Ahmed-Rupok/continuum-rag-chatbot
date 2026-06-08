"""
Public API for the Continuum RAG utility package.

Exports:
    ContinuumConfig  – Frozen configuration dataclass (env-var aware).
    get_config       – Returns the cached singleton ContinuumConfig instance.
                       Prefer this over instantiating ContinuumConfig directly.
    RAGMemory        – ChromaDB-backed memory store with Ebbinghaus decay.
    MemoryResult     – Dataclass returned by RAGMemory.retrieve().
    ConversationBuffer – Sliding-window conversation history with persistence.
    extract_facts    – Regex-based personal fact extractor.
"""

from .config import ContinuumConfig, get_config          # Configuration
from .memory import RAGMemory, MemoryResult               # Memory / retrieval
from .conversation import ConversationBuffer              # Conversation state
from .helpers import extract_facts                        # NLP utilities

__all__ = [
    "ContinuumConfig",
    "get_config",
    "RAGMemory",
    "MemoryResult",
    "ConversationBuffer",
    "extract_facts",
]
