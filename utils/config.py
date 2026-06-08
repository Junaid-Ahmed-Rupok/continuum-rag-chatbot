import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, default))

def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, default))


@dataclass(frozen=True)
class ContinuumConfig:
    """Immutable configuration for Continuum RAG system."""
    max_tokens: int          = field(default_factory=lambda: _env_int('MAX_TOKENS', 512))
    temperature: float       = field(default_factory=lambda: _env_float('TEMPERATURE', 0.7))
    embed_model: str         = field(default_factory=lambda: os.getenv('EMBED_MODEL', 'all-MiniLM-L6-v2'))
    chroma_collection: str   = field(default_factory=lambda: os.getenv('CHROMA_COLLECTION', 'continuum_memory'))
    top_k: int               = field(default_factory=lambda: _env_int('TOP_K', 5))
    min_strength: float      = field(default_factory=lambda: _env_float('MIN_STRENGTH', 0.05))
    decay_lambda: float      = field(default_factory=lambda: _env_float('DECAY_LAMBDA', 0.1))
    ctx_window_turns: int    = field(default_factory=lambda: _env_int('CTX_WINDOW_TURNS', 8))
    data_dir: Path           = field(default_factory=lambda: Path(os.getenv('CONTINUUM_DATA_DIR', './continuum_data')))

    def __post_init__(self):
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(f"temperature must be in [0, 2], got {self.temperature}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.decay_lambda <= 0:
            raise ValueError(f"decay_lambda must be > 0, got {self.decay_lambda}")


@lru_cache(maxsize=1)
def get_config() -> ContinuumConfig:
    """Get singleton configuration instance."""
    return ContinuumConfig()
