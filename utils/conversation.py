import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Literal

logger = logging.getLogger(__name__)

Role = Literal["user", "assistant", "system"]

class ConversationBuffer:
    """
    Sliding window conversation history with JSON persistence.
    Stores up to max_turns * 2 messages (one user + one assistant per turn).
    """

    def __init__(self, data_path: Path, max_turns: int = 8) -> None:
        self.max_turns = max_turns
        self._buffer: deque[Dict] = deque(maxlen=max_turns * 2)
        self._path = data_path / "config" / "conversation_buffer.json"
        self._load()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def add(self, role: Role, content: str) -> None:
        """Append a validated message to the buffer."""
        if not content.strip():
            raise ValueError("Message content must not be empty.")
        self._buffer.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
        })

    def format_for_llm(self) -> List[Dict[str, str]]:
        """Return buffer as plain {role, content} dicts for the LLM API."""
        return [{"role": m["role"], "content": m["content"]} for m in self._buffer]

    def messages(self) -> List[Dict]:
        """Return all messages including metadata (e.g. timestamps)."""
        return list(self._buffer)

    def save(self) -> None:
        """Persist current buffer to disk."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(list(self._buffer), indent=2))
        except OSError as e:
            logger.warning("ConversationBuffer: save failed — %s", e)

    def clear(self) -> None:
        """Clear buffer in memory and on disk."""
        self._buffer.clear()
        self.save()

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return (
            f"ConversationBuffer("
            f"turns={len(self._buffer) // 2}, "
            f"max_turns={self.max_turns}, "
            f"path={self._path})"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load buffer from disk, capped to current max_turns setting."""
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text())
            if not isinstance(raw, list):
                raise ValueError("Expected a JSON array.")
            for msg in raw[-(self.max_turns * 2):]:
                if {"role", "content", "timestamp"} <= msg.keys():
                    self._buffer.append(msg)
                else:
                    logger.debug("Skipping malformed message: %s", msg)
        except (json.JSONDecodeError, ValueError, OSError) as e:
            logger.warning("ConversationBuffer: failed to load from %s — %s", self._path, e)
