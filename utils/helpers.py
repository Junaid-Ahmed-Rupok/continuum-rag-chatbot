import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class FactPattern:
    pattern: str
    label_template: str        # Use {match} as placeholder
    min_len: int = 3
    max_len: int = 80


FACT_PATTERNS: List[FactPattern] = [
    FactPattern(r"my name is ([a-z]+)",                        "User's name is {match}"),
    FactPattern(r"i(?:'m| am) called ([a-z]+)",                "User's name is {match}"),
    FactPattern(r"i(?:'m| am) from ([a-z][\w\s,]+?)(?:[.,]|$)","User is from {match}"),
    FactPattern(r"i live (?:in|at) ([a-z][\w\s,]+?)(?:[.,]|$)","User lives in {match}"),
    FactPattern(r"i (?:work|worked) (?:at|for) ([\w\s]+?)(?:[.,]|$)", "User works at {match}"),
    FactPattern(r"i (?:like|love|enjoy|prefer) ([\w\s]+?)(?:[.,]|$)", "User enjoys {match}"),
    FactPattern(r"i(?:'m| am) (?:a |an )?([\w][\w\s]{2,20}?) (?:who|by|at|in)\b", "User is a {match}"),
    FactPattern(r"i have (?:a |an )?([\w\s]+?)(?:[.,]|$)",     "User has {match}"),
    FactPattern(r"i(?:'ve| have) been ([\w\s]+?)(?:[.,]|$)",   "User has been {match}"),
    FactPattern(r"my ([\w\s]+?) is ([\w\s,]+?)(?:[.,]|$)",     "User's {match}"),  # tuple: joined below
]

# Words that make a "has been" or "have" fact too generic to store
_NOISE_FACTS = frozenset({
    "thinking", "wondering", "trying", "meaning", "wanting",
    "hoping", "looking", "working on it", "busy",
})


def _clean(text: str) -> str:
    """Normalize whitespace and strip trailing filler."""
    text = re.sub(r"\s+", " ", text).strip().rstrip(".,;")
    return text


def _is_valid(fact_body: str, min_len: int, max_len: int) -> bool:
    if not (min_len <= len(fact_body) <= max_len):
        return False
    if fact_body in _NOISE_FACTS:
        return False
    # Reject if it's mostly stop words / very short words
    words = fact_body.split()
    if len(words) > 6:
        return False
    return True


def _format_label(template: str, match: tuple | str) -> Optional[str]:
    """Apply label template, handling both single and tuple matches."""
    if isinstance(match, tuple):
        # e.g. "my (X) is (Y)" → "User's X is Y"
        parts = [_clean(p) for p in match]
        body = " is ".join(parts)  # join tuple parts naturally
        return template.format(match=body)
    else:
        body = _clean(match)
        return template.format(match=body.capitalize() if len(body.split()) == 1 else body)


def extract_facts(user_msg: str, bot_msg: str = "", max_facts: int = 3) -> List[str]:
    """
    Extract personal facts from a user message.

    Args:
        user_msg: The user's message (primary source).
        bot_msg: Optionally included for context, not extracted from directly.
        max_facts: Maximum facts to return per exchange.

    Returns:
        List of unique, human-readable fact strings.
    """
    # Only extract from user message; bot_msg reserved for future context use
    text = user_msg.lower().strip()

    seen_bodies: set[str] = set()
    facts: List[str] = []

    for fp in FACT_PATTERNS:
        for match in re.findall(fp.pattern, text):
            label = _format_label(fp.label_template, match)
            if label is None:
                continue

            # Extract the body (after the prefix) for dedup + validation
            body = _clean(match if isinstance(match, str) else " ".join(match))

            if body in seen_bodies:
                continue
            if not _is_valid(body, fp.min_len, fp.max_len):
                continue

            seen_bodies.add(body)
            facts.append(label)

            if len(facts) >= max_facts:
                return facts

    return facts
