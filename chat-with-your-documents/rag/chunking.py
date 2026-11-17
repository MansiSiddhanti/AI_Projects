# Chunk sizes are in embedding-model tokens, not characters.
# MiniLM only reads 256 tokens, anything past that never gets embedded.
import re
from collections.abc import Callable

TokenCounter = Callable[[str], int]

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(\"'])")


def whitespace_tokens(text: str) -> int:
    return len(text.split())


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]


def _split_oversized(sentence: str, max_tokens: int, count: TokenCounter) -> list[str]:
    # happens a lot with tables / reference lists where there are no periods
    pieces: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in sentence.split():
        word_len = max(count(word), 1)
        if current and current_len + word_len > max_tokens:
            pieces.append(" ".join(current))
            current, current_len = [], 0
        current.append(word)
        current_len += word_len
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_text(
    text: str,
    max_tokens: int,
    overlap_tokens: int = 0,
    count_tokens: TokenCounter = whitespace_tokens,
) -> list[str]:
    """Pack sentences into chunks of <= max_tokens. Each chunk repeats the last
    few sentences of the previous one (up to overlap_tokens)."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    overlap_tokens = min(overlap_tokens, max_tokens // 2)

    units: list[tuple[str, int]] = []
    for sentence in split_sentences(text):
        length = count_tokens(sentence)
        if length <= max_tokens:
            units.append((sentence, length))
        else:
            units.extend(
                (piece, count_tokens(piece))
                for piece in _split_oversized(sentence, max_tokens, count_tokens)
            )

    chunks: list[str] = []
    current: list[tuple[str, int]] = []
    current_len = 0

    for unit, length in units:
        if current and current_len + length > max_tokens:
            chunks.append(" ".join(u for u, _ in current))

            carry: list[tuple[str, int]] = []
            carry_len = 0
            for prev, prev_len in reversed(current):
                if carry_len + prev_len > overlap_tokens:
                    break
                carry.insert(0, (prev, prev_len))
                carry_len += prev_len

            # overlap + next sentence can still be too big, trim the overlap
            while carry and carry_len + length > max_tokens:
                carry_len -= carry.pop(0)[1]

            current, current_len = carry, carry_len

        current.append((unit, length))
        current_len += length

    if current:
        chunks.append(" ".join(u for u, _ in current))

    return chunks
