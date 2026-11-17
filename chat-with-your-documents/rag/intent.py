import re
from pathlib import Path

_SUMMARY_PATTERNS = [
    r"\bsummar(?:y|ies|ize|ise|ized|ised|izing|ising)\b",
    r"\b(?:key|main|major) (?:points?|findings?|takeaways?|contributions?)\b",
    r"\btl;?dr\b",
    r"\boverview of (?:the|this|these|my|that) (?:document|paper|pdf|article|survey)\b",
    r"\bbriefly explain (?:the|this|these|my|that) (?:document|paper|pdf|article|survey)\b",
    r"\bwhat (?:is|was) (?:the|this|that) (?:document|paper|pdf|article|survey) about\b",
    r"\bconclusion of (?:the|this|my|that) (?:document|paper|pdf|article|survey)\b",
]
_SUMMARY_RE = re.compile("|".join(_SUMMARY_PATTERNS), re.IGNORECASE)


def is_summary_request(question: str) -> bool:
    return bool(_SUMMARY_RE.search(question))


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def match_source(question: str, sources: list[str]) -> str | None:
    # "Paper_03.pdf", "paper 03" and "paper 3" should all match Paper_03.pdf
    text = f" {_normalize(question)} "
    for source in sources:
        stem = _normalize(Path(source).stem)
        variants = {stem, re.sub(r"\b0+(\d)", r"\1", stem)}
        if any(v and f" {v} " in text for v in variants):
            return source
    return None
