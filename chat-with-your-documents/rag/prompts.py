from .llm import Message
from .models import ScoredChunk

QA_SYSTEM = """You are a precise research assistant answering questions about a \
collection of documents.

Rules:
1. Answer ONLY from the numbered sources supplied with the question.
2. Cite every claim with the label of the source that supports it, e.g. [1] or [2][3].
3. Never invent sources, numbers, or facts. Never cite a label that was not supplied.
4. If the sources do not contain the answer, say so plainly instead of guessing.
5. Be concise and well structured; use bullet points for lists."""

CONDENSE_PROMPT = """Rewrite the follow-up question as a single standalone question \
that can be understood without the conversation. Resolve pronouns such as "it", \
"they" or "that paper". If it is already standalone, return it unchanged.
Return only the question, nothing else.

Conversation:
{history}

Follow-up question: {question}
Standalone question:"""

SUMMARY_MAP_PROMPT = """Summarize the following excerpt from the document "{source}" \
(part {part} of {total}).

Capture the objective, methods, key findings, numbers and conclusions that appear \
in this excerpt. Use only the supplied text and do not invent anything. Do not \
claim the paper lacks something just because this excerpt does not mention it. \
Keep it under 150 words.

EXCERPT:
{text}

SUMMARY:"""

SUMMARY_REDUCE_PROMPT = """Below are partial summaries of consecutive parts of the \
SAME document, "{source}". Merge them into one coherent summary with these \
markdown sections:

**Objective** - **Methods / Approach** - **Key Findings** - **Limitations** - **Conclusion**

Use only the partial summaries, merge duplicates, keep technical terms and numbers \
accurate, and omit a section if nothing supports it.

PARTIAL SUMMARIES:
{text}

COMBINED SUMMARY:"""

QUESTION_GEN_PROMPT = """Here is a passage from a research paper.

PASSAGE:
{text}

Write ONE specific factual question that this passage answers. The question \
must make sense to someone who has not seen the passage, so do not say "the \
passage", "this paper" or "the authors". Return only the question."""


def format_sources(passages: list[ScoredChunk]) -> str:
    return "\n\n".join(
        f"[{i}] (from {p.chunk.source}, page {p.chunk.page})\n{p.chunk.text}"
        for i, p in enumerate(passages, start=1)
    )


def build_qa_messages(
    question: str,
    passages: list[ScoredChunk],
    history: list[Message] | None = None,
) -> list[Message]:
    messages: list[Message] = [{"role": "system", "content": QA_SYSTEM}]
    messages.extend(history or [])
    messages.append(
        {
            "role": "user",
            "content": f"SOURCES:\n{format_sources(passages)}\n\nQUESTION: {question}",
        }
    )
    return messages


def format_history(history: list[Message]) -> str:
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in history)
