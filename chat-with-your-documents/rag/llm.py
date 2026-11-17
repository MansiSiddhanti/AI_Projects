from collections.abc import Iterator
from typing import Protocol

Message = dict[str, str]


class ChatModel(Protocol):
    name: str

    def chat(self, messages: list[Message]) -> str: ...
    def stream(self, messages: list[Message]) -> Iterator[str]: ...


class OllamaChat:
    def __init__(
        self,
        model: str,
        host: str | None = None,
        temperature: float = 0.1,
        context_window: int = 8192,
    ):
        import ollama

        self.name = model
        self.client = ollama.Client(host=host)
        self.options = {"temperature": temperature, "num_ctx": context_window}

    def chat(self, messages: list[Message]) -> str:
        response = self.client.chat(model=self.name, messages=messages, options=self.options)
        return response.message.content.strip()

    def stream(self, messages: list[Message]) -> Iterator[str]:
        for part in self.client.chat(
            model=self.name, messages=messages, options=self.options, stream=True
        ):
            if part.message.content:
                yield part.message.content

    def health(self) -> tuple[bool, str]:
        try:
            available = {m.model for m in self.client.list().models}
        except Exception as exc:  # connection refused, timeout, ...
            return False, f"Ollama is not reachable: {exc}"
        if not any(name.split(":")[0] == self.name.split(":")[0] for name in available):
            return False, f"Model '{self.name}' is not pulled. Run: ollama pull {self.name}"
        return True, f"Ollama ready ({self.name})"
