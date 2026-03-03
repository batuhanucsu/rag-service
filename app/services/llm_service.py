"""LLM service using Ollama for answer generation."""

import httpx


class LLMService:
    """Generate natural-language answers using a local Ollama model.

    Args:
        model: Ollama model name (e.g. ``llama3.1``).
        base_url: Ollama API base URL.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_answer(self, query: str, context_chunks: list[str]) -> str:
        """Generate an answer to *query* grounded on *context_chunks*.

        Args:
            query: The user's natural-language question.
            context_chunks: Retrieved document chunks to use as context.

        Returns:
            A natural-language answer string.
        """
        if not context_chunks:
            return "Üzgünüm, bu soruyla ilgili bilgi bulunamadı."

        context = "\n\n---\n\n".join(context_chunks)

        prompt = (
            "Sen bir soru-cevap asistanısın. Görevin, aşağıdaki bağlam bilgilerinden "
            "yararlanarak kullanıcının sorusunu cevaplamak.\n\n"
            "KURALLAR:\n"
            "1. Cevabını YALNIZCA bağlamdaki bilgilere dayandır.\n"
            "2. Bağlamda soruyla DOĞRUDAN İLGİLİ bilgi varsa, bu bilgiyi doğal "
            "ve akıcı Türkçe ile özetle.\n"
            "3. Bağlamda soruyla ilgili HİÇBİR bilgi yoksa, şunu söyle: "
            "'Bu soruyla ilgili yüklenen dokümanlarda bilgi bulunamadı.'\n"
            "4. Bağlamda OLMAYAN bilgileri uydurma veya kendi bilginden ekleme.\n"
            "5. Kısa ve net cevap ver.\n\n"
            f"### Bağlam:\n{context}\n\n"
            f"### Soru:\n{query}\n\n"
            "### Cevap:"
        )

        return self._call_ollama(prompt)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _call_ollama(self, prompt: str) -> str:
        """Send a prompt to the Ollama generate API and return the response."""
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 512,
                    },
                },
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
