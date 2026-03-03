"""Production-ready text chunker with paragraph-aware splitting."""

import re


class TextChunker:
    """Split long texts into semantically meaningful chunks.

    First splits documents into paragraphs / sections (on blank lines,
    ``\\n\\n``, bullet boundaries, or heading-like patterns), then merges
    small consecutive paragraphs up to *chunk_size* so that each chunk
    preserves its semantic context.

    Args:
        chunk_size: Maximum chunk length in characters.
        overlap_sentences: Number of sentences from the end of the
            previous chunk to prepend to the next chunk for continuity.
    """

    # Patterns that mark section / paragraph boundaries (ordered by priority).
    _SECTION_SPLIT_RE = re.compile(
        r"""
        \n\s*\n            # blank line(s) — strongest signal
        | \n(?=[-•●]\s)    # line starting with a bullet
        | \n(?=\d+[.)]\s)  # line starting with a numbered list item
        | \n(?=[A-ZÇĞIİÖŞÜ].{0,60}:\s*\n)  # heading-like line ending with ':'
        """,
        re.VERBOSE,
    )

    def __init__(
        self,
        chunk_size: int = 800,
        overlap_sentences: int = 1,
    ) -> None:
        self._chunk_size = chunk_size
        self._overlap_sentences = overlap_sentences

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, text: str) -> list[str]:
        """Split *text* into semantically coherent chunks.

        Args:
            text: The input text to split.

        Returns:
            A list of text chunks, each at most *chunk_size* characters.
        """
        text = text.strip()
        if not text:
            return []
        if len(text) <= self._chunk_size:
            return [text]

        paragraphs = self._split_into_paragraphs(text)
        return self._merge_paragraphs(paragraphs)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """Split text on paragraph / section boundaries."""
        parts = self._SECTION_SPLIT_RE.split(text)
        return [p.strip() for p in parts if p and p.strip()]

    def _last_n_sentences(self, text: str, n: int) -> str:
        """Return the last *n* sentences of *text* (best-effort)."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        overlap = sentences[-n:] if len(sentences) > n else []
        return " ".join(overlap)

    def _merge_paragraphs(self, paragraphs: list[str]) -> list[str]:
        """Merge consecutive paragraphs until *chunk_size* is reached."""
        chunks: list[str] = []
        current = ""
        prev_overlap = ""

        for para in paragraphs:
            candidate = f"{current}\n\n{para}".strip() if current else para

            if len(candidate) <= self._chunk_size:
                current = candidate
            else:
                # Current buffer is full — flush it.
                if current:
                    chunks.append(current)
                    prev_overlap = self._last_n_sentences(
                        current, self._overlap_sentences,
                    )

                # If a single paragraph exceeds chunk_size, hard-split it.
                if len(para) > self._chunk_size:
                    sub_chunks = self._hard_split(para)
                    for sc in sub_chunks:
                        piece = f"{prev_overlap}\n\n{sc}".strip() if prev_overlap else sc
                        if len(piece) > self._chunk_size:
                            piece = sc  # drop overlap if it causes overflow
                        chunks.append(piece)
                        prev_overlap = self._last_n_sentences(
                            sc, self._overlap_sentences,
                        )
                    current = ""
                else:
                    current = f"{prev_overlap}\n\n{para}".strip() if prev_overlap else para
                    if len(current) > self._chunk_size:
                        current = para  # drop overlap if it causes overflow

        if current:
            chunks.append(current)

        return chunks

    def _hard_split(self, text: str) -> list[str]:
        """Character-level split for paragraphs exceeding *chunk_size*."""
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + self._chunk_size
            if end >= len(text):
                chunks.append(text[start:].strip())
                break
            # Try to break at the last sentence boundary.
            for i in range(end, start + self._chunk_size // 2, -1):
                if text[i - 1] in ".!?\n" and (i == len(text) or text[i].isspace()):
                    end = i
                    break
            else:
                # Fall back to last space.
                last_space = text.rfind(" ", start + self._chunk_size // 2, end)
                if last_space != -1:
                    end = last_space + 1
            chunks.append(text[start:end].strip())
            start = end
        return chunks
