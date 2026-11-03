"""
neurochat.ingestion.splitter
============================
Split plain text into overlapping fixed-size chunks.

Design rationale
----------------
Embedding models work best on short, focused passages.  We break long text
into overlapping windows so that sentences near a chunk boundary appear in
two successive chunks, preventing retrieval gaps.

Example
-------
>>> splitter = TextSplitter(chunk_size=20, overlap=5)
>>> chunks = splitter.split("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
>>> chunks[0]
'ABCDEFGHIJKLMNOPQRST'
>>> chunks[1]
'NOPQRSTUVWXYZ'
"""

from __future__ import annotations

from typing import List


class TextSplitter:
    """Split text into overlapping character-level windows.

    Parameters
    ----------
    chunk_size:
        Maximum number of characters per chunk (default 512).
    overlap:
        Number of characters that consecutive chunks share (default 64).
        Must be strictly less than *chunk_size*.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {overlap}")
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.overlap = overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(self, text: str) -> List[str]:
        """Return a list of (possibly overlapping) text chunks.

        Empty or whitespace-only input returns an empty list.
        A text shorter than *chunk_size* is returned as a single chunk.

        Parameters
        ----------
        text:
            The raw text to split.
        """
        if not text or not text.strip():
            return []

        step = self.chunk_size - self.overlap
        chunks: List[str] = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            # Strip trailing whitespace but preserve internal structure
            stripped = chunk.rstrip()
            if stripped:
                chunks.append(stripped)
            if end >= len(text):
                break
            start += step

        return chunks

    def split_by_paragraphs(self, text: str) -> List[str]:
        """Split text at blank lines first, then apply character windowing.

        This produces more semantically coherent chunks when the source has
        natural paragraph structure (e.g. plain-text articles, README files).
        """
        if not text or not text.strip():
            return []

        # Split on one or more blank lines
        import re
        paragraphs = re.split(r"\n\s*\n", text)
        chunks: List[str] = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(para) <= self.chunk_size:
                chunks.append(para)
            else:
                chunks.extend(self.split(para))

        return chunks

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"TextSplitter(chunk_size={self.chunk_size}, overlap={self.overlap})"
        )

