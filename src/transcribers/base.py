"""Abstract base class for all transcribers. See plan.md §3.1."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTranscriber(ABC):
    """Common interface every model adapter must implement.

    The benchmark loop loads one model at a time, calls `transcribe` once per
    clip, then calls `cleanup` before loading the next model so GPU memory
    doesn't pile up.
    """

    def __init__(self, model_id: str, device: str = "cuda") -> None:
        self.model_id = model_id
        self.device = device

    @abstractmethod
    def transcribe(self, audio_path: str) -> tuple[str, float]:
        """Transcribe one audio file.

        Returns:
            (predicted_text, latency_seconds)
        """
        ...

    def cleanup(self) -> None:
        """Release model weights and GPU memory. Default is a no-op; override
        in subclasses that hold heavy resources."""
        return None
