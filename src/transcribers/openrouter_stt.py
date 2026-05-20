"""OpenRouter STT adapter — model 9 only (`google/chirp-3`).

Uses the OpenAI Python SDK pointed at OpenRouter's base URL. Calls
`audio.transcriptions.create`. Per DECISIONS.md §C, if OpenRouter rejects
this endpoint/model at runtime we let the adapter raise and the model is
marked FAILED. There is no fallback to chat-completions or any other model.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from src.transcribers.base import BaseTranscriber

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterSTTTranscriber(BaseTranscriber):
    def __init__(
        self,
        model_id: str,
        device: str = "cuda",  # ignored — cloud model, kept for interface compatibility
        api_key_env: str = "OPENROUTER_API_KEY",
    ) -> None:
        super().__init__(model_id=model_id, device=device)

        from openai import OpenAI

        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{api_key_env} is not set in the environment — required for OpenRouter STT"
            )
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
        logger.info("Initialized OpenRouter STT client for %s", model_id)

    def transcribe(self, audio_path: str) -> tuple[str, float]:
        path = Path(audio_path)
        start = time.perf_counter()
        with open(path, "rb") as f:
            result = self.client.audio.transcriptions.create(
                model=self.model_id,
                file=f,
            )
        latency = time.perf_counter() - start
        text = (getattr(result, "text", None) or "").strip()
        return text, latency

    def cleanup(self) -> None:
        # Cloud client — no GPU memory to release.
        return None
