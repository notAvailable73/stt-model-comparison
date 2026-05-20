"""Hugging Face Whisper adapter — covers models 1, 2, 3, 8 from plan.md §3.

Uses transformers' ASR pipeline so we get chunking for clips longer than 30s
for free. Defaults to language='bn' + task='transcribe'; if a particular
fine-tune rejects those generate kwargs we retry once without them and log it.
"""

from __future__ import annotations

import logging
import time

import torch
from transformers import pipeline

from src.transcribers.base import BaseTranscriber

logger = logging.getLogger(__name__)


class WhisperHFTranscriber(BaseTranscriber):
    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        language: str | None = "bn",
        task: str = "transcribe",
        chunk_length_s: int = 30,
    ) -> None:
        super().__init__(model_id=model_id, device=device)
        self.language = language
        self.task = task
        self.chunk_length_s = chunk_length_s

        use_cuda = device.startswith("cuda") and torch.cuda.is_available()
        torch_dtype = torch.float16 if use_cuda else torch.float32
        resolved_device = device if use_cuda else "cpu"
        if device.startswith("cuda") and not use_cuda:
            logger.warning("CUDA requested but not available — falling back to CPU for %s", model_id)

        logger.info("Loading Whisper-HF model: %s (device=%s, dtype=%s)", model_id, resolved_device, torch_dtype)
        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=model_id,
            torch_dtype=torch_dtype,
            device=resolved_device,
        )
        # Some fine-tunes hardcode language/task and reject these kwargs.
        # We discover that lazily on the first transcribe call.
        self._generate_kwargs: dict[str, str] | None = self._build_generate_kwargs()
        self._gen_kwargs_disabled = False

    def _build_generate_kwargs(self) -> dict[str, str] | None:
        if self.language is None:
            return None
        return {"language": self.language, "task": self.task}

    def transcribe(self, audio_path: str) -> tuple[str, float]:
        start = time.perf_counter()
        try:
            result = self.pipe(
                audio_path,
                chunk_length_s=self.chunk_length_s,
                generate_kwargs=self._generate_kwargs if not self._gen_kwargs_disabled else None,
            )
        except ValueError as e:
            # Likely "generation_config has no `language`" or similar — retry without kwargs.
            if self._gen_kwargs_disabled:
                raise
            logger.warning(
                "Model %s rejected generate_kwargs (%s); retrying without language/task",
                self.model_id, e,
            )
            self._gen_kwargs_disabled = True
            result = self.pipe(audio_path, chunk_length_s=self.chunk_length_s)

        latency = time.perf_counter() - start
        text = (result.get("text") or "").strip() if isinstance(result, dict) else str(result).strip()
        return text, latency

    def cleanup(self) -> None:
        logger.info("Releasing Whisper-HF model: %s", self.model_id)
        try:
            del self.pipe
        except AttributeError:
            logger.debug("No pipe attribute to delete for %s", self.model_id)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
