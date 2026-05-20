"""OmniASR adapter — model 7 only (`facebook/omniASR-CTC-300M`).

Wraps Meta's `omnilingual-asr` package. The plan locks the model id and the
language flag (`lang="ben_Beng"`); we do NOT substitute either. If the
package or the repo doesn't resolve, the adapter raises and the model is
marked FAILED per §3.2 / DECISIONS.md §D.

The exact entry-point name in `omnilingual-asr` has shifted across releases.
We try the two most-documented paths in order and surface whichever import
error is most useful to the user via the logger.
"""

from __future__ import annotations

import logging
import time

from src.transcribers.base import BaseTranscriber

logger = logging.getLogger(__name__)


def _load_omni_class():
    """Import OmniLingualASR via the two known module paths.

    Returns the class. Raises ImportError with both paths logged if neither
    works — caller turns that into FAILED state for the model.
    """
    errors: list[str] = []
    try:
        from omnilingual_asr.models.inference import OmniLingualASR  # type: ignore[import-not-found]
        return OmniLingualASR
    except ImportError as e:
        errors.append(f"omnilingual_asr.models.inference: {e}")
        logger.debug("OmniASR import path 1 failed: %s", e)
    try:
        from omnilingual_asr import OmniLingualASR  # type: ignore[import-not-found,no-redef]
        return OmniLingualASR
    except ImportError as e:
        errors.append(f"omnilingual_asr: {e}")
        logger.debug("OmniASR import path 2 failed: %s", e)
    raise ImportError("OmniLingualASR not found in `omnilingual-asr` package; tried: " + " | ".join(errors))


class OmniASRTranscriber(BaseTranscriber):
    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        lang: str = "ben_Beng",
    ) -> None:
        super().__init__(model_id=model_id, device=device)
        self.lang = lang

        OmniLingualASR = _load_omni_class()
        logger.info("Loading OmniASR model: %s (lang=%s)", model_id, lang)
        self.model = OmniLingualASR.from_pretrained(model_id)
        if hasattr(self.model, "to"):
            self.model = self.model.to(device)
        if hasattr(self.model, "eval"):
            self.model.eval()

    def transcribe(self, audio_path: str) -> tuple[str, float]:
        start = time.perf_counter()
        result = self.model.transcribe(audio_paths=[audio_path], lang=self.lang)
        latency = time.perf_counter() - start

        # Normalize: package returns either a list of strings or a list of dicts/objects.
        first = result[0] if isinstance(result, list) and result else result
        if isinstance(first, dict):
            text = str(first.get("text", "")).strip()
        elif hasattr(first, "text"):
            text = str(first.text).strip()
        else:
            text = str(first).strip()
        return text, latency

    def cleanup(self) -> None:
        logger.info("Releasing OmniASR model: %s", self.model_id)
        if hasattr(self, "model"):
            del self.model
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            logger.debug("torch not importable during cleanup — skipping cuda cache clear")
