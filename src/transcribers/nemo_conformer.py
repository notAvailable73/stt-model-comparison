"""NVIDIA NeMo Conformer adapter — covers models 4, 5, 6 from plan.md §3.

Model 4 (`ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large`) is a hybrid
CTC/RNN-T model. Per DECISIONS.md §B it starts on RNN-T; after the first 5
clips it runs a one-clip CTC probe, and if RNN-T median latency is more than
~3× the CTC probe it switches to CTC for the remaining clips. The chosen
decoder is exposed via `current_decoder` so the driver loop can write it to
the predictions CSV under `decoder_variant`.

Models 5 and 6 are not hybrid — `decoder` is left as None and no auto-switch
runs.
"""

from __future__ import annotations

import logging
import statistics
import time

import torch

from src.transcribers.base import BaseTranscriber

logger = logging.getLogger(__name__)

RNNT_SLOWDOWN_THRESHOLD = 3.0
PROBE_AFTER_N_CLIPS = 5


class NemoConformerTranscriber(BaseTranscriber):
    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        decoder: str | None = None,
        language_id: str | None = None,
    ) -> None:
        super().__init__(model_id=model_id, device=device)

        # Lazy import — NeMo is slow to import and heavy. Tests / smoke checks
        # that don't touch NeMo shouldn't pay the cost.
        import nemo.collections.asr as nemo_asr

        logger.info("Loading NeMo model: %s", model_id)
        self.asr = self._load_model(nemo_asr, model_id)
        use_cuda = device.startswith("cuda") and torch.cuda.is_available()
        target_device = "cuda" if use_cuda else "cpu"
        if device.startswith("cuda") and not use_cuda:
            logger.warning("CUDA requested but unavailable — running %s on CPU", model_id)
        # freeze() sets requires_grad=False on every param AND calls eval(),
        # which is the right pattern for inference (matches AI4Bharat's docs).
        self.asr.freeze()
        self.asr = self.asr.to(target_device)

        self.is_hybrid: bool = decoder is not None
        self.current_decoder: str | None = decoder
        if self.is_hybrid:
            self._set_decoder(decoder)  # type: ignore[arg-type]

        # language_id is set in models.yaml only for AI4Bharat models. Stock
        # hishab models reject this kwarg, so we pass it only when set.
        self._language_id: str | None = language_id

        # Auto-switch bookkeeping (only meaningful when is_hybrid is True).
        self._auto_switch_done: bool = not self.is_hybrid
        self._rnnt_latencies: list[float] = []

    @staticmethod
    def _load_model(nemo_asr, model_id: str):
        """Load via from_pretrained; fall back to restore_from on the raw .nemo file.

        NeMo 1.23 + huggingface_hub leaves the downloaded .nemo tarball in the
        cache dir but assumes the dir is already extracted, then fails with
        FileNotFoundError on model_config.yaml. Workaround: pull the .nemo via
        hf_hub_download and call restore_from directly.
        """
        try:
            return nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
        except FileNotFoundError as e:
            if "model_config.yaml" not in str(e):
                raise
            from huggingface_hub import HfApi, hf_hub_download
            files = HfApi().list_repo_files(model_id)
            nemo_file = next((f for f in files if f.endswith(".nemo")), None)
            if not nemo_file:
                raise RuntimeError(f"No .nemo file found in {model_id}") from e
            logger.info("Fallback: downloading %s and using restore_from", nemo_file)
            local = hf_hub_download(repo_id=model_id, filename=nemo_file)
            return nemo_asr.models.ASRModel.restore_from(restore_path=local)

    def _set_decoder(self, decoder_type: str) -> None:
        """Flip the hybrid model's active decoder. No-op for non-hybrid models."""
        logger.info("Setting decoder for %s -> %s", self.model_id, decoder_type)
        self.asr.change_decoding_strategy(decoder_type=decoder_type)
        # Some NeMo versions also key off cur_decoder directly.
        if hasattr(self.asr, "cur_decoder"):
            self.asr.cur_decoder = decoder_type
        self.current_decoder = decoder_type

    def _run(self, audio_path: str) -> str:
        """Run inference on one file and return the decoded string."""
        kwargs: dict = {}
        # AI4Bharat indicconformer requires language_id; passing it to stock
        # hishab models would error, so we only set kwargs when configured.
        if self._language_id:
            kwargs["language_id"] = self._language_id
            kwargs["batch_size"] = 1
            # The hybrid CTC head expects logprobs=False to get plain text out.
            if self.is_hybrid and self.current_decoder == "ctc":
                kwargs["logprobs"] = False

        with torch.inference_mode():
            output = self.asr.transcribe([audio_path], **kwargs)
        # NeMo's transcribe() shape varies by version: List[str], List[Hypothesis],
        # or sometimes a tuple (best_hyps, all_hyps). Normalize.
        if isinstance(output, tuple) and output:
            output = output[0]
        if not output:
            return ""
        first = output[0]
        if isinstance(first, str):
            return first.strip()
        if hasattr(first, "text"):
            return str(first.text).strip()
        return str(first).strip()

    def transcribe(self, audio_path: str) -> tuple[str, float]:
        start = time.perf_counter()
        text = self._run(audio_path)
        latency = time.perf_counter() - start

        # Hybrid-only auto-switch: track RNN-T latencies; after N clips probe CTC once.
        if self.is_hybrid and not self._auto_switch_done:
            if self.current_decoder == "rnnt":
                self._rnnt_latencies.append(latency)
                if len(self._rnnt_latencies) >= PROBE_AFTER_N_CLIPS:
                    self._auto_switch_probe(audio_path)
            else:
                # User explicitly asked for CTC from the start — nothing to probe.
                self._auto_switch_done = True

        return text, latency

    def _auto_switch_probe(self, audio_path: str) -> None:
        """Compare RNN-T median latency against a one-clip CTC probe and pick the winner."""
        rnnt_median = statistics.median(self._rnnt_latencies)

        self._set_decoder("ctc")
        probe_start = time.perf_counter()
        _ = self._run(audio_path)
        ctc_latency = time.perf_counter() - probe_start

        if rnnt_median > RNNT_SLOWDOWN_THRESHOLD * ctc_latency:
            logger.warning(
                "Auto-switch: %s RNN-T median %.2fs > %.1f× CTC probe %.2fs — staying on CTC",
                self.model_id, rnnt_median, RNNT_SLOWDOWN_THRESHOLD, ctc_latency,
            )
            # Decoder already flipped to CTC for the probe; leave it there.
        else:
            logger.info(
                "Auto-switch: %s keeping RNN-T (median %.2fs vs CTC probe %.2fs)",
                self.model_id, rnnt_median, ctc_latency,
            )
            self._set_decoder("rnnt")
        self._auto_switch_done = True

    def cleanup(self) -> None:
        logger.info("Releasing NeMo model: %s", self.model_id)
        if hasattr(self, "asr"):
            del self.asr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
