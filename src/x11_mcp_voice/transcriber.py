from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class Transcriber:
    """Speech-to-text via NVIDIA Parakeet TDT (NeMo toolkit)."""

    def __init__(self, model_name: str = "nvidia/parakeet-tdt-0.6b-v2", device: str = "cuda"):
        log.info("Loading STT model %s on %s...", model_name, device)
        import nemo.collections.asr as nemo_asr

        self._model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
        self._model = self._model.to(device)
        self._model.eval()
        log.info("STT model loaded")

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio to text.

        Args:
            audio: 16kHz mono float32 numpy array.

        Returns:
            Transcribed text string.
        """
        import torch

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # NeMo expects [batch, time] tensor
        signal = torch.tensor(audio).unsqueeze(0).to(self._model.device)
        signal_len = torch.tensor([audio.shape[0]]).to(self._model.device)

        with torch.no_grad():
            hypotheses = self._model.transcribe(audio=[audio], batch_size=1)

        if not hypotheses:
            return ""

        # hypotheses is a list of strings (one per audio in the batch)
        text = hypotheses[0] if isinstance(hypotheses[0], str) else hypotheses[0].text
        return text.strip()
