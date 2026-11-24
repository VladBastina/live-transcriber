"""Online speaker diarization: embed each utterance, cluster by cosine similarity.

Uses a WeSpeaker ResNet34 embedding model through sherpa-onnx (CPU, ~25 MB).
Speakers are discovered on the fly: Speaker 1, Speaker 2, ...
"""
import os
import numpy as np

MODEL_NAME = "wespeaker_en_voxceleb_resnet34.onnx"
MIN_EMBED_SEC = 1.0      # shorter audio gives unreliable voiceprints


class Diarizer:
    def __init__(self, model_path, threshold=0.5, max_speakers=12):
        import sherpa_onnx
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=model_path, num_threads=2, provider="cpu")
        self.ext = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self.threshold = threshold
        self.max_speakers = max_speakers
        self.cent = []       # unit-norm centroids
        self.count = []      # utterances merged into each centroid
        self.last = None     # last speaker id (fallback for short clips)

    def _embed(self, audio):
        s = self.ext.create_stream()
        s.accept_waveform(sample_rate=16000, waveform=audio.astype(np.float32))
        s.input_finished()
        e = np.asarray(self.ext.compute(s), dtype=np.float32)
        return e / (np.linalg.norm(e) + 1e-9)

    def identify(self, audio):
        """Return a 0-based speaker id for this utterance."""
        if len(audio) / 16000 < MIN_EMBED_SEC:
            # too short to fingerprint: assume whoever spoke last (None if nobody yet)
            return self.last
        e = self._embed(audio)
        if not self.cent:
            return self._new(e)
        sims = np.array([float(c @ e) for c in self.cent])
        k = int(sims.argmax())
        if sims[k] >= self.threshold or len(self.cent) >= self.max_speakers:
            n = self.count[k]                      # running mean keeps the voiceprint stable
            c = self.cent[k] * n + e
            self.cent[k] = c / (np.linalg.norm(c) + 1e-9)
            self.count[k] += 1
            self.last = k
            return k
        return self._new(e)

    def _new(self, e):
        self.cent.append(e); self.count.append(1)
        self.last = len(self.cent) - 1
        return self.last
