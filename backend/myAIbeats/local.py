"""LocalRenderer — the real HuggingFace implementation of BeatRenderer.

The hidden gem lives here: synthesize() passes the last continuation_secs
of the previous section as audio conditioning to MusicGen. Each section
grows from the last. The song flows.

Trust separation (CONSTITUTION Article III): MusicGen generates audio,
CLAP scores it for section_verify. The generator never grades itself.
"""
from __future__ import annotations

import os
import subprocess
import wave
from pathlib import Path
from typing import Protocol

import numpy as np

from .renderers import BeatRenderer, SectionOut, StitchOut, VocalOut
from .spec import Section, SongSpec

MUSICGEN_MODEL  = "facebook/musicgen-stereo-medium"
CLAP_MODEL      = "laion/clap-htsat-unfused"
BARK_VOICE      = "v2/en_speaker_6"
MUSICGEN_SR     = 32000   # MusicGen native sample rate
OUTPUT_SR       = 44100   # final output sample rate
TOKENS_PER_SEC  = 50      # MusicGen encoding rate


# ---- helpers ------------------------------------------------------------

def _device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Simple linear resample via scipy."""
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(from_sr, to_sr)
    return resample_poly(audio, to_sr // g, from_sr // g, axis=-1).astype(np.float32)


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    """Write float32 [-1,1] audio to a WAV file (stdlib only)."""
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype("<i2")
    # audio shape: (channels, samples) or (samples,)
    if pcm.ndim == 1:
        channels, frames = 1, len(pcm)
        data = pcm.tobytes()
    else:
        channels, frames = pcm.shape[0], pcm.shape[1]
        data = pcm.T.tobytes()   # interleaved
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data)


def _read_wav_tail(path: str, tail_secs: float, sr: int = MUSICGEN_SR) -> np.ndarray | None:
    """Read the last tail_secs of a WAV as float32 at the given sample rate."""
    if not os.path.exists(path):
        return None
    try:
        with wave.open(path, "rb") as w:
            file_sr   = w.getframerate()
            n_ch      = w.getnchannels()
            n_frames  = w.getnframes()
            tail_frames = int(tail_secs * file_sr)
            start = max(0, n_frames - tail_frames)
            w.setpos(start)
            raw = w.readframes(n_frames - start)
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0
        if n_ch > 1:
            pcm = pcm.reshape(-1, n_ch).T   # (channels, samples)
        # resample to target sr if needed
        if file_sr != sr:
            pcm = _resample(pcm, file_sr, sr)
        return pcm
    except Exception:
        return None


def _ffprobe_duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


# ---- MusicGen synthesizer -----------------------------------------------

class MusicGenSynth:
    """MusicGen stereo with audio continuation.

    The hidden gem: when prev_audio is supplied, it is passed as
    audio conditioning so the new section grows from where the last
    one ended — musical rather than arbitrary stitching.
    """

    def __init__(self, model_id: str = MUSICGEN_MODEL):
        self.model_id = model_id
        self._model   = None
        self._proc    = None

    def _ensure(self):
        if self._model is None:
            import torch
            from transformers import AutoProcessor, MusicgenForConditionalGeneration
            dev = _device()
            # Use float32 throughout — MPS audio ops (EnCodec conv layers)
            # have dtype mismatches with float16 when continuation audio
            # is passed through the encoder. float32 is safe on all devices.
            self._proc  = AutoProcessor.from_pretrained(self.model_id)
            self._model = MusicgenForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=torch.float32
            ).to(dev)
        return self._model, self._proc

    def generate(
        self,
        prompt: str,
        duration_s: float,
        prev_audio: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate stereo audio. Returns float32 (channels, samples) at MUSICGEN_SR."""
        import torch
        model, proc = self._ensure()
        dev = next(model.parameters()).device
        n_tokens = int(duration_s * TOKENS_PER_SEC) + 32

        if prev_audio is not None:
            # THE HIDDEN GEM: pass the tail of the previous section
            # as audio conditioning — MusicGen continues from it
            inputs = proc(
                text=[prompt],
                audio=prev_audio,
                sampling_rate=MUSICGEN_SR,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = proc(text=[prompt], padding=True, return_tensors="pt")

        inputs = {k: v.to(dev) for k, v in inputs.items()
                  if isinstance(v, torch.Tensor)}

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=n_tokens)

        # out: (batch, channels, samples) — take batch[0]
        audio = out[0].cpu().numpy().astype(np.float32)
        return audio   # (channels, samples) — stereo


# ---- CLAP scorer --------------------------------------------------------

class CLAPScorer:
    """laion/clap-htsat-unfused via transformers.
    Returns cosine similarity between audio and text in [0, 1].
    This is section_verify's judge — never the model being scored."""

    def __init__(self, model_id: str = CLAP_MODEL):
        self.model_id = model_id
        self._model   = None
        self._proc    = None

    CLAP_SR = 48000   # CLAP's native sample rate

    def _ensure(self):
        if self._model is None:
            from transformers import AutoTokenizer, ClapFeatureExtractor, ClapModel
            self._fe    = ClapFeatureExtractor.from_pretrained(self.model_id)
            self._tok   = AutoTokenizer.from_pretrained(self.model_id)
            self._model = ClapModel.from_pretrained(self.model_id)
            self._model.eval()
        return self._model

    def score(self, wav_path: str, text: str) -> float:
        import torch
        model = self._ensure()
        try:
            with wave.open(wav_path, "rb") as w:
                sr, n_ch = w.getframerate(), w.getnchannels()
                raw = w.readframes(w.getnframes())
            pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32767.0
            if n_ch > 1:
                pcm = pcm.reshape(-1, n_ch).mean(axis=1)   # mono for CLAP
            if sr != self.CLAP_SR:
                pcm = _resample(pcm, sr, self.CLAP_SR)
            audio_inputs = self._fe(
                raw_speech=[pcm], return_tensors="pt",
                sampling_rate=self.CLAP_SR, padding=True,
            )
            text_inputs = self._tok([text], return_tensors="pt", padding=True)
            with torch.no_grad():
                ae = model.get_audio_features(**audio_inputs).pooler_output
                te = model.get_text_features(**text_inputs).pooler_output
            ae = ae / ae.norm(dim=-1, keepdim=True)
            te = te / te.norm(dim=-1, keepdim=True)
            return float((ae * te).sum().clamp(0.0, 1.0))
        except Exception as e:
            print(f"[CLAP] scoring failed: {e}")
            return 0.0


# ---- the renderer -------------------------------------------------------

class LocalRenderer:
    def __init__(
        self,
        out_dir: str | Path = "/tmp/myAIbeats",
        synth: MusicGenSynth | None = None,
        scorer: CLAPScorer | None = None,
        fps: int = 24,
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.synth  = synth  or MusicGenSynth()
        self.scorer = scorer or CLAPScorer()

    # ---- Phase 2: synthesize + CLAP score --------------------------------

    def synthesize(
        self,
        section: Section,
        spec: SongSpec,
        prev_audio_path: str | None,
    ) -> SectionOut:
        duration_s = section.duration_at_tempo(spec.song.tempo)

        # pull the tail of the previous section for continuation
        prev_audio = None
        used_continuation = False
        if prev_audio_path and section.continuation:
            prev_audio = _read_wav_tail(
                prev_audio_path, spec.song.continuation_secs, MUSICGEN_SR
            )
            used_continuation = prev_audio is not None

        # synthesize
        audio = self.synth.generate(
            prompt=section.full_prompt,
            duration_s=duration_s,
            prev_audio=prev_audio,
        )

        # resample to output sample rate and save
        if audio.shape[-1] and MUSICGEN_SR != OUTPUT_SR:
            audio = _resample(audio, MUSICGEN_SR, OUTPUT_SR)

        path = self.out_dir / f"{section.id}.wav"
        _write_wav(path, audio, OUTPUT_SR)

        # CLAP scores the output — trust separation: not the generator
        clap_score = self.scorer.score(str(path), section.full_prompt)

        return SectionOut(
            audio_path=str(path),
            duration_s=_ffprobe_duration(str(path)),
            clap_score=clap_score,
            used_continuation=used_continuation,
        )

    # ---- Phase 3: stitch (pending) ---------------------------------------

    def stitch(
        self,
        section_paths: list[str],
        spec: SongSpec,
        out_path: str,
    ) -> StitchOut:
        raise NotImplementedError("stitch(): Phase 3 — ffmpeg acrossfade + normalize")

    # ---- Phase 4: vocal (pending) ----------------------------------------

    def vocal(self, section: Section, lyric: str, spec: SongSpec) -> VocalOut:
        raise NotImplementedError("vocal(): Phase 4 — Bark TTS")
