"""Named-stage orchestrator. Bounded retry. Defined fallback. One pipeline.

The hidden gem fires here: prev_audio_path is threaded from section to
section so MusicGen can use audio continuation. Each section grows from
the last. The song flows rather than stutters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import verifiers as V
from .events import EventEmitter
from .renderers import BeatRenderer, SectionOut
from .spec import SongSpec

MAX_RETRIES = 2
TONE_PAD_PATH = "/tmp/myAIbeats/tone_pad.wav"   # fallback: a neutral tone


class PipelineError(RuntimeError):
    """A blocking gate could not be satisfied within MAX_RETRIES."""


@dataclass
class SectionResult:
    section_id: str
    audio_path: str
    duration_s: float
    clap_score: float
    used_continuation: bool
    clap_gate: V.GateResult
    dur_gate: V.GateResult
    tone_pad_fallback: bool = False
    vocal_path: str | None = None
    vocal_gate: V.GateResult | None = None


@dataclass
class SongManifest:
    title: str
    sections: list[SectionResult] = field(default_factory=list)
    stitch_gate: V.GateResult | None = None
    song_path: str = ""
    ok: bool = False

    def summary(self) -> dict[str, Any]:
        passed = sum(
            int(s.clap_gate.passed) + int(s.dur_gate.passed)
            for s in self.sections
        ) + int(bool(self.stitch_gate and self.stitch_gate.passed))
        failed = sum(
            int(not s.clap_gate.passed) + int(not s.dur_gate.passed)
            for s in self.sections
        ) + int(bool(self.stitch_gate and not self.stitch_gate.passed))
        vocals_dropped = sum(
            1 for s in self.sections
            if s.vocal_gate and not s.vocal_gate.passed
        )
        return {
            "sections": len(self.sections),
            "gates_passed": passed,
            "gates_failed": failed,
            "tone_pad_fallbacks": sum(s.tone_pad_fallback for s in self.sections),
            "vocals_dropped": vocals_dropped,
            "total_duration_s": round(sum(s.duration_s for s in self.sections), 2),
        }


def run(
    spec: SongSpec,
    renderer: BeatRenderer,
    emitter: EventEmitter | None = None,
    out_path: str = "/tmp/myAIbeats/song.wav",
    max_retries: int = MAX_RETRIES,
    limit: int | None = None,
) -> SongManifest:
    em = emitter or EventEmitter()
    sections = spec.sections[:limit] if limit else spec.sections
    em.step_start("spec_load", title=spec.song.title,
                  sections=len(sections), tempo=spec.song.tempo)
    em.step_complete("spec_load")

    manifest = SongManifest(title=spec.song.title)
    prev_audio: str | None = None   # the continuation thread

    for i, section in enumerate(sections):
        declared_s = section.duration_at_tempo(spec.song.tempo)
        em.step_start("section_synth", section=section.id,
                      index=i + 1, total=len(sections),
                      type=section.type, energy=section.energy,
                      continuation=section.continuation and prev_audio is not None)

        tone_pad = False
        out: SectionOut | None = None
        clap_gate = dur_gate = None

        for attempt in range(max_retries + 1):
            out = renderer.synthesize(section, spec, prev_audio)
            clap_gate = V.section_verify(section.id, section.full_prompt,
                                         out.clap_score, section.energy)
            dur_gate  = V.duration_verify(section.id, out.duration_s, declared_s)

            both = clap_gate.passed and dur_gate.passed
            if both:
                em.gate_pass("section_verify", **clap_gate.metrics)
                em.gate_pass("duration_verify", **dur_gate.metrics)
                break
            if not clap_gate.passed:
                em.gate_fail("section_verify", **clap_gate.metrics)
            if not dur_gate.passed:
                em.gate_fail("duration_verify", **dur_gate.metrics)
            if attempt < max_retries:
                em.retry("section_synth", section=section.id, attempt=attempt + 1)
        else:
            # exhausted retries — use tone pad fallback (Article III/VI).
            # The renderer writes a real in-key pad file so the stitch holds.
            tone_pad = True
            pad_path = renderer.tone_pad(section, spec)
            em.fallback("section_synth", section=section.id, to="tone_pad", path=pad_path)
            out = type(out)(
                audio_path=pad_path,
                duration_s=declared_s,
                clap_score=0.0,
                used_continuation=False,
            )

        em.step_complete("section_synth", section=section.id,
                         audio=out.audio_path,
                         clap=round(out.clap_score, 3),
                         continuation=out.used_continuation,
                         fallback=tone_pad)

        # ---- vocal (non-blocking) ----------------------------------------
        vocal_path = None
        vocal_gate = None
        lyric = spec.vocals.lyrics.get(section.id, "")
        # stitch_path is what gets crossfaded into the master. It starts as
        # the instrumental and is replaced by a vocal-mixed version only when
        # a vocal passes its (non-blocking) gate.
        stitch_path = out.audio_path

        if spec.vocals.enabled and lyric:
            em.step_start("vocal_synth", section=section.id)
            v_out = renderer.vocal(section, lyric, spec)
            vocal_gate = V.vocal_verify(section.id, v_out.audio_path, v_out.duration_s)
            if vocal_gate.passed:
                em.gate_pass("vocal_verify", **vocal_gate.metrics)
                vocal_path = v_out.audio_path
                # mix the vocal onto the instrumental for this section
                mixed = out.audio_path.replace(".wav", "_mixed.wav")
                stitch_path = renderer.mix_vocal(out.audio_path, vocal_path, mixed)
            else:
                em.skip("vocal_verify", section=section.id, detail=vocal_gate.detail)
            em.step_complete("vocal_synth", section=section.id)

        sr = SectionResult(
            section_id=section.id,
            audio_path=stitch_path,        # vocal-mixed if a vocal landed, else instrumental
            duration_s=out.duration_s,
            clap_score=out.clap_score,
            used_continuation=out.used_continuation,
            clap_gate=clap_gate,
            dur_gate=dur_gate,
            tone_pad_fallback=tone_pad,
            vocal_path=vocal_path,
            vocal_gate=vocal_gate,
        )
        manifest.sections.append(sr)

        # thread the continuation from the INSTRUMENTAL (out.audio_path),
        # never the vocal-mixed version — keeps musical continuity clean.
        prev_audio = out.audio_path

    # ---- stitch + verify -------------------------------------------------
    em.step_start("stitch", sections=len(spec.sections))
    section_paths = [s.audio_path for s in manifest.sections]
    stitch_out = renderer.stitch(section_paths, spec, out_path)

    # Expected duration is based on ACTUAL rendered section lengths (MusicGen
    # overshoots nominal durations by a few seconds each), minus crossfade
    # overlaps. Using nominal spec durations here would falsely fail the gate.
    expected_s = sum(
        s.duration_s for s in manifest.sections
    ) - (len(manifest.sections) - 1) * spec.audio.crossfade_s

    stitch_gate = V.stitch_verify(
        exists=stitch_out.exists,
        duration_s=stitch_out.duration_s,
        expected_s=expected_s,
        channels=stitch_out.channels,
        expected_channels=2 if spec.audio.channels == "stereo" else 1,
    )
    manifest.stitch_gate = stitch_gate

    if stitch_gate.passed:
        em.gate_pass("stitch_verify", **stitch_gate.metrics)
    else:
        em.gate_fail("stitch_verify", detail=stitch_gate.detail, **stitch_gate.metrics)
        raise PipelineError(f"stitch_verify failed: {stitch_gate.detail}")

    em.step_complete("stitch", path=stitch_out.path)

    manifest.song_path = stitch_out.path
    manifest.ok = True
    em.done(path=stitch_out.path, **manifest.summary())
    return manifest
