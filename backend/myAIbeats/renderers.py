"""BeatRenderer protocol + ScriptedRenderer offline fake.

The pipeline is defined entirely behind this protocol so it can be
proven offline with zero model downloads. LocalRenderer (Phase 2)
implements the same contract over the HuggingFace stack.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .spec import Section, SongSpec


@dataclass
class SectionOut:
    audio_path: str
    duration_s: float
    clap_score: float         # CLAP cosine(prompt, audio) — for section_verify
    used_continuation: bool   # whether the hidden gem fired


@dataclass
class StitchOut:
    path: str
    exists: bool
    duration_s: float
    channels: int


@dataclass
class VocalOut:
    audio_path: str | None
    duration_s: float | None


class BeatRenderer(Protocol):
    def synthesize(
        self,
        section: Section,
        spec: SongSpec,
        prev_audio_path: str | None,   # None for first section
    ) -> SectionOut: ...

    def stitch(
        self,
        section_paths: list[str],
        spec: SongSpec,
        out_path: str,
    ) -> StitchOut: ...

    def vocal(
        self,
        section: Section,
        lyric: str,
        spec: SongSpec,
    ) -> VocalOut: ...

    def mix_vocal(
        self,
        instrumental_path: str,
        vocal_path: str,
        out_path: str,
    ) -> str:
        """Overlay a vocal onto an instrumental section. Returns the mixed
        audio path (used for stitching). Continuation always threads from
        the instrumental, never the vocal-mixed version."""
        ...


# ---- offline fake -------------------------------------------------------

@dataclass
class ScriptedRenderer:
    """Deterministic gate-passing fake. Inject failures for test coverage."""
    fail_sections: set[str]  = field(default_factory=set)   # section ids to fail CLAP
    fail_duration: set[str]  = field(default_factory=set)   # section ids to fail duration
    fail_vocal: set[str]     = field(default_factory=set)   # section ids to fail vocal
    heal_after: int          = 0
    _attempts: dict[str, int] = field(default_factory=dict)

    def _attempt(self, key: str) -> int:
        self._attempts[key] = self._attempts.get(key, 0) + 1
        return self._attempts[key]

    def synthesize(self, section: Section, spec: SongSpec,
                   prev_audio_path: str | None) -> SectionOut:
        n = self._attempt(f"synth:{section.id}")
        clap = 0.08 if (section.id in self.fail_sections
                        and n <= self.heal_after) else 0.27
        dur = section.duration_at_tempo(spec.song.tempo)
        if section.id in self.fail_duration and n <= self.heal_after:
            dur += 8.0   # push over tolerance
        return SectionOut(
            audio_path=f"/tmp/myAIbeats/{section.id}.wav",
            duration_s=dur,
            clap_score=clap,
            used_continuation=bool(prev_audio_path and section.continuation),
        )

    def stitch(self, section_paths: list[str], spec: SongSpec,
               out_path: str) -> StitchOut:
        # Expected total without crossfade overlaps
        total = sum(
            sec.duration_at_tempo(spec.song.tempo)
            for sec in spec.sections
        ) - (len(spec.sections) - 1) * spec.audio.crossfade_s
        return StitchOut(path=out_path, exists=True,
                         duration_s=max(total, 0),
                         channels=2 if spec.audio.channels == "stereo" else 1)

    def vocal(self, section: Section, lyric: str,
              spec: SongSpec) -> VocalOut:
        if section.id in self.fail_vocal:
            return VocalOut(audio_path=None, duration_s=None)
        return VocalOut(
            audio_path=f"/tmp/myAIbeats/{section.id}_vocal.wav",
            duration_s=max(1.0, len(lyric.split()) * 0.4),
        )

    def mix_vocal(self, instrumental_path: str, vocal_path: str,
                  out_path: str) -> str:
        # offline: no real mixing — the mixed section is just the instrumental
        return instrumental_path
