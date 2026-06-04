"""SongSpec schema + loader + structural validation.

Structural failures are authoring bugs — they raise SpecError before the
pipeline runs, never during model synthesis.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BEATS_PER_BAR = 4   # 4/4 time assumed; configurable via time_signature
ENERGY_TOL    = 1e-6
CONTINUATION_SECS_DEFAULT = 3.0


class SpecError(ValueError):
    """Structural / authoring error in a SongSpec."""


@dataclass(frozen=True)
class AudioSettings:
    sample_rate: int   = 44100
    channels: str      = "stereo"
    crossfade_s: float = 1.5
    normalize: bool    = True


@dataclass(frozen=True)
class VocalSettings:
    enabled: bool  = False
    engine: str    = "bark"
    voice: str     = "v2/en_speaker_6"
    lyrics: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Song:
    title: str
    genre: str
    tempo: float
    key: str
    mood: str
    mode: str               = "major"
    time_signature: str     = "4/4"
    instruments: list[str]  = field(default_factory=list)
    reference_feel: str     = ""
    continuation_secs: float = CONTINUATION_SECS_DEFAULT


@dataclass(frozen=True)
class Section:
    id: str
    type: str       # intro / verse / chorus / bridge / outro / break
    bars: int
    prompt: str
    energy: float   # 0.0–1.0
    continuation: bool = True

    @property
    def duration_s(self) -> float:
        """Nominal duration in seconds (4/4 assumed)."""
        return (self.bars * BEATS_PER_BAR * 60.0) / 1.0  # tempo injected at pipeline time

    def duration_at_tempo(self, tempo: float) -> float:
        return (self.bars * BEATS_PER_BAR * 60.0) / tempo

    @property
    def energy_tag(self) -> str:
        if self.energy < 0.4:
            return "minimal, sparse, quiet"
        if self.energy < 0.7:
            return "medium energy, balanced"
        return "full, energetic, powerful"

    @property
    def full_prompt(self) -> str:
        """Prompt with energy tag injected."""
        return f"{self.prompt}, {self.energy_tag}"


@dataclass(frozen=True)
class SongSpec:
    song: Song
    sections: list[Section]
    audio: AudioSettings
    vocals: VocalSettings


def _require(d: dict, keys: tuple, where: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise SpecError(f"{where}: missing required fields: {', '.join(missing)}")


def load_spec(path: str | Path) -> SongSpec:
    p = Path(path)
    try:
        raw = json.loads(p.read_text())
    except FileNotFoundError as e:
        raise SpecError(f"spec file not found: {p}") from e
    except json.JSONDecodeError as e:
        raise SpecError(f"spec is not valid JSON: {e}") from e
    return parse_spec(raw)


def parse_spec(raw: dict[str, Any]) -> SongSpec:
    _require(raw, ("song", "sections"), "spec")
    s = raw["song"]
    _require(s, ("title", "genre", "tempo", "key", "mood"), "song")

    song = Song(
        title=s["title"], genre=s["genre"],
        tempo=float(s["tempo"]), key=s["key"], mood=s["mood"],
        mode=s.get("mode", "major"),
        time_signature=s.get("time_signature", "4/4"),
        instruments=s.get("instruments", []),
        reference_feel=s.get("reference_feel", ""),
        continuation_secs=float(s.get("continuation_secs", CONTINUATION_SECS_DEFAULT)),
    )

    if not raw["sections"]:
        raise SpecError("spec has no sections")

    sections = []
    for i, sec in enumerate(raw["sections"]):
        _require(sec, ("id", "type", "bars", "prompt", "energy"), f"sections[{i}]")
        energy = float(sec["energy"])
        if not (0.0 - ENERGY_TOL <= energy <= 1.0 + ENERGY_TOL):
            raise SpecError(f"sections[{i}].energy {energy} out of [0,1]")
        if int(sec["bars"]) < 1:
            raise SpecError(f"sections[{i}].bars must be ≥ 1")

        # first section should not use continuation
        cont = sec.get("continuation", i > 0)
        if i == 0 and cont:
            cont = False   # silently correct — first section has nothing to continue from

        sections.append(Section(
            id=sec["id"], type=sec["type"],
            bars=int(sec["bars"]), prompt=sec["prompt"],
            energy=min(max(energy, 0.0), 1.0),
            continuation=cont,
        ))

    ids = [sec.id for sec in sections]
    if len(ids) != len(set(ids)):
        raise SpecError("section ids are not unique")

    a = raw.get("audio", {})
    audio = AudioSettings(
        sample_rate=int(a.get("sample_rate", 44100)),
        channels=a.get("channels", "stereo"),
        crossfade_s=float(a.get("crossfade_s", 1.5)),
        normalize=bool(a.get("normalize", True)),
    )

    v = raw.get("vocals", {})
    vocals = VocalSettings(
        enabled=bool(v.get("enabled", False)),
        engine=v.get("engine", "bark"),
        voice=v.get("voice", "v2/en_speaker_6"),
        lyrics=v.get("lyrics", {}),
    )

    return SongSpec(song=song, sections=sections, audio=audio, vocals=vocals)
