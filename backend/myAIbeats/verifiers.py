"""Pure-Python gates. No LLMs. The grader is never the generator.

section_verify  — CLAP cosine(prompt, audio) ≥ threshold  [blocking]
duration_verify — actual_s within tolerance of declared_s  [blocking]
stitch_verify   — final song file: exists, duration, stereo [blocking]
vocal_verify    — vocal audio exists and has content        [non-blocking]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CLAP_THRESHOLD    = 0.20   # audio↔text cosine — analogous to CLIP in my-AI-scene
DURATION_TOL_S    = 2.0    # ±seconds on section duration
STITCH_DUR_TOL_S  = 3.0    # ±seconds on final song duration


@dataclass
class GateResult:
    gate: str
    passed: bool
    blocking: bool
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


def section_verify(section_id: str, prompt: str, clap_score: float) -> GateResult:
    """Blocking. The generated audio must sound like what was asked for."""
    passed = clap_score >= CLAP_THRESHOLD
    return GateResult(
        gate="section_verify", passed=passed, blocking=True,
        detail=f"CLAP score {clap_score:.3f} vs threshold {CLAP_THRESHOLD}",
        metrics={"clap_score": round(clap_score, 4), "threshold": CLAP_THRESHOLD,
                 "section": section_id},
    )


def duration_verify(section_id: str, actual_s: float, declared_s: float) -> GateResult:
    """Blocking. Audio must match the declared bar count ±tolerance."""
    diff = abs(actual_s - declared_s)
    passed = diff <= DURATION_TOL_S
    return GateResult(
        gate="duration_verify", passed=passed, blocking=True,
        detail=f"actual {actual_s:.2f}s vs declared {declared_s:.2f}s (diff {diff:+.2f}s)",
        metrics={"actual_s": round(actual_s, 3), "declared_s": round(declared_s, 3),
                 "diff_s": round(diff, 3), "tol_s": DURATION_TOL_S,
                 "section": section_id},
    )


def stitch_verify(
    *, exists: bool, duration_s: float, expected_s: float,
    channels: int, expected_channels: int,
) -> GateResult:
    """Blocking. The stitched song must exist and match declared structure."""
    problems = []
    if not exists:
        problems.append("output file missing")
    dur_diff = abs(duration_s - expected_s)
    if dur_diff > STITCH_DUR_TOL_S:
        problems.append(f"duration {duration_s:.1f}s ≠ expected {expected_s:.1f}s")
    if channels != expected_channels:
        problems.append(f"channels {channels} ≠ expected {expected_channels}")
    passed = not problems
    return GateResult(
        gate="stitch_verify", passed=passed, blocking=True,
        detail="ok" if passed else "; ".join(problems),
        metrics={"duration_s": round(duration_s, 2), "expected_s": round(expected_s, 2),
                 "channels": channels},
    )


def vocal_verify(section_id: str, audio_path: str | None, duration_s: float | None) -> GateResult:
    """NON-blocking. Vocals enhance — a failed vocal is dropped, never fails the song."""
    if not audio_path:
        return GateResult(gate="vocal_verify", passed=False, blocking=False,
                          detail="no vocal audio produced",
                          metrics={"section": section_id})
    if not duration_s or duration_s < 0.1:
        return GateResult(gate="vocal_verify", passed=False, blocking=False,
                          detail=f"vocal too short: {duration_s}s",
                          metrics={"section": section_id, "duration_s": duration_s})
    return GateResult(gate="vocal_verify", passed=True, blocking=False,
                      detail=f"vocal ok ({duration_s:.2f}s)",
                      metrics={"section": section_id, "duration_s": round(duration_s, 3)})
