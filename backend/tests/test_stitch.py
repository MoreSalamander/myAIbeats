"""Phase 3 tests — LocalRenderer.stitch() with real ffmpeg, synthetic WAVs.
Requires system ffmpeg. No MusicGen / CLAP downloads."""
import wave
from pathlib import Path
import numpy as np
import pytest

from myAIbeats.local import LocalRenderer, _write_wav, OUTPUT_SR
from myAIbeats.spec import load_spec, parse_spec
import json

SPEC_PATH = Path(__file__).resolve().parents[2] / "specs" / "something_blue.json"


@pytest.fixture
def spec():
    return load_spec(SPEC_PATH)


def _tone(seconds: float, freq: float = 220.0, sr: int = OUTPUT_SR) -> np.ndarray:
    """Stereo sine tone, shape (2, samples)."""
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    mono = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    return np.stack([mono, mono])


def _make_sections(tmp_path, durations, sr=OUTPUT_SR):
    paths = []
    for i, d in enumerate(durations):
        p = tmp_path / f"sec{i}.wav"
        _write_wav(p, _tone(d, freq=220 + i * 110), sr)
        paths.append(str(p))
    return paths


def test_stitch_three_sections(spec, tmp_path):
    r = LocalRenderer(out_dir=tmp_path)
    paths = _make_sections(tmp_path, [4.0, 4.0, 4.0])
    out = r.stitch(paths, spec, str(tmp_path / "song.wav"))
    assert out.exists
    assert out.channels == 2
    # 3×4s with 2 crossfades of 1.5s ≈ 12 - 3 = 9s (±1s for loudnorm framing)
    assert abs(out.duration_s - 9.0) < 1.5


def test_stitch_single_section(spec, tmp_path):
    r = LocalRenderer(out_dir=tmp_path)
    paths = _make_sections(tmp_path, [5.0])
    out = r.stitch(paths, spec, str(tmp_path / "song.wav"))
    assert out.exists
    assert out.channels == 2
    assert abs(out.duration_s - 5.0) < 1.5


def test_stitch_without_normalize(tmp_path):
    raw = json.loads(SPEC_PATH.read_text())
    raw["audio"]["normalize"] = False
    spec = parse_spec(raw)
    r = LocalRenderer(out_dir=tmp_path)
    paths = _make_sections(tmp_path, [3.0, 3.0])
    out = r.stitch(paths, spec, str(tmp_path / "song.wav"))
    assert out.exists
    # 2×3s, 1 crossfade 1.5s ≈ 6 - 1.5 = 4.5s
    assert abs(out.duration_s - 4.5) < 1.0


def test_stitch_output_is_valid_wav(spec, tmp_path):
    r = LocalRenderer(out_dir=tmp_path)
    paths = _make_sections(tmp_path, [4.0, 4.0])
    out = r.stitch(paths, spec, str(tmp_path / "song.wav"))
    with wave.open(out.path) as w:
        assert w.getframerate() == spec.audio.sample_rate
        assert w.getnchannels() == 2
        assert w.getnframes() > 0


def test_full_pipeline_real_stitch(spec, tmp_path):
    """End-to-end: ScriptedRenderer synth (fake) but REAL ffmpeg stitch."""
    from myAIbeats.events import EventEmitter
    from myAIbeats.pipeline import run
    from myAIbeats.renderers import ScriptedRenderer

    # ScriptedRenderer makes fake section paths that don't exist on disk,
    # so we make a hybrid: real LocalRenderer.stitch + pre-written tones.
    r = LocalRenderer(out_dir=tmp_path)

    # Pre-write a tone for every section id the scripted synth will "produce"
    scripted = ScriptedRenderer()
    for sec in spec.sections:
        _write_wav(tmp_path / f"{sec.id}.wav",
                   _tone(sec.duration_at_tempo(spec.song.tempo)), OUTPUT_SR)

    # Borrow scripted synth (returns those exact paths) + real stitch
    r.synthesize = lambda section, spec, prev: scripted.synthesize(
        section, spec, prev
    ).__class__(
        audio_path=str(tmp_path / f"{section.id}.wav"),
        duration_s=section.duration_at_tempo(spec.song.tempo),
        clap_score=0.27,
        used_continuation=bool(prev and section.continuation),
    )

    em = EventEmitter(out=None)
    m = run(spec, r, em, out_path=str(tmp_path / "something_blue.wav"))
    assert m.ok
    assert m.stitch_gate.passed
    assert Path(m.song_path).exists()
