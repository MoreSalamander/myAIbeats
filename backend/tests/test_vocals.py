"""Phase 4 tests — vocal synth (fake Bark) + real ffmpeg mix_vocal.
Vocals are non-blocking: every failure path must keep the song alive."""
import json
import wave
from pathlib import Path
import numpy as np
import pytest

from myAIbeats.local import LocalRenderer, _write_wav, OUTPUT_SR
from myAIbeats.spec import load_spec, parse_spec

SPEC_PATH = Path(__file__).resolve().parents[2] / "specs" / "something_blue.json"


def _vocal_spec(tmp_lyrics):
    raw = json.loads(SPEC_PATH.read_text())
    raw["vocals"]["enabled"] = True
    raw["vocals"]["lyrics"] = tmp_lyrics
    return parse_spec(raw)


def _tone(seconds, freq=220.0, sr=OUTPUT_SR):
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    mono = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
    return np.stack([mono, mono])


class FakeBark:
    """Returns a short mono tone as a stand-in vocal. No model download."""
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []
    def synth(self, text, voice):
        self.calls.append({"text": text, "voice": voice})
        if self.fail:
            raise RuntimeError("bark blew up")
        sr = 24000
        dur = 2.0
        t = np.linspace(0, dur, int(dur * sr), endpoint=False)
        return (0.4 * np.sin(2 * np.pi * 330 * t)).astype(np.float32), sr


def test_vocal_synth_writes_wav(tmp_path):
    r = LocalRenderer(out_dir=tmp_path, bark=FakeBark())
    spec = _vocal_spec({"s02": "something blue in the morning light"})
    sec = next(s for s in spec.sections if s.id == "s02")
    out = r.vocal(sec, spec.vocals.lyrics["s02"], spec)
    assert out.audio_path is not None
    assert Path(out.audio_path).exists()
    assert out.duration_s > 0


def test_vocal_wraps_lyric_for_singing(tmp_path):
    bark = FakeBark()
    r = LocalRenderer(out_dir=tmp_path, bark=bark)
    spec = _vocal_spec({"s02": "hold the line"})
    sec = next(s for s in spec.sections if s.id == "s02")
    r.vocal(sec, "hold the line", spec)
    # ♪ wrapping biases Bark toward singing
    assert "♪" in bark.calls[0]["text"]
    assert "hold the line" in bark.calls[0]["text"]


def test_vocal_failure_returns_empty_not_raises(tmp_path):
    r = LocalRenderer(out_dir=tmp_path, bark=FakeBark(fail=True))
    spec = _vocal_spec({"s02": "this will fail"})
    sec = next(s for s in spec.sections if s.id == "s02")
    out = r.vocal(sec, "this will fail", spec)   # must NOT raise
    assert out.audio_path is None
    assert out.duration_s is None


def test_mix_vocal_overlays_onto_instrumental(tmp_path):
    r = LocalRenderer(out_dir=tmp_path, bark=FakeBark())
    instr = tmp_path / "s02.wav"
    vocal = tmp_path / "s02_vocal.wav"
    _write_wav(instr, _tone(6.0, freq=220), OUTPUT_SR)
    _write_wav(vocal, _tone(2.0, freq=440)[0], 24000)   # mono vocal
    mixed = r.mix_vocal(str(instr), str(vocal), str(tmp_path / "s02_mixed.wav"))
    assert Path(mixed).exists()
    with wave.open(mixed) as w:
        assert w.getnchannels() == 2
        # mixed length matches the instrumental (duration=first)
        assert abs(w.getnframes() / w.getframerate() - 6.0) < 1.0


def test_full_pipeline_with_vocals_real_mix(tmp_path):
    """End-to-end: fake synth + fake bark + REAL ffmpeg stitch & mix.
    Vocals land on some sections; song still completes."""
    from myAIbeats.events import EventEmitter
    from myAIbeats.pipeline import run

    spec = _vocal_spec({
        "s02": "something blue in the morning",
        "s03": "and i would wait for you",
    })
    r = LocalRenderer(out_dir=tmp_path, bark=FakeBark())

    # pre-write instrumental tones for each section (stand-in for MusicGen)
    for sec in spec.sections:
        _write_wav(tmp_path / f"{sec.id}.wav",
                   _tone(sec.duration_at_tempo(spec.song.tempo)), OUTPUT_SR)

    # fake synth returns those real tone paths
    from myAIbeats.renderers import SectionOut
    def fake_synth(section, spec, prev):
        return SectionOut(
            audio_path=str(tmp_path / f"{section.id}.wav"),
            duration_s=section.duration_at_tempo(spec.song.tempo),
            clap_score=0.27,
            used_continuation=bool(prev and section.continuation),
        )
    r.synthesize = fake_synth

    em = EventEmitter(out=None)
    m = run(spec, r, em, out_path=str(tmp_path / "song.wav"))
    assert m.ok
    assert m.stitch_gate.passed
    # the two vocal sections should have a vocal_path; others None
    s02 = next(s for s in m.sections if s.section_id == "s02")
    assert s02.vocal_path is not None
    assert s02.vocal_gate.passed
    # the mixed file is what fed the stitch
    assert "_mixed" in s02.audio_path
    assert Path(m.song_path).exists()


def test_vocal_drop_keeps_song_alive_real(tmp_path):
    """A failing vocal on one section must not corrupt the song."""
    from myAIbeats.events import EventEmitter
    from myAIbeats.pipeline import run
    from myAIbeats.renderers import SectionOut

    spec = _vocal_spec({"s02": "doomed vocal line"})
    r = LocalRenderer(out_dir=tmp_path, bark=FakeBark(fail=True))
    for sec in spec.sections:
        _write_wav(tmp_path / f"{sec.id}.wav",
                   _tone(sec.duration_at_tempo(spec.song.tempo)), OUTPUT_SR)
    r.synthesize = lambda section, spec, prev: SectionOut(
        audio_path=str(tmp_path / f"{section.id}.wav"),
        duration_s=section.duration_at_tempo(spec.song.tempo),
        clap_score=0.27, used_continuation=False,
    )
    em = EventEmitter(out=None)
    m = run(spec, r, em, out_path=str(tmp_path / "song.wav"))
    assert m.ok                       # song survives the vocal failure
    s02 = next(s for s in m.sections if s.section_id == "s02")
    assert s02.vocal_path is None     # vocal dropped
    assert "_mixed" not in s02.audio_path   # stitched the bare instrumental
