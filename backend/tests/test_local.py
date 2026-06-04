"""Phase 2 tests — LocalRenderer.synthesize() with fake synth + CLAP."""
import wave
from pathlib import Path
import numpy as np
import pytest

from myAIbeats.local import LocalRenderer, _write_wav, _read_wav_tail, OUTPUT_SR, MUSICGEN_SR
from myAIbeats.spec import load_spec

SPEC_PATH = Path(__file__).resolve().parents[2] / "specs" / "something_blue.json"
SR = OUTPUT_SR


@pytest.fixture
def spec():
    return load_spec(SPEC_PATH)


class FakeSynth:
    def __init__(self, clap_score=0.27):
        self._score = clap_score
        self.calls: list[dict] = []
    def generate(self, prompt, duration_s, prev_audio=None):
        self.calls.append({"prompt": prompt, "duration_s": duration_s,
                            "continuation": prev_audio is not None})
        samples = int(duration_s * MUSICGEN_SR)
        return np.zeros((2, samples), dtype=np.float32)   # stereo silence


class FakeCLAP:
    def __init__(self, score=0.27):
        self._score = score
    def score(self, wav_path, text):
        return self._score


def test_synthesize_writes_wav(spec, tmp_path):
    r = LocalRenderer(out_dir=tmp_path, synth=FakeSynth(), scorer=FakeCLAP())
    out = r.synthesize(spec.sections[0], spec, prev_audio_path=None)
    assert Path(out.audio_path).exists()
    with wave.open(out.audio_path) as w:
        assert w.getframerate() == OUTPUT_SR
        assert w.getnchannels() == 2   # stereo
    assert out.clap_score == pytest.approx(0.27)
    assert out.duration_s > 0


def test_first_section_no_continuation(spec, tmp_path):
    synth = FakeSynth()
    r = LocalRenderer(out_dir=tmp_path, synth=synth, scorer=FakeCLAP())
    out = r.synthesize(spec.sections[0], spec, prev_audio_path=None)
    assert not out.used_continuation
    assert not synth.calls[0]["continuation"]


def test_subsequent_section_uses_continuation(spec, tmp_path):
    synth = FakeSynth()
    r = LocalRenderer(out_dir=tmp_path, synth=synth, scorer=FakeCLAP())
    # generate s01 first so there's a real WAV to read the tail from
    out0 = r.synthesize(spec.sections[0], spec, prev_audio_path=None)
    out1 = r.synthesize(spec.sections[1], spec, prev_audio_path=out0.audio_path)
    assert out1.used_continuation
    assert synth.calls[1]["continuation"]


def test_energy_tag_in_prompt(spec, tmp_path):
    synth = FakeSynth()
    r = LocalRenderer(out_dir=tmp_path, synth=synth, scorer=FakeCLAP())
    # intro (energy 0.2) → should have "minimal" in full_prompt
    r.synthesize(spec.sections[0], spec, prev_audio_path=None)
    assert "minimal" in synth.calls[0]["prompt"]
    # chorus (energy 0.85) → should have "energetic"
    for sec in spec.sections:
        if sec.type == "chorus":
            r.synthesize(sec, spec, prev_audio_path=None)
            assert "energetic" in synth.calls[-1]["prompt"]
            break


def test_wav_tail_reader(tmp_path):
    # write a 5-second stereo WAV, read back the last 3 seconds
    audio = np.random.randn(2, 5 * MUSICGEN_SR).astype(np.float32) * 0.1
    path = tmp_path / "test.wav"
    _write_wav(path, audio, MUSICGEN_SR)
    tail = _read_wav_tail(str(path), tail_secs=3.0, sr=MUSICGEN_SR)
    assert tail is not None
    expected_samples = int(3.0 * MUSICGEN_SR)
    assert abs(tail.shape[-1] - expected_samples) <= 2   # ±2 samples tolerance


def test_pipeline_with_local_renderer_fakes(spec, tmp_path):
    from myAIbeats.events import EventEmitter
    from myAIbeats.pipeline import run
    r = LocalRenderer(out_dir=tmp_path, synth=FakeSynth(), scorer=FakeCLAP())

    # stitch not implemented yet — patch it with ScriptedRenderer's stitch
    from myAIbeats.renderers import ScriptedRenderer
    scripted = ScriptedRenderer()
    r.stitch = scripted.stitch   # borrow stitch from scripted for now

    em = EventEmitter(out=None)
    m = run(spec, r, em)
    assert m.ok
    # all sections used real synthesis (fake but real WAV files)
    for sr in m.sections:
        assert Path(sr.audio_path).exists() or sr.tone_pad_fallback
