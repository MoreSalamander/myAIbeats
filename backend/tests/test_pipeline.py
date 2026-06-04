from pathlib import Path
import pytest
from myAIbeats.events import EventEmitter
from myAIbeats.pipeline import PipelineError, run
from myAIbeats.renderers import ScriptedRenderer
from myAIbeats.spec import load_spec

SPEC_PATH = Path(__file__).resolve().parents[2] / "specs" / "something_blue.json"

@pytest.fixture
def spec():
    return load_spec(SPEC_PATH)

def quiet():
    return EventEmitter(out=None)

def test_full_pipeline_offline_passes(spec):
    m = run(spec, ScriptedRenderer(), quiet())
    assert m.ok
    assert len(m.sections) == 8
    assert m.stitch_gate.passed
    s = m.summary()
    assert s["gates_failed"] == 0
    assert s["tone_pad_fallbacks"] == 0

def test_continuation_threads_through_sections(spec):
    m = run(spec, ScriptedRenderer(), quiet())
    # first section never uses continuation
    assert not m.sections[0].used_continuation
    # all subsequent sections do
    for sr in m.sections[1:]:
        assert sr.used_continuation

def test_clap_failure_retries_then_falls_back(spec):
    r = ScriptedRenderer(fail_sections={"s03"}, heal_after=99)
    m = run(spec, r, quiet(), max_retries=1)
    assert m.ok
    s03 = next(s for s in m.sections if s.section_id == "s03")
    assert s03.tone_pad_fallback

def test_clap_failure_heals_on_retry(spec):
    r = ScriptedRenderer(fail_sections={"s03"}, heal_after=1)
    m = run(spec, r, quiet(), max_retries=2)
    assert m.ok
    s03 = next(s for s in m.sections if s.section_id == "s03")
    assert not s03.tone_pad_fallback

def test_vocal_drop_is_non_blocking(spec):
    import json
    raw = json.loads(Path(SPEC_PATH).read_text())
    raw["vocals"]["enabled"] = True
    raw["vocals"]["lyrics"] = {"s02": "something blue in the morning light"}
    from myAIbeats.spec import parse_spec
    spec_v = parse_spec(raw)
    r = ScriptedRenderer(fail_vocal={"s02"})
    m = run(spec_v, r, quiet())
    assert m.ok   # vocal drop never fails the song
    assert m.summary()["vocals_dropped"] == 1

def test_energy_arc_in_manifest(spec):
    m = run(spec, ScriptedRenderer(), quiet())
    energies = [spec.sections[i].energy for i in range(len(m.sections))]
    assert energies[0] < energies[2]   # intro < chorus
    assert energies[-1] < energies[-2]  # outro < final chorus

def test_event_stream_vocabulary(spec):
    em = EventEmitter(out=None)
    run(spec, ScriptedRenderer(), em)
    names = {e["event"] for e in em.collected}
    assert {"step_start", "step_complete", "gate_pass", "done"} <= names
    assert em.collected[-1]["event"] == "done"

def test_stitch_verify_blocking_on_missing_file(spec):
    class BrokenStitch(ScriptedRenderer):
        def stitch(self, paths, spec, out_path):
            out = super().stitch(paths, spec, out_path)
            out.exists = False
            return out
    with pytest.raises(PipelineError, match="stitch_verify"):
        run(spec, BrokenStitch(), quiet())
