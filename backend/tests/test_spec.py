import json
from pathlib import Path
import pytest
from myAIbeats.spec import SpecError, load_spec, parse_spec

SPEC_PATH = Path(__file__).resolve().parents[2] / "specs" / "something_blue.json"

@pytest.fixture
def raw():
    return json.loads(SPEC_PATH.read_text())

def test_reference_spec_loads():
    spec = load_spec(SPEC_PATH)
    assert spec.song.title == "Something Blue"
    assert len(spec.sections) == 8
    assert spec.sections[0].continuation == False   # first section never continues
    assert spec.sections[1].continuation == True
    assert spec.song.tempo == 88

def test_energy_arc():
    spec = load_spec(SPEC_PATH)
    energies = [s.energy for s in spec.sections]
    # intro low, chorus high, outro low
    assert energies[0] < 0.3        # intro
    assert energies[2] > 0.8        # first chorus
    assert energies[-1] < 0.3       # outro

def test_full_prompt_injects_energy_tag():
    spec = load_spec(SPEC_PATH)
    # intro energy=0.2 → "minimal, sparse, quiet"
    assert "minimal" in spec.sections[0].full_prompt
    # chorus energy=0.85 → "full, energetic, powerful"
    assert "energetic" in spec.sections[2].full_prompt

def test_duration_at_tempo():
    spec = load_spec(SPEC_PATH)
    # s01: 8 bars at 88 BPM = 8*4*60/88 ≈ 21.8s
    assert abs(spec.sections[0].duration_at_tempo(88) - 21.818) < 0.01

def test_missing_required_field(raw):
    del raw["song"]["tempo"]
    with pytest.raises(SpecError, match="tempo"):
        parse_spec(raw)

def test_energy_out_of_range(raw):
    raw["sections"][0]["energy"] = 1.5
    with pytest.raises(SpecError, match="energy"):
        parse_spec(raw)

def test_duplicate_ids_rejected(raw):
    raw["sections"][1]["id"] = raw["sections"][0]["id"]
    with pytest.raises(SpecError, match="unique"):
        parse_spec(raw)

def test_first_section_continuation_auto_corrected(raw):
    raw["sections"][0]["continuation"] = True
    spec = parse_spec(raw)
    assert spec.sections[0].continuation == False
