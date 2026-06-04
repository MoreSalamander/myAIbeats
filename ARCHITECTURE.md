# my-AI-beats — ARCHITECTURE

How the doctrine in `CONSTITUTION.md` is realized.

---

## The pipeline

```
spec_load
  → per section:
      section_synth  →  section_verify (CLAP ●)
                     →  duration_verify (bars ●)
                     →  retry ↺  →  tone_pad fallback
  → stitch (ffmpeg crossfade)  →  stitch_verify (●)
  → vocal_synth (optional)     →  vocal_verify (○ non-blocking)
  → master_mix  →  song.wav
  → done
```

Blocking gates ●: section quality (CLAP), section duration, final stitch.
Non-blocking gates ○: vocal synthesis and alignment.

## The hidden gem — musical continuation

After the first section is synthesized, every subsequent section is
generated using MusicGen's `audio_prompt_inputs`:

```python
# Take the last CONTINUATION_SECS of the previous section
conditioning = previous_audio[-CONTINUATION_SECS * sample_rate:]
# Pass as conditioning — MusicGen continues from this audio
audio = model.generate(inputs, audio_prompt_inputs=conditioning)
```

The result is musical coherence across sections — each one grows
organically from the last rather than starting from silence. This is
what makes the output sound like a song.

CONTINUATION_SECS = 3.0 (configurable in SongSpec)

## SongSpec → duration mapping

```
duration_s = (bars × beats_per_bar × 60) / tempo
           = (bars × 4 × 60) / tempo   [for 4/4 time]

MusicGen tokens: max_new_tokens = int(duration_s × 50) + 32
```

Example: 8 bars at 90 BPM = 21.3s → 1097 tokens

## Modules

| Module | Responsibility |
|---|---|
| `spec.py` | SongSpec / Section dataclasses + JSON loader + structural validation |
| `verifiers.py` | `section_verify` (CLAP), `duration_verify`, `stitch_verify`, `vocal_verify` |
| `renderers.py` | `BeatRenderer` protocol + `ScriptedRenderer` offline fake |
| `pipeline.py` | named-stage orchestrator: retry, fallback, continuation, events |
| `local.py` | `LocalRenderer`: MusicGen, Stable Audio, Bark vocals |
| `events.py` | NDJSON emitter (shared studio vocabulary) |
| `cli.py` | `--spec`, `--dry-run`, `--generate`, `--limit` |
| `api.py` | FastAPI + NDJSON stream + web UI |

## HuggingFace local-free stack (M4 Pro / 24 GB / MPS)

| Stage | Model | License | Notes |
|---|---|---|---|
| Section synth | `facebook/musicgen-stereo-medium` | CC-BY-NC | Stereo, good quality; already in venv |
| Continuation | same model, `audio_prompt_inputs` | — | The hidden gem |
| Long sections | `stabilityai/stable-audio-open-1.0` | Community | ≤47s, 44.1kHz stereo |
| Verify: content | `laion/clap-htsat-unfused` | Apache-2.0 | Audio ↔ text cosine score |
| Vocals (opt.) | `suno/bark` | MIT | Rough but free; non-blocking |
| Stitch + master | **ffmpeg** | — | acrossfade, normalize, loudness |

## Verification thresholds

| Gate | Metric | Default | Class |
|---|---|---|---|
| `section_verify` | CLAP cosine(prompt, audio) | ≥ 0.20 | blocking |
| `duration_verify` | actual_s vs declared_s | ±2.0s tol | blocking |
| `stitch_verify` | ffprobe: exists, dur ≈ Σ, stereo | dur tol 2.0s | blocking |
| `vocal_verify` | audio exists + duration > 0 | — | non-blocking |

## Audio globals

Sample rate: 32,000 Hz (MusicGen native) upsampled to 44,100 Hz for
final master. Channels: stereo throughout. Crossfade: 1.5s between
sections (ffmpeg acrossfade). Normalization: loudness-normalize final
mix to -14 LUFS (streaming standard).
