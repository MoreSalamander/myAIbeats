# my-AI-beats — SPEC (SongSpec schema)

The authoring contract. A SongSpec is a JSON file validated by `spec.py`
before the pipeline starts. Structural failures are a hard error — an
authoring bug, never a model failure.

---

## Top level

```jsonc
{
  "song":     { ... },   // metadata + global style
  "sections": [ ... ],   // ordered musical sections
  "audio":    { ... },   // global mix settings
  "vocals":   { ... }    // optional non-blocking vocal layer
}
```

## `song`

| Field | Type | Required | Meaning |
|---|---|---|---|
| `title` | str | ✓ | song title |
| `genre` | str | ✓ | e.g. "indie folk", "trap", "ambient" |
| `tempo` | number | ✓ | BPM |
| `key` | str | ✓ | e.g. "G", "A minor", "F# major" |
| `mode` | str | | "major" / "minor" (default "major") |
| `time_signature` | str | | "4/4" (default) |
| `mood` | str | ✓ | emotional descriptor |
| `instruments` | [str] | | list of instruments to reference in prompts |
| `reference_feel` | str | | "sounds like X meets Y" — injected into all prompts |
| `continuation_secs` | number | | conditioning window (default 3.0) |

## `sections[]` — the heart

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | str | ✓ | stable id, e.g. "s01" |
| `type` | str | ✓ | "intro" / "verse" / "chorus" / "bridge" / "outro" / "break" |
| `bars` | number | ✓ | musical bar count; determines duration |
| `prompt` | str | ✓ | text prompt for this section (CLAP-verified) |
| `energy` | number | ✓ | 0.0–1.0; shapes prompt and synthesis intensity |
| `continuation` | bool | | true = use previous section as conditioning (default true except first) |

**Duration formula:** `(bars × 4 × 60) / tempo` seconds (4/4 time)

**Energy → prompt injection:** energy < 0.4 adds "minimal, sparse, quiet";
0.4–0.7 adds "medium energy, balanced"; > 0.7 adds "full, energetic, powerful".

**Invariants (validated at load):**
- sections non-empty, ids unique
- each `bars` > 0, `energy` in [0.0, 1.0]
- first section has `continuation: false` or omitted

## `audio`

| Field | Default | Meaning |
|---|---|---|
| `sample_rate` | 44100 | output Hz |
| `channels` | "stereo" | mono/stereo |
| `crossfade_s` | 1.5 | between-section crossfade |
| `normalize` | true | loudness-normalize to -14 LUFS |

## `vocals` (optional, non-blocking)

| Field | Default | Meaning |
|---|---|---|
| `enabled` | false | enable vocal layer |
| `engine` | "bark" | synthesis engine |
| `voice` | "v2/en_speaker_6" | Bark speaker preset |
| `lyrics` | {} | `{section_id: "lyric line"}` |

---

## Reference spec

`specs/something_blue.json` — a complete indie folk song demonstrating
the energy arc (intro 0.2 → verse 0.45 → chorus 0.85 → bridge 0.35 →
chorus 0.9 → outro 0.15) and musical continuation across all sections.
The reference fixture for Phase-1 offline tests.
