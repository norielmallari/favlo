---
name: music-prompting
description: Write an ACE-Step caption, lyrics, and metadata in Favlo, then check the clip with the inspect report. Use when changing a music prompt, lyrics, key, tempo, or meter, or when the user asks for better music. Do not use for generation-time optimization, image generation, or language-model prefill.
---

# Music prompting

Favlo does not run ACE-Step's caption LM. An empty `bpm`, `keyscale`, or `timesignature` is written as `N/A`. Set them on the request when the clip needs a key or a tempo.

Generation time is [optimization-music](../optimization-music/SKILL.md). Take the GPU lock the same way before a generate.

## Three fields

Caption, lyrics, and metadata do different jobs. Do not put BPM, a key name, or a tempo word in the caption. Those belong in metadata.

| Field | Carries | Leave out |
|---|---|---|
| Caption | Style, emotion, instruments, timbre, production | BPM, key, tempo |
| Lyrics | How the clip moves in time | A style essay |
| Metadata | `bpm`, `keyscale`, `timesignature` | Instruments and mood |

Caption is the main control. Name several dimensions. Specific words beat vague ones. Repeat an instrument you need. Do not combine styles that fight, such as a slow caption with a fast BPM.

`keyscale` is a note, an optional accidental, a space, then `major` or `minor`: `D major`, `F# minor`, `Bb major`. `timesignature` is `4/4`, `3/4`, or `6/8`. `bpm` is an integer from 30 to 300. These are anchors. A requested tempo can land near that number, not on it.

Stable keys in the ACE-Step notes are C, G, D, A minor, and E minor. Rare keys are ignored or shifted. Slow tempos sit around 60–80, mid tempos around 90–120, and fast tempos around 130–180. `4/4` is the reliable meter. `3/4` and `6/8` are usable. `5/4` and `7/8` are not.

## Lyrics

Put a blank line between sections. Keep a tag short. One qualifier is enough: `[Hook - synth lead]`. The tag's instrument must also be in the caption.

A clip with no singing uses `[Instrumental]`, or section tags and no lyric lines. Fit the number of sections to the duration. A 10 second clip holds two sections.

## Check

Run `uv run python harness/inspect_wav.py WAV --keyscale "<key>" --bpm <bpm>`. The report is the source of truth. Its `template_output` block is `seconds`, `key`, and `bpm`. Compare that with the request: the key must match, and the tempo may sit within 3% of the request. `requested_key miss`, a `key` margin near 0, a large `bpm_error`, `bpm none`, or a high `quiet_slices` share means the metadata did not hold. `clashes` counts half-seconds with two strong pitches one or two semitones apart. If either is high, change the metadata or the section tags. Do not put the key back into the caption. Pass `--reference` only to compare against a saved clip after a pipeline change. The music smoke prints `template_input` from the smoke file and fails when it disagrees with `template_output`.

Sources: ACE-Step 1.5 `docs/en/Tutorial.md`, `docs/en/INFERENCE.md`, and `acestep/constants.py`. The 3% tempo band is the six-interval relative just-noticeable difference in Drake and Botte, Perception & Psychophysics, 1993.
