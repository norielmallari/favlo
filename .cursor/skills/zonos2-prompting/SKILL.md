---
name: zonos2-prompting
description: ZONOS2 failed and is not served. Use when asked why it moved to pipelines/failed or how its emotion tags were parsed. Do not call generate_voice for it.
---

# ZONOS2 prompting

The pipeline is not discovered and `serve` does not load it. The note is [docs/zonos2/README.md](../../../docs/zonos2/README.md). Do not generate a clip.

Favlo speaks through `generate_voice`. The model id is `ZONOS2-fp8-fp4`. The listener contract is [favlo-server](../favlo-server/SKILL.md).

The spoken language is the text in `input`. There is no language argument. `instructions` is ignored. `voice` is one id from `list_voices` in `zonos2.voices`, or empty. Read that list from the workspace before choosing an id. An empty `voice` is the zero speaker. Do not invent an id. The voice id does not select a language.

A delivery tag is `[name]` or `[name:strength]` in `input`, before the words it covers. A later tag starts a new span. `name` is one id from `list_emotions` in `zonos2.emotion`. Read that list from the workspace before choosing an id. Those ids are the Zyphra directions: `happy`, `sad`, `angry`, `surprised`, `valence`, and `arousal`. The row's `sliders` name that direction. Do not invent an id. A whisper, a laugh, or a sigh is not an id. The tag is not spoken.

## What the prompt already fixes

These are not request fields. The pipeline inserts them on every line:

- Noisy background, token 517. The guide's `clean_speaker_background` stays false. The clean marker is not a language or voice control.
- Accurate mode, token 518, on every span. An emotion mix does not remove it.
- Trailing silence, bucket 3, token 511, on the last span.
- No speaking-rate token. The guide leaves speaking rate off unless asked.

`response_format` is `mp3`, `wav`, or `flac`.

## Language

Write the utterance in the language it should be spoken. The pipeline encodes each UTF-8 byte at row 192 plus that byte, between BOS and EOS. A Tagalog line such as `Magandang umaga. Kumusta ka na?` is those bytes and nothing else.

Zyphra's `language` field only picks a NeMo text normalizer. The codes are `en_us`, `en_gb`, `fr_fr`, `de`, `es`, `it`, `pt_br`, `ja`, `cmn`, and `ko`. Tagalog is not one of them. Tier 3 languages, and tier-2 languages outside that list, are triggered the same way: the text itself. Do not invent a code such as `tl`, `fil`, or `tagalog`.

Favlo does not run that normalizer. Write numbers, dates, and currency as words when the pronunciation matters.

Tiers name training coverage, not an input. Tier 1 is English, Mandarin Chinese, and Japanese. Tier 2 is Korean, Russian, Italian, Portuguese, French, Spanish, Vietnamese, German, Hebrew, and Dutch. Tier 3 is Swedish, Hindi, Tamil, Telugu, Thai, Norwegian, Bengali, Tagalog, Arabic, Danish, Indonesian, Polish, Ukrainian, Romanian, Finnish, Hungarian, Lithuanian, Estonian, Slovak, Croatian, and Latvian.

## Check

A prompt is wrong when a language code was added, when the language was placed in `instructions` or `voice`, when an emotion was placed in `instructions` or `voice`, or when the line is still in English after a non-English request. Compare registers with `harness/inspect_voice.py`. Its `f0_hz` is the fundamental. Do not add a language field to correct a draw. Confirm the byte path with `text_ids` from `zonos2.audio`: the ids are BOS `2`, then `192` plus each UTF-8 byte, then EOS `3`.

Sources: [Zyphra/ZONOS2](https://github.com/Zyphra/ZONOS2) README (`POST /tts/generate`) and the [SGLang-Omni ZONOS2 cookbook](https://sgl-project.github.io/sglang-omni/cookbook/zonos2.html). Both say the model infers the language from the text. The normalizer does not add a language condition.
