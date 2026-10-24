---
name: inspect-voice
description: Measure the speech fundamental of a WAV and the semitone gap to a reference clip. Use when checking whether two voice clips are the same speaker, or when a pitch line from the music inspector looks like a harmonic. Do not use for music key, tempo, or harmony.
---

# Inspect voice

You cannot hear audio. Run the script, then read the printed fundamental.

```bash
uv run python harness/inspect_voice.py WAV --reference other.wav
```

`--reference` is optional. Omit it when the clip has no comparison.

`f0_hz` is the median fundamental from YIN (de Cheveigné and Kawahara, 2002). The period is the shortest lag whose cumulative mean normalized difference is under 0.1. `f0_p10_hz` and `f0_p90_hz` are that distribution. `voiced` is the number of frames that had a dip. `f0_semitones` is how far `f0_hz` sits from the reference fundamental. A gap near 12 is one octave.

`harness/inspect_wav.py` reports the strongest partial between 70 Hz and 800 Hz. That partial can be a harmonic. Use this script for the speaking voice.
