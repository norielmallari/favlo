---
name: klein-4b-prompting
description: >-
  Write FLUX.2 [klein] 4B prompts for text-to-image, a one-image edit, and a
  composite of up to four references. Use when writing or improving a Klein
  prompt, an edit instruction, image roles, or a multi-image composite. Do not
  use for generation-time optimization or another image model.
---

# Klein 4B prompting

Klein 4B has no prompt upsampler. The words you send are the words it encodes. There is no negative prompt and no guidance scale. The default is 4 steps. Fix the seed while you revise the wording, and change the seed only when you want a different sample.

Write prose. Do not send a keyword list, a JSON object, or quality tags (`8K`, `masterpiece`). Put the subject first. Word order is the priority order: subject, action, style, setting, then light and other detail.

The chat wrap counts toward a 512-token budget. Tokens past that are cut from the end, so a constraint buried after a long prompt never arrives. A generation prompt is about 40–70 words. An edit is shorter. A composite grows only by naming each image's role and how the pieces meet.

English is the precise default. Other languages are accepted.

## Request

`generate_image` takes `prompt`. Omit `image`. An omitted `size` is 1024×1024.

`edit_image` takes `prompt` and `image`. `image` is one still or a list of one to four. List order is the prompt's order: the first entry is image 1. Call `generate_image` when there is no still. Do not invent a blank image.

`size` is the output canvas, `WxH`. Both edges are multiples of 16, from 16 to 2048. An omitted edit size follows image 1 after it is fitted, so put the base or the setting first when you omit `size`. A reference whose long side is above 1024 shrinks until that side is 1024, aspect kept, then both edges snap to a multiple of 16. A smaller reference is only center-cropped to that grid. References are not stretched onto the canvas. Each one keeps its own grid.

Leave `steps` at 4 unless the user asks. A negative prompt and a guidance value are ignored.

## Text to image

Describe the scene the way a novelist would, in this order:

1. Subject, with the attributes that must survive.
2. Action or pose.
3. Medium or style, as a visual fact.
4. Place, time, and weather.
5. Light: source, direction, and softness.

Close with one line: `Style: <style>. Mood: <mood>.`

Name a camera, a lens, or a film stock only when the look depends on it. Hex colors use the word `color` or `hex` and `#RRGGBB`. Text that must appear is in double quotes, with placement and lettering.

```text
A tan paper horse mid-gallop on a white field, a red leather saddle
strapped across its back, a soft gray shadow under the hooves, even
studio light from the upper left. Style: flat illustration. Mood: quiet.
```

## One image

One still is image 1. The prompt is an edit, not a fresh scene description. Use three parts:

1. Name the target so only one thing matches. Use position, size, color, or what it is next to.
2. Say what it becomes. For a removal, name the object and ask for it to be gone.
3. Name what stays: identity, pose, framing, wardrobe, lettering, or the background layout.

One visible change per prompt when identity matters. A second change gets its own sentence with its own place. Do not write "preserve everything."

```text
Add a red leather saddle across the horse's back. Keep the pose,
the white background, and the framing.
```

```text
Change the saddle leather to color #8B1E3F. Keep the stitching,
the pose, and the background.
```

```text
Replace the shop sign with a red neon sign that says "Night Bloom".
Keep the sign's place, the street, and the night lighting.
```

## Composite

Two, three, or four stills. Give each image one job, then say how the pieces meet: who touches what, where they stand, and which light wins. Number them `image 1` through `image 4` in list order, and use those words every time.

| Role | What it supplies | Phrase |
|---|---|---|
| Subject | The person, animal, or character to keep | `the horse from image 1` |
| Product | An object whose shape, color, and marks stay | `the saddle from image 2, facing the camera` |
| Setting | The place the others enter | `the meadow in image 3` |
| Style | Palette, medium, or light to apply | `in the style of image 4` |
| Pose | Body position to copy | `in the pose from image 2` |

```text
The horse from image 1 stands in the meadow from image 2, wearing
the saddle from image 3. It faces left, hooves on the grass, and
the meadow's sunlight falls across the saddle. Keep the horse's
colors and the meadow's horizon.
```

A vague "combine these images" is not a prompt. Keep the same image order on the next attempt. If image 1 was the subject, do not silently make image 2 the subject.

When the subject should interact with the setting, name the contact and the shared light. "Standing in front of the barn" is placement. "Rearing, front hooves against the barn door, dust kicked up in the barn's shade" is an interaction.

## Checklist

- Subject or edit target is the first clause.
- Light has a source and a direction when the image is new or restyled.
- Each still has one role, and the numbers match the list.
- Quoted text is exact. A brand color is `#RRGGBB`.
- What must survive is named. There is no negative field to carry it.
- The prompt is short enough that the last sentence is inside the 512-token wrap.

Sources: Black Forest Labs, [FLUX.2 model prompting](https://github.com/black-forest-labs/skills/blob/master/skills/flux-image-best-practices/rules/flux2-models.md) (Klein has no upsampler, novelist prose, subject first, 40–70 words, four references), [prompt reference](https://docs.bfl.ai/guides/prompting_unified_reference) (`Style` / `Mood`, light, `input_image` through `input_image_4`), [single-reference editing](https://docs.bfl.ai/guides/prompting_editing_single_reference) (name the target, the change, and what stays; hex and quoted text), and [multi-reference editing](https://github.com/black-forest-labs/skills/blob/master/skills/flux-image-best-practices/rules/multi-reference-editing.md) (one role per image, an explicit relationship). This package does not accept box rows.
