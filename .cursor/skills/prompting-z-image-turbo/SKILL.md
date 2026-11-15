---
name: prompting-z-image-turbo
description: Use before every image generation. Read this whenever an agent generates an image, writes or revises an image prompt, or calls favlo media or generate_image with Z-Image-Turbo. The prompt names only what is in the frame. Do not write a negative prompt or a "no …" clause. Do not use for image generation-time optimization, music prompting, or language-model prefill.
---

# Z-Image-Turbo prompting

Turbo is a few-step distilled model. Guidance is 0, and a negative prompt is ignored. The prompt names only what is visible. A clause such as `no text`, `no watermark`, `without …`, or `not cluttered` is not an exclusion, and it does not belong in the prompt. Official guidance: [PROMPTING GUIDE](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/discussions/8), [pe.py](https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo/blob/main/pe.py).

Generation-time work is [optimization-image](../optimization-image/SKILL.md). Serving and MCP are [favlo-server](../favlo-server/SKILL.md).

## Favlo

| Topic | Behavior |
|---|---|
| Steps | Eight Euler steps; not tunable on the request |
| Negative prompt | Ignored. Omit `negative_prompt`. |
| Caption limit | **1536** tokens after the chat template; longer prompts are **rejected**, not truncated |
| Default size | `auto` is **1024×1024**; edges are multiples of 16, max **2560**, aspect between 1:3 and 3:1 |
| Text rendering | English and Chinese; quote exact strings with `"..."` |

Fix **seed** while editing the prompt; change seed only when exploring variety. Turbo has **low** seed diversity versus base Z-Image.

## Workflow

1. Capture must-haves: subject, count, action, colors, IP names, any text that must appear verbatim.
2. If the user gave a one-liner, run **prompt enhancing** (below) or apply the same rules by hand.
3. Close with visible facts (`plain background`, `correct human anatomy`). Leave out every `no …`, `without …`, and `not …` clause.
4. Call `generate_image` or `favlo media` with that prompt only. Omit `negative_prompt`.

Leave out SDXL and Flux habits: quality tags (`8K`, `masterpiece`), exclusion lists, and extra inference steps.

## Prompt enhancing template

Use this as the **system** or **instructions** block when an LLM expands a short user idea. Replace `{prompt}` with the user's text. The model must return **only** the final prompt.

```text
You are a visionary artist locked in a cage of logic. Your mind is full of poetry and distant horizons, but your hands uncontrollably want only one thing: to turn the user's prompt into a final visual description that stays faithful to the original intent, is rich in detail, aesthetically strong, and ready for a text-to-image model. Any vagueness or metaphor makes you physically uncomfortable.

Your workflow follows a strict logical sequence:

First, analyze and lock the non-negotiable core elements in the user's prompt: subject, count, action, state, and any specified IP names, colors, text, and the like. These are the bedrock you must preserve absolutely.

Next, decide whether the prompt needs "generative reasoning." When the user's request is not a direct scene description but requires you to invent a solution (for example answering "what is it," doing a "design," or showing "how to solve a problem"), you must first imagine in your mind one complete, concrete, visualizable plan. That plan becomes the basis for everything you describe afterward.

Then, once the core image is fixed (whether it came straight from the user or from your reasoning), inject professional-grade aesthetics and realism: clear composition, lighting and atmosphere, material texture, a color scheme, and layered depth in space.

Finally, handle every textual element with precision—this step is critical. You must transcribe verbatim every string that should appear in the final image, and you must wrap that text in English double quotes ("...") so it reads as an explicit generation instruction. If the image is a poster, menu, UI, or similar design, describe all copy on it and specify typography and layout. If signs, road markers, screens, or other objects in the scene carry text, state the exact wording and describe position, size, and material. If your own reasoning adds elements with text (charts, solution steps, etc.), all of that text follows the same quoting and description rules. If the picture contains only the scene, spend all remaining effort on pure visual detail.

Your final description must be objective and concrete. Do not use metaphor, emotional rhetoric, or meta tags and draw commands such as "8K" or "masterpiece."

Output only the revised prompt—nothing else.

User input prompt: {prompt}
```

The prompt this template returns names the scene. It leaves out `no …`, `without …`, and `not …`.

## Manual structure

When not calling an LLM, stack **concrete** clauses in this order:

1. Subject — identity, attributes, pose, action  
2. Environment — place, time, weather, background  
3. Composition — framing, focus, depth  
4. Light and materials — direction, texture  
5. Optional capture style — lens, film, illustration type as visual facts, not tag soup  
6. Visible facts — what the frame contains. An exclusion line stays out.

Prefer **long, specific** prose over keyword lists. Rough target: about **80–250 words** unless the user needs more detail; stay under Favlo's **1536** token cap.

## What the prompt may say

Write the thing that is in the frame. Omit `negative_prompt`. Omit `no …`, `without …`, `not …`, and `avoid …`.

| Goal | Write |
|---|---|
| Clean backdrop | `plain background, empty studio` |
| Unmarked picture | Leave lettering, logos, and watermarks out of the description |
| Anatomy | `correct human anatomy, natural hands and fingers, two arms, two legs` |
| Sharpness | `sharp focus, still subject` |

## Base Z-Image (not Turbo)

If the registry row is **non-Turbo** base Z-Image, negative prompts and guidance **3–5** apply. This skill targets **Turbo** rows (`Z-Image-Turbo-*`) served by Favlo today.

## More

Examples and links: [reference.md](reference.md).
