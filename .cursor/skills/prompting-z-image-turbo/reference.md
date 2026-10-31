# Z-Image-Turbo prompting reference

## Official sources

| Resource | URL |
|---|---|
| Model card | https://huggingface.co/Tongyi-MAI/Z-Image-Turbo |
| Prompting discussion | https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/discussions/8 |
| PE template (Chinese) | https://huggingface.co/spaces/Tongyi-MAI/Z-Image-Turbo/blob/main/pe.py |
| Diffusers pipeline | https://huggingface.co/docs/diffusers/main/api/pipelines/z_image |
| Favlo pipeline note | docs/z-image-turbo/README.md |

Authors recommend long prompts, optional LLM enhancement with `pe.py`, and `max_sequence_length=1024` in Diffusers when captions exceed the 512-token demo default. Favlo's caption encoder allows **1536** tokens.

Turbo inference defaults from the model card: **`guidance_scale=0`**, **`num_inference_steps=9`** (eight DiT evaluations). Favlo fixes eight steps internally.

## Turbo vs base

| | Z-Image-Turbo | Z-Image (base) |
|---|---|---|
| Steps | 8 | 28–50 |
| Guidance | 0 | 3–5 |
| Negative prompt | Inert | Recommended |
| Seed diversity | Low | Medium |

## Example expansion

**User:** `a red cube on a white table`

**Enhanced (illustrative):** A single solid red cube with matte paint sits centered on a rectangular white table with a smooth laminate surface. The table stands in an empty studio room with a seamless off-white floor and wall. Soft daylight from a large window on the left casts a gentle shadow to the right of the cube. Eye-level three-quarter view, shallow depth of field, sharp focus on the cube, plain background, no text, no watermark.

Smoke and harness tests in this repo use the short cube prompt at fixed seeds; production prompts are usually longer.

## Photorealism hints

Community guides aligned with official examples often add:

- Capture device or film stock as a **visual** fact (`shot on a 35mm lens`, `Kodak Portra color`)  
- Plausible anatomy and asymmetric facial detail for portraits  
- Named materials (`brushed aluminum`, `worn leather`)  

Avoid metaphor (`like a dream`) and mood-only words without visual anchors.

## In-image text

- Copy strings **exactly**; wrap in `"..."`.  
- Posters, menus, UI: describe layout, type style, and every line of copy.  
- Supported rendering languages: **English** and **Chinese**.
