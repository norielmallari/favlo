"""Favlo image generation provider."""

from __future__ import annotations

import logging
import os
from typing import Any

from agent.image_gen_provider import DEFAULT_ASPECT_RATIO, resolve_aspect_ratio, success_response
from plugins.image_gen._common import (
    StaticImageGenProvider,
    error_factory,
    materialize_image,
    prompt_required_error,
)

logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8129/v1"
API_PATH = "/images/generations"

# Edges stay inside the pipeline cap of 1280 and on a multiple of 16.
SIZES = {"landscape": "1280x768", "square": "1024x1024", "portrait": "768x1280"}

# The endpoint's model ids are case-sensitive; the payload must carry the server's
# exact id. `api_model` maps a picker id to the id the endpoint actually accepts.
MODELS = {
    "z-image-turbo-fp4": {
        "display": "Z-Image Turbo",
        "speed": "~100s",
        "strengths": "High-fidelity, local generation",
        "api_model": "Z-Image-Turbo-fp4",
    },
}
DEFAULT_MODEL = "z-image-turbo-fp4"


class FavloImageGenProvider(StaticImageGenProvider):
    """Image generation against a local OpenAI-compatible endpoint at localhost:8129."""

    provider_id = "favlo_image_gen"
    label = "Favlo"
    models = MODELS
    default_model_id = DEFAULT_MODEL

    @property
    def display_name(self) -> str:
        return os.environ.get("FAVLO_LABEL", self.label)

    def is_available(self) -> bool:
        return True

    def get_setup_schema(self) -> dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "local",
            "tag": "Z-Image Turbo on localhost:8129",
            "env_vars": [],
        }

    def capabilities(self) -> dict[str, Any]:
        return {"modalities": ["text"], "max_reference_images": 0}

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        *,
        image_url: str | None = None,
        reference_image_urls: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del image_url, reference_image_urls
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)
        if not prompt:
            return prompt_required_error(self.name, aspect)

        model_id = kwargs.get("model_id") or kwargs.get("model") or DEFAULT_MODEL
        api_model = MODELS.get(model_id, {}).get("api_model", model_id)
        api_key = os.environ.get("OPENAI_API_KEY", "")
        size = SIZES.get(aspect, SIZES["square"])
        fail = error_factory(self.name, aspect, model=model_id, prompt=prompt)

        payload: dict[str, Any] = {
            "model": api_model,
            "prompt": prompt,
            "size": size,
            "n": 1,
            "response_format": "b64_json",
        }

        try:
            import httpx

            headers: dict[str, str] = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            url = BASE_URL.rstrip("/") + API_PATH
            with httpx.Client(timeout=httpx.Timeout(300.0)) as client:
                resp = client.post(url, json=payload, headers=headers)

            if resp.status_code != 200:
                detail = resp.text[:500]
                return fail(
                    f"Favlo image generation failed ({resp.status_code}): {detail}",
                    "api_error",
                )

            result_data = resp.json().get("data", [])
            if not result_data:
                return fail("Favlo returned no images", "empty_response")

            item = result_data[0]
            image_path, err = materialize_image(
                item.get("b64_json"),
                item.get("url"),
                prefix=model_id,
                label="Favlo",
                provider=self.name,
                model=model_id,
                prompt=prompt,
                aspect=aspect,
                log=logger,
            )
            if err:
                return err
            return success_response(
                image=image_path,
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
                provider=self.name,
                extra={"size": size},
            )
        except Exception as exc:
            logger.debug("Favlo generation failed", exc_info=True)
            return fail(f"Favlo connection failed: {exc}", "connection_error")


def register(ctx) -> None:
    ctx.register_image_gen_provider(FavloImageGenProvider())
