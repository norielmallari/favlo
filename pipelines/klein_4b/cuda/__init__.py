"""CUDA runtime for FLUX.2 [klein] 4B."""

from klein_4b.cuda.pipeline import pipeline

handle_image = pipeline


def main(argv: list[str] | None = None) -> int:
    return pipeline.main([] if argv is None else argv)
