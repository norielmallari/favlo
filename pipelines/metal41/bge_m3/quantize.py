"""Pack the float32 checkout into one W8A8 Neural Engine program.

``embed`` does not import this module. The float16 graphs are temporaries.
Each finished length stays in a partial directory, so a later invocation
continues. A finished directory is registered and returned without reading
the weights again.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from bge_m3.passages import passages
from bge_m3.text import (
    HIDDEN,
    bucket_for,
    key_mask,
    l2_normalize,
    lengths_from_names,
    pad_ids,
    position_ids,
)

RECORD = "bge-m3-w8a8"
_LENGTHS = (128, 256, 512, 1024, 2048, 4096)
# coremltools leaves the nn.Embedding index path on the CPU. A projection uses
# another op name, and the pack fails if one of those leaves the Neural Engine.
# Bump the stamp when the graph or the quantizer changes so a partial pack is rebuilt.
_INDEX_OPS = frozenset(
    {
        "ios18.greater_equal",
        "ios18.add",
        "select",
        "ios18.gather",
        "ios18.cast",
    }
)
_PACK_STAMP = "w8a8-prefix-minmax"
_COSINE_MIN = 0.96


def quantize(source: Path, dest: Path) -> None:
    """Write ``dest`` from ``source`` and register the folder when it is complete."""
    if _finished(dest):
        _register(dest, source)
        shutil.rmtree(_cache_dir(dest), ignore_errors=True)
        return
    cache = _partial(dest)
    if _cached_lengths(cache):
        _publish(source, dest, cache)
    dest.mkdir(parents=True, exist_ok=True)
    _build(source, dest, cache)
    _register(dest, source)
    shutil.rmtree(cache, ignore_errors=True)


def _cache_dir(dest: Path) -> Path:
    return dest.parent / f".{dest.name}.partial"


def _partial(dest: Path) -> Path:
    """Lengths already quantized. A stamp mismatch deletes them."""
    cache = _cache_dir(dest)
    stamp = cache / "stamp"
    if not stamp.is_file() or stamp.read_text(encoding="utf-8") != _PACK_STAMP:
        shutil.rmtree(cache, ignore_errors=True)
        cache.mkdir(parents=True)
        stamp.write_text(_PACK_STAMP, encoding="utf-8")
        return cache
    for saving in list(cache.glob("*.saving")) + list(cache.glob("*.saving.mlpackage")):
        if saving.is_dir():
            shutil.rmtree(saving, ignore_errors=True)
        else:
            saving.unlink(missing_ok=True)
    for package in cache.glob("*.mlpackage"):
        if (package / "Manifest.json").is_file():
            continue
        shutil.rmtree(package, ignore_errors=True)
        (cache / f"{package.stem}.cosine.json").unlink(missing_ok=True)
    return cache


def _build(source: Path, dest: Path, cache: Path) -> None:
    from tokenizers import Tokenizer

    from bge_m3.model import load_encoder, load_sparse

    _require_tools()
    encoder = load_encoder(source)
    weight, bias = load_sparse(source)
    tokenizer_path = source / "tokenizer.json"
    if not tokenizer_path.is_file():
        message = "tokenizer.json is not in the checkout"
        raise FileNotFoundError(message)
    shutil.copyfile(tokenizer_path, dest / "tokenizer.json")
    np.savez(dest / "sparse.npz", weight=weight.numpy(), bias=float(bias.reshape(())))
    tokenizer = Tokenizer.from_file(str(dest / "tokenizer.json"))
    functions: list[tuple[str, Path]] = []
    recorded: dict[str, dict[str, float]] = {}
    for length in _LENGTHS:
        name, _model = _export_cached(encoder, tokenizer, length, cache)
        functions.append((name, cache / f"{name}.mlpackage"))
        try:
            recorded[str(length)] = _cosine_gate(encoder, tokenizer, cache, name, length)
        except Exception:
            _discard_length(cache, name)
            raise
    package = dest / "model.mlpackage"
    _save_functions(functions, package)
    for name, _model in functions:
        _assert_engine_function(package, name)
    cosines = _cosine(encoder, package, tokenizer)
    _require_cosine(cosines, package.name)
    floor = min(item["min"] for item in recorded.values())
    _write_json(
        dest / "cosine.json",
        {
            "passages": 128,
            "min": floor,
            "mean": sum(item["mean"] for item in recorded.values()) / len(recorded),
            "lengths": recorded,
        },
    )
    _write_json(
        dest / "manifest.json",
        {
            "complete": True,
            "quant": "w8a8",
            "hidden": HIDDEN,
            "functions": [name for name, _model in functions],
        },
    )


def _export_cached(
    encoder: Any,
    tokenizer: Any,
    length: int,
    cache: Path,
) -> tuple[str, Any]:
    """Reuse a length saved by an earlier run. The stamp owns invalidation."""
    import coremltools as ct

    name = f"encode_{length}"
    path = cache / f"{name}.mlpackage"
    if (path / "Manifest.json").is_file():
        try:
            model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
            _assert_engine_model(model, name)
        except Exception:
            print(f"rebuilding {name}", flush=True)
            _discard_length(cache, name)
        else:
            print(f"reused {name}", flush=True)
            return name, model
    name, model = _export_one(encoder, tokenizer, length)
    _publish_package(model, path)
    return name, model


def _cosine_gate(
    encoder: Any,
    tokenizer: Any,
    cache: Path,
    name: str,
    length: int,
) -> dict[str, float]:
    """Score this length before the next one starts. A saved passing score is reused."""
    path = cache / f"{name}.cosine.json"
    scores = _read_cosine(path)
    if scores is None:
        scores = _cosine(encoder, cache / f"{name}.mlpackage", tokenizer, fixed_length=length)
        _require_cosine(scores, name)
        _write_json(path, scores)
    _report_cosine(scores)
    return scores


def _discard_length(cache: Path, name: str) -> None:
    for suffix in (".mlpackage", ".saving.mlpackage", ".cosine.json"):
        path = cache / f"{name}{suffix}"
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.is_file():
            path.unlink()


def _publish_package(model: Any, path: Path) -> None:
    saving = path.with_name(f"{path.stem}.saving.mlpackage")
    shutil.rmtree(saving, ignore_errors=True)
    try:
        model.save(str(saving))
        if not (saving / "Manifest.json").is_file():
            message = f"{path.name} save did not finish"
            raise RuntimeError(message)
        if path.exists():
            shutil.rmtree(path)
        saving.rename(path)
    except Exception:
        shutil.rmtree(saving, ignore_errors=True)
        raise


def _write_json(path: Path, body: dict[str, Any]) -> None:
    saving = path.with_suffix(path.suffix + ".saving")
    saving.write_text(json.dumps(body), encoding="utf-8")
    saving.replace(path)


def _export_one(encoder: Any, tokenizer: Any, length: int) -> tuple[str, Any]:
    import coremltools as ct
    import torch

    name = f"encode_{length}"
    print(f"exporting {name}", flush=True)
    ids = torch.zeros((1, length), dtype=torch.int32)
    positions = torch.arange(length, dtype=torch.int32).view(1, length)
    mask = torch.zeros((1, 1, length), dtype=torch.float16)
    traced = torch.jit.trace(encoder, (ids, positions, mask))
    model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, length), dtype=np.int32),
            ct.TensorType(name="position_ids", shape=(1, length), dtype=np.int32),
            ct.TensorType(name="key_mask", shape=(1, 1, length), dtype=np.float16),
        ],
        outputs=[ct.TensorType(name="hidden"), ct.TensorType(name="colbert")],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    _assert_engine_model(model, name)
    model = _w8a8(model, _samples(tokenizer, length), _group_size(length))
    _assert_engine_model(model, name)
    print(f"quantized {name}", flush=True)
    return name, model


def _group_size(length: int) -> int:
    """How many intermediate ops one calibration model returns.

    Short buckets fit in one model. Longer buckets split the outputs so the
    temporary model does not hold every activation at once.
    """
    if length >= 2048:
        return 200
    if length >= 1024:
        return 400
    return -1


def _w8a8(model: Any, samples: list[dict[str, np.ndarray]], group_size: int) -> Any:
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
    )

    activations = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(mode="linear_symmetric"),
        op_type_configs={"add": None},
    )
    model = _quantize_activations(model, activations, samples, group_size)
    weights = OptimizationConfig(
        global_config=OpLinearQuantizerConfig(
            mode="linear_symmetric",
            granularity="per_channel",
        )
    )
    return linear_quantize_weights(model, weights)


def _quantize_activations(
    model: Any,
    config: Any,
    samples: list[dict[str, np.ndarray]],
    group_size: int,
) -> Any:
    """Quantize conv inputs over the observed range. Conv outputs stay float16."""
    import coremltools as ct
    from coremltools import _SPECIFICATION_VERSION_IOS_17
    from coremltools.converters.mil.frontend.milproto import load as milproto_load
    from coremltools.converters.mil.mil.passes.graph_pass import PassOption
    from coremltools.converters.mil.mil.passes.pass_registry import PASS_REGISTRY
    from coremltools.models import utils as model_utils
    from coremltools.optimize.coreml.experimental._model_debugger import ModelDebugger
    from coremltools.optimize.coreml.experimental._post_training_quantization import (
        _get_activation_calibration_stats,
    )

    original_model = ct.models.MLModel
    original_record = ModelDebugger.record_intermediate_output
    compiled: dict[tuple[str, ...], Any] = {}

    def cached(*args: Any, **kwargs: Any) -> Any:
        spec = args[0] if args else None
        outputs = getattr(getattr(spec, "description", None), "output", None)
        if outputs is None:
            return original_model(*args, **kwargs)
        names = tuple(item.name for item in outputs)
        hit = compiled.get(names)
        if hit is not None:
            return hit
        kwargs["compute_units"] = ct.ComputeUnit.CPU_AND_NE
        loaded = original_model(*args, **kwargs)
        compiled[names] = loaded
        return loaded

    def record(
        self: Any,
        output_value: Any,
        output_name: str,
        stats: dict[str, dict[str, float]],
    ) -> None:
        del self
        value = np.asarray(output_value, dtype=np.float32)
        low = float(np.min(value))
        high = float(np.max(value))
        previous = stats.get(output_name)
        if previous is not None and "rmin" in previous:
            low = min(low, float(previous["rmin"]))
            high = max(high, float(previous["rmax"]))
        stats[output_name] = {"rmin": low, "rmax": high}

    ct.models.MLModel = cached
    ModelDebugger.record_intermediate_output = record
    try:
        activation_stats = _get_activation_calibration_stats(model, samples, group_size)
    finally:
        ct.models.MLModel = original_model
        ModelDebugger.record_intermediate_output = original_record
    prefix = PASS_REGISTRY["compression::insert_prefix_quantize_dequantize_pair"]
    prefix.set_options(
        [PassOption("config", config), PassOption("activation_stats", activation_stats)]
    )
    return model_utils._apply_graph_pass(
        model,
        [prefix, PASS_REGISTRY["common::dequantize_quantize_pair_elimination"]],
        spec_version=_SPECIFICATION_VERSION_IOS_17,
        pymil_load_func=milproto_load.load,
        skip_model_load=model.__proxy__ is None,
    )


def _samples(tokenizer: Any, length: int) -> list[dict[str, np.ndarray]]:
    rows: list[list[int]] = []
    for text in passages():
        ids = list(tokenizer.encode(text).ids)
        if len(ids) > length:
            ids = ids[:length]
        rows.append(pad_ids(ids, length))
    if len(rows) != 128:
        message = "calibration is not 128 passages"
        raise RuntimeError(message)
    samples: list[dict[str, np.ndarray]] = []
    for row in rows:
        samples.append(
            {
                "input_ids": np.asarray([row], dtype=np.int32),
                "position_ids": np.asarray([position_ids(row)], dtype=np.int32),
                "key_mask": key_mask(row),
            }
        )
    return samples


def _cached_lengths(cache: Path) -> list[int]:
    """Lengths in the partial cache whose cosine cleared the gate."""
    found: list[int] = []
    for length in _LENGTHS:
        name = f"encode_{length}"
        package = cache / f"{name}.mlpackage"
        if not (package / "Manifest.json").is_file():
            continue
        if _read_cosine(cache / f"{name}.cosine.json") is None:
            continue
        found.append(length)
    return found


def _publish(source: Path, dest: Path, cache: Path) -> None:
    """Register every length that has passed. A later length can still be added."""
    lengths = _cached_lengths(cache)
    if not lengths:
        return
    tokenizer_path = source / "tokenizer.json"
    if not tokenizer_path.is_file():
        message = "tokenizer.json is not in the checkout"
        raise FileNotFoundError(message)
    functions: list[tuple[str, Path]] = []
    recorded: dict[str, Any] = {}
    for length in lengths:
        name = f"encode_{length}"
        functions.append((name, cache / f"{name}.mlpackage"))
        recorded[str(length)] = json.loads(
            (cache / f"{name}.cosine.json").read_text(encoding="utf-8")
        )
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tokenizer_path, dest / "tokenizer.json")
    _save_functions(functions, dest / "model.mlpackage")
    floor = min(float(item["min"]) for item in recorded.values())
    mean = sum(float(item["mean"]) for item in recorded.values()) / len(recorded)
    _write_json(
        dest / "cosine.json",
        {"passages": 128, "min": floor, "mean": mean, "lengths": recorded},
    )
    _write_json(
        dest / "manifest.json",
        {
            "complete": True,
            "quant": "w8a8",
            "hidden": HIDDEN,
            "functions": [name for name, _model in functions],
        },
    )
    _register(dest, source)


def _save_functions(functions: list[tuple[str, Path]], package: Path) -> None:
    import coremltools as ct

    descriptor = ct.utils.MultiFunctionDescriptor()
    for name, path in functions:
        descriptor.add_function(
            str(path),
            src_function_name="main",
            target_function_name=name,
        )
    descriptor.default_function_name = functions[0][0]
    saving = package.with_name(f"{package.stem}.saving.mlpackage")
    shutil.rmtree(saving, ignore_errors=True)
    try:
        ct.utils.save_multifunction(descriptor, str(saving))
        if not (saving / "Manifest.json").is_file():
            message = f"{package.name} save did not finish"
            raise RuntimeError(message)
        if package.exists():
            shutil.rmtree(package)
        saving.rename(package)
        shutil.rmtree(package.with_suffix(".mlmodelc"), ignore_errors=True)
    except Exception:
        shutil.rmtree(saving, ignore_errors=True)
        raise


def _assert_engine_model(model: Any, name: str) -> None:
    loaded = _compiled_model(model)
    _assert_engine_path(Path(loaded.get_compiled_model_path()), name)


def _compiled_model(model: Any) -> Any:
    import coremltools as ct

    try:
        if model.get_compiled_model_path():
            return model
    except Exception:
        pass
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.mlpackage"
        model.save(str(path))
        return ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)


def _assert_engine_function(package: Path, name: str) -> None:
    import coremltools as ct

    model = ct.models.MLModel(
        str(package),
        function_name=name,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    _assert_engine_model(model, name)


def _assert_engine_path(package: Path, name: str) -> None:
    import coremltools as ct

    loader = getattr(getattr(ct.models, "compute_plan", None), "MLComputePlan", None)
    if loader is None or not hasattr(loader, "load_from_path"):
        message = "coremltools has no MLComputePlan, so Neural Engine residency cannot be checked"
        raise RuntimeError(message)
    plan = loader.load_from_path(str(package), compute_units=ct.ComputeUnit.CPU_AND_NE)
    missed = _off_engine(plan)
    names = sorted({item.split(" on ", 1)[0] for item in missed})
    print(f"{name} cpu {names}", flush=True)
    missed = _projection_misses(missed)
    if missed:
        shown = ", ".join(missed[:8])
        message = f"{name} left the Neural Engine ({len(missed)}): {shown}"
        raise RuntimeError(message)


def _projection_misses(missed: list[str]) -> list[str]:
    """Drop the embedding index ops. Any other op stays a failure."""
    extra: list[str] = []
    for item in missed:
        op = item.split(" on ", 1)[0]
        if op not in _INDEX_OPS:
            extra.append(item)
    return extra


def _off_engine(plan: Any) -> list[str]:
    program = plan.model_structure.program
    missed: list[str] = []
    for function in program.functions.values():
        for operation in function.block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
            if usage is None:
                continue
            devices = list(getattr(usage, "supported_compute_devices", []) or [])
            preferred = getattr(usage, "preferred_compute_device", None)
            if preferred is not None:
                devices.append(preferred)
            names = [type(device).__name__ for device in devices]
            if any("NeuralEngine" in name for name in names):
                continue
            where = type(preferred).__name__ if preferred is not None else "an unknown device"
            missed.append(f"{operation.operator_name} on {where}")
    return missed


def _report_cosine(scores: dict[str, float]) -> None:
    print(
        f"cosine passages {scores['passages']} min {scores['min']} mean {scores['mean']}",
        flush=True,
    )


def _require_finite_cosine(scores: dict[str, float], name: str) -> None:
    if math.isfinite(scores["min"]) and math.isfinite(scores["mean"]):
        return
    message = f"{name} cosine is not finite"
    raise RuntimeError(message)


def _require_cosine(scores: dict[str, float], name: str) -> None:
    _require_finite_cosine(scores, name)
    if scores["min"] >= _COSINE_MIN:
        return
    message = f"{name} cosine min {scores['min']} is below {_COSINE_MIN}"
    raise RuntimeError(message)


def _cosine(
    encoder: Any,
    package: Path,
    tokenizer: Any,
    *,
    fixed_length: int | None = None,
    compute_units: Any = None,
) -> dict[str, float]:
    import coremltools as ct
    import torch

    if compute_units is None:
        compute_units = ct.ComputeUnit.CPU_AND_NE
    scores: list[float] = []
    loaded: dict[int, Any] = {}
    encoder.eval()
    with torch.no_grad():
        for text in passages():
            ids = list(tokenizer.encode(text).ids)
            length = fixed_length if fixed_length is not None else bucket_for(len(ids))
            padded = pad_ids(ids, length)
            array = np.asarray([padded], dtype=np.int32)
            positions = np.asarray([position_ids(padded)], dtype=np.int32)
            mask = key_mask(padded)
            reference = encoder(
                torch.from_numpy(array.astype(np.int64)),
                torch.from_numpy(positions.astype(np.int64)),
                torch.from_numpy(mask.astype(np.float32)),
            )[0]
            reference = reference[0, 0].to(torch.float32).numpy()
            model = loaded.get(length)
            if model is None:
                kwargs: dict[str, Any] = {"compute_units": compute_units}
                if fixed_length is None:
                    kwargs["function_name"] = f"encode_{length}"
                model = ct.models.MLModel(str(package), **kwargs)
                loaded[length] = model
            predicted = np.asarray(
                model.predict(
                    {
                        "input_ids": array,
                        "position_ids": positions,
                        "key_mask": mask,
                    }
                )["hidden"],
                dtype=np.float32,
            )
            predicted = predicted.reshape(length, HIDDEN)[0]
            scores.append(float(np.dot(l2_normalize(reference), l2_normalize(predicted))))
    return {"passages": len(scores), "min": min(scores), "mean": sum(scores) / len(scores)}


def _read_cosine(path: Path) -> dict[str, float] | None:
    if not path.is_file():
        return None
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
        scores = {
            "passages": int(body["passages"]),
            "min": float(body["min"]),
            "mean": float(body["mean"]),
        }
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    if scores["passages"] != 128:
        return None
    if not math.isfinite(scores["min"]) or not math.isfinite(scores["mean"]):
        return None
    if scores["min"] < _COSINE_MIN:
        return None
    return scores


def _finished(dest: Path) -> bool:
    if not (dest / "model.mlpackage" / "Manifest.json").is_file():
        return False
    if not (dest / "tokenizer.json").is_file() or _read_cosine(dest / "cosine.json") is None:
        return False
    try:
        body = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(body, dict):
        return False
    if not body.get("complete") or body.get("quant") != "w8a8":
        return False
    return body.get("functions") == [f"encode_{length}" for length in _LENGTHS]


def _register(dest: Path, source: Path) -> None:
    from favlo.registry import register_quantized

    body = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    names = body.get("functions")
    if not isinstance(names, list):
        message = f"{dest.name} has no encode functions"
        raise RuntimeError(message)
    limit = lengths_from_names([str(name) for name in names])[-1]
    register_quantized(
        dest.name,
        source=source,
        alias=None,
        pipeline="bge-m3",
        runtime="metal41",
        max_prompt_tokens=limit,
    )


def _require_tools() -> None:
    try:
        import coremltools  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        message = "quantize needs torch and coremltools installed for this pipeline"
        raise RuntimeError(message) from exc
