"""Load one function of the packed mlprogram through Core ML.

Quantize is the only importer of coremltools. This module compiles the
package once and predicts with the Neural Engine.
"""

from __future__ import annotations

import ctypes
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def compiled_model(package: Path, dest: Path) -> Path:
    """Compile ``package`` to ``dest`` when that directory is absent."""
    if dest.is_dir():
        return dest
    import CoreML
    from Foundation import NSURL

    url = NSURL.fileURLWithPath_(str(package))
    compiled, error = CoreML.MLModel.compileModelAtURL_error_(url, None)
    if compiled is None:
        message = "the mlprogram did not compile"
        if error is not None:
            message = f"{message}: {error.localizedDescription()}"
        raise RuntimeError(message)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(compiled.path()), dest)
    return dest


def load_function(compiled: Path, name: str) -> Any:
    """One Neural Engine function from the compiled program."""
    import CoreML
    from Foundation import NSURL

    config = CoreML.MLModelConfiguration.alloc().init()
    config.setComputeUnits_(CoreML.MLComputeUnitsCPUAndNeuralEngine)
    config.setFunctionName_(name)
    url = NSURL.fileURLWithPath_(str(compiled))
    model, error = CoreML.MLModel.modelWithContentsOfURL_configuration_error_(url, config, None)
    if model is None:
        message = f"{name} did not load"
        if error is not None:
            message = f"{message}: {error.localizedDescription()}"
        raise RuntimeError(message)
    return model


def predict_hidden(model: Any, features: dict[str, np.ndarray]) -> np.ndarray:
    """Hidden state for one feature dictionary."""
    import CoreML
    import objc

    with objc.autorelease_pool():
        values = {
            name: CoreML.MLFeatureValue.featureValueWithMultiArray_(
                multiarray_from_numpy(np.ascontiguousarray(array))
            )
            for name, array in features.items()
        }
        provider, error = CoreML.MLDictionaryFeatureProvider.alloc().initWithDictionary_error_(
            values,
            None,
        )
        if provider is None:
            message = "the input was rejected"
            if error is not None:
                message = f"{message}: {error.localizedDescription()}"
            raise RuntimeError(message)
        result, error = model.predictionFromFeatures_error_(provider, None)
        if result is None:
            message = "the prediction failed"
            if error is not None:
                message = f"{message}: {error.localizedDescription()}"
            raise RuntimeError(message)
        feature = result.featureValueForName_("hidden")
        if feature is None:
            message = "the program did not return hidden"
            raise RuntimeError(message)
        return numpy_from_multiarray(feature.multiArrayValue())


def multiarray_from_numpy(array: np.ndarray) -> Any:
    """A contiguous Core ML array with the same values."""
    import CoreML

    values = np.ascontiguousarray(array)
    kind = _coreml_type(values.dtype)
    shape = [int(n) for n in values.shape]
    multi, error = CoreML.MLMultiArray.alloc().initWithShape_dataType_error_(shape, kind, None)
    if multi is None:
        message = "the multiarray was not created"
        if error is not None:
            message = f"{message}: {error.localizedDescription()}"
        raise RuntimeError(message)
    addr = int(multi.mutableBytes())
    ctypes.memmove(addr, values.ctypes.data, values.nbytes)
    return multi


def numpy_from_multiarray(multi: Any) -> np.ndarray:
    """Copy a Core ML array into a NumPy array of the same dtype."""
    import CoreML

    table = {
        int(CoreML.MLMultiArrayDataTypeInt32): np.int32,
        int(CoreML.MLMultiArrayDataTypeFloat32): np.float32,
        int(CoreML.MLMultiArrayDataTypeFloat): np.float32,
        int(CoreML.MLMultiArrayDataTypeFloat16): np.float16,
        int(CoreML.MLMultiArrayDataTypeDouble): np.float64,
        int(CoreML.MLMultiArrayDataTypeFloat64): np.float64,
    }
    dtype = table.get(int(multi.dataType()))
    if dtype is None:
        message = f"unsupported multiarray type {int(multi.dataType())}"
        raise RuntimeError(message)
    shape = tuple(int(n) for n in multi.shape())
    strides = [int(n) for n in multi.strides()]
    item = int(np.dtype(dtype).itemsize)
    last = sum((size - 1) * stride for size, stride in zip(shape, strides, strict=True)) + 1
    raw = np.empty(last, dtype=dtype)
    ctypes.memmove(raw.ctypes.data, int(multi.mutableBytes()), last * item)
    view = np.lib.stride_tricks.as_strided(
        raw,
        shape=shape,
        strides=tuple(stride * item for stride in strides),
    )
    return np.array(view, copy=True)


def _coreml_type(dtype: np.dtype) -> Any:
    import CoreML

    kind = np.dtype(dtype)
    if kind == np.int32:
        return CoreML.MLMultiArrayDataTypeInt32
    if kind == np.float32:
        return CoreML.MLMultiArrayDataTypeFloat32
    if kind == np.float16:
        return CoreML.MLMultiArrayDataTypeFloat16
    if kind == np.float64:
        return CoreML.MLMultiArrayDataTypeDouble
    message = f"unsupported array dtype {kind}"
    raise TypeError(message)
