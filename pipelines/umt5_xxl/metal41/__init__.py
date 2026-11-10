"""Shared umT5-XXL download and int8 pack.

Not a Favlo CLI pipeline. Wan packages own ``umt5 setup`` / ``umt5 quantize``
and import this package. Registry ``parts.umt5`` points at the shared
``models/google/umt5-xxl-int8`` directory.
"""

from __future__ import annotations

__all__: list[str] = []
