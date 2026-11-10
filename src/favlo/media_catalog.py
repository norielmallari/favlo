"""Media tools declared by registered pipeline modules.

The listener publishes ``{pipeline}__{local name}``. A module with no
``mcp_tools()`` is not a media plugin. Importing the module to read that
list does not open a device. A row that fails to import or declare tools is
skipped so ``favlo serve`` still listens.
"""

from __future__ import annotations

import sys
from typing import Any, NamedTuple

_RESULTS = frozenset({"image", "video", "audio", "json"})
_KIND_RESULT = {"image": "image", "video": "video", "music": "audio", "voice": "audio"}


class CatalogError(RuntimeError):
    """A registry row cannot be served as a media plugin."""


class ToolSpec(NamedTuple):
    """One local tool. The host writes the published name."""

    name: str
    description: str
    parameters: dict[str, Any]
    task: bool
    handler: str
    result: str
    kind: str


class PipelineRecord(NamedTuple):
    """One pipeline id and the folders that load its module."""

    pipeline: str
    folders: tuple[str, ...]
    tools: tuple[ToolSpec, ...]


class Catalog(NamedTuple):
    """Plugin tools for the current registry, in published-name order."""

    records: tuple[PipelineRecord, ...]
    by_name: dict[str, tuple[PipelineRecord, ToolSpec]]

    def for_kind(self, kind: str, root: Any = None) -> tuple[PipelineRecord, ToolSpec, str] | None:
        """The pipeline, tool, and folder a harness probe uses for ``kind``."""
        del root
        matches = [
            (record, spec)
            for _name, (record, spec) in self.by_name.items()
            if tool_kind(spec) == kind
        ]
        if not matches:
            return None
        generators = [item for item in matches if item[1].name.startswith("generate_")]
        record, spec = (generators or matches)[0]
        return record, spec, _resolvable_folder(record)

    def published(self) -> list[tuple[str, PipelineRecord, ToolSpec]]:
        rows = [
            (name, record, spec)
            for name, (record, spec) in self.by_name.items()
        ]
        rows.sort(key=lambda item: item[0])
        return rows


def _resolvable_folder(record: PipelineRecord) -> str:
    """A model id ``resolve`` accepts. The pipeline id is the fallback."""
    from favlo.registry import resolve

    for candidate in (*record.folders, record.pipeline):
        try:
            resolve(candidate)
        except LookupError:
            continue
        return candidate
    return record.folders[0]


def publish(pipeline: str, local: str) -> str:
    """The tool name a client calls. Prefix is the pipeline id."""
    return f"{pipeline}__{local}"


def tool_kind(spec: ToolSpec) -> str:
    """The harness key for one tool. Video splits on the local name."""
    if spec.kind == "video":
        return "t2v" if "t2v" in spec.name else "i2v"
    if spec.kind in {"image", "music", "voice"}:
        return spec.kind
    message = f"{spec.handler} is not a media handler"
    raise CatalogError(message)


def _skip(message: str) -> None:
    """Leave the broken row out of the catalog. Serve keeps listening."""
    sys.stderr.write(f"media catalog: {message}\n")


def build_catalog(
    root: Any = None,
    *,
    description_max: int = 256,
    param_max: int = 48,
) -> Catalog:
    """Walk the registry once. A broken pipeline is skipped; the rest stay."""
    from favlo.cli import load_pipeline
    from favlo.registry import load_registry

    data = load_registry(root)
    models = data.get("models")
    if not isinstance(models, dict):
        models = {}
    groups: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    order: list[str] = []
    for folder, entry in models.items():
        if not isinstance(folder, str) or not isinstance(entry, dict):
            continue
        pipeline = entry.get("pipeline")
        runtime = entry.get("runtime")
        if not isinstance(pipeline, str) or not pipeline or not isinstance(runtime, str) or not runtime:
            continue
        if pipeline not in groups:
            order.append(pipeline)
            groups[pipeline] = []
        groups[pipeline].append((folder, entry))

    records: list[PipelineRecord] = []
    by_name: dict[str, tuple[PipelineRecord, ToolSpec]] = {}
    for pipeline in order:
        rows = groups[pipeline]
        folder, entry = rows[0]
        root_path = entry.get("pipeline_root")
        try:
            module = load_pipeline(
                pipeline,
                runtime=str(entry.get("runtime")),
                pipeline_root=root_path if isinstance(root_path, str) else None,
            )
        except Exception as exc:
            _skip(f"{folder} did not import ({type(exc).__name__}: {exc})")
            continue
        declare = getattr(module, "mcp_tools", None)
        if not callable(declare):
            continue
        try:
            raw = declare()
        except Exception as exc:
            _skip(f"{pipeline} mcp_tools failed ({type(exc).__name__}: {exc})")
            continue
        if not raw:
            continue
        try:
            specs = _tools(module, pipeline, raw, description_max, param_max)
        except CatalogError as exc:
            _skip(str(exc))
            continue
        pending: list[tuple[str, ToolSpec]] = []
        collision = False
        for spec in specs:
            name = publish(pipeline, spec.name)
            if name in by_name:
                _skip(f"published name {name} collides; skipping {pipeline}")
                collision = True
                break
            pending.append((name, spec))
        if collision or not pending:
            continue
        record = PipelineRecord(
            pipeline,
            tuple(name for name, _entry in rows),
            tuple(spec for _name, spec in pending),
        )
        for name, spec in pending:
            by_name[name] = (record, spec)
        records.append(record)
    return Catalog(tuple(records), by_name)


def default_folder(record: PipelineRecord, spec: ToolSpec, root: Any = None) -> str:
    """The folder a call uses when ``model`` is omitted. The pipeline is the model."""
    del spec, root
    return record.folders[0]


def _tools(
    module: Any,
    pipeline: str,
    raw: Any,
    description_max: int,
    param_max: int,
) -> list[ToolSpec]:
    if not isinstance(raw, list):
        message = f"{pipeline} mcp_tools did not return a list"
        raise CatalogError(message)
    seen: set[str] = set()
    specs: list[ToolSpec] = []
    for item in raw:
        spec = _spec(module, pipeline, item, description_max, param_max)
        if spec.name in seen:
            message = f"{pipeline} repeats {spec.name}"
            raise CatalogError(message)
        seen.add(spec.name)
        specs.append(spec)
    return specs


def _spec(
    module: Any,
    pipeline: str,
    item: Any,
    description_max: int,
    param_max: int,
) -> ToolSpec:
    if not isinstance(item, dict):
        message = f"{pipeline} tool is not an object"
        raise CatalogError(message)
    name = item.get("name")
    if not isinstance(name, str) or not name or "__" in name:
        message = f"{pipeline} tool name is not a local name"
        raise CatalogError(message)
    description = item.get("description")
    if not isinstance(description, str) or len(description) > description_max:
        message = f"{pipeline} {name} description is over {description_max}"
        raise CatalogError(message)
    parameters = item.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        message = f"{pipeline} {name} parameters are not an object schema"
        raise CatalogError(message)
    _parameters(pipeline, name, parameters, param_max)
    task = item.get("task")
    if not isinstance(task, bool):
        message = f"{pipeline} {name} task is not a boolean"
        raise CatalogError(message)
    handler = item.get("handler")
    call = getattr(module, handler, None) if isinstance(handler, str) else None
    if not isinstance(handler, str) or not handler or not callable(call):
        message = f"{pipeline} {name} handler is missing"
        raise CatalogError(message)
    result = item.get("result")
    if result not in _RESULTS:
        message = f"{pipeline} {name} result is not image, video, audio, or json"
        raise CatalogError(message)
    kind = getattr(call, "kind", "")
    if task:
        from favlo.media_pipeline import MediaPipeline

        if not isinstance(call, MediaPipeline) or kind not in _KIND_RESULT:
            message = f"{pipeline} {name} handler is not a MediaPipeline"
            raise CatalogError(message)
        if result != _KIND_RESULT[kind]:
            message = f"{pipeline} {name} result does not match {kind}"
            raise CatalogError(message)
    elif not isinstance(kind, str):
        kind = ""
    return ToolSpec(name, description, parameters, task, handler, str(result), str(kind))


def _parameters(pipeline: str, name: str, schema: dict[str, Any], param_max: int) -> None:
    properties = schema.get("properties")
    if properties is None:
        properties = {}
    if not isinstance(properties, dict):
        message = f"{pipeline} {name} properties are not an object"
        raise CatalogError(message)
    for key, prop in properties.items():
        if not isinstance(prop, dict):
            message = f"{pipeline} {name} {key} is not an object"
            raise CatalogError(message)
        text = prop.get("description") or ""
        if not isinstance(text, str) or len(text) > param_max:
            message = f"{pipeline} {name} {key} description is over {param_max}"
            raise CatalogError(message)
    required = schema.get("required") or []
    if not isinstance(required, list) or any(not isinstance(key, str) for key in required):
        message = f"{pipeline} {name} required is not a list of names"
        raise CatalogError(message)
