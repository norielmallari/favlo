"""Process memory manager.

Pipelines ask for integer handles. This module is the only caller of
``MTLCreateSystemDefaultDevice``, ``newBufferWithLength``, and
``newBufferWithBytes``. A handle names a span in one heap buffer. ``release``
returns the span. ``sweep`` drops that owner's heap buffers.
"""

from __future__ import annotations

import ctypes
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from favlo.offset import BumpMap, OffsetMap, align, bin_size

_PAGE = 4096
_SLAB = 64 << 20
_RECLAIM_WAIT = 5.0
_RECLAIM_PAUSE = 0.05
_EMPTY = 4
_NONVOLATILE = 2
_READ = 1
_WRITE = 2


def reclaimable_bytes() -> int:
    """Free, inactive, and speculative pages."""
    libc = ctypes.CDLL("/usr/lib/libSystem.dylib", use_errno=True)

    class Stats(ctypes.Structure):
        _fields_ = (
            ("free_count", ctypes.c_uint),
            ("active_count", ctypes.c_uint),
            ("inactive_count", ctypes.c_uint),
            ("wire_count", ctypes.c_uint),
            ("zero_fill_count", ctypes.c_uint64),
            ("reactivations", ctypes.c_uint64),
            ("pageins", ctypes.c_uint64),
            ("pageouts", ctypes.c_uint64),
            ("faults", ctypes.c_uint64),
            ("cow_faults", ctypes.c_uint64),
            ("lookups", ctypes.c_uint64),
            ("hits", ctypes.c_uint64),
            ("purges", ctypes.c_uint64),
            ("purgeable_count", ctypes.c_uint),
            ("speculative_count", ctypes.c_uint),
        )

    count = ctypes.c_uint(ctypes.sizeof(Stats) // 4)
    stats = Stats()
    kern = libc.host_statistics64(
        libc.mach_host_self(),
        ctypes.c_int(4),
        ctypes.byref(stats),
        ctypes.byref(count),
    )
    page = int(os.sysconf("SC_PAGE_SIZE"))
    if kern != 0:
        return page * int(os.sysconf("SC_PHYS_PAGES"))
    reclaimable = stats.free_count + stats.inactive_count + stats.speculative_count
    return int(reclaimable) * page


class Handle:
    """Integer span. The heap buffer stays in the manager."""

    __slots__ = ("id",)

    def __init__(self, id_: int) -> None:
        self.id = int(id_)

    def length(self) -> int:
        try:
            return memory.nbytes(self)
        except KeyError:
            return 0

    def contents(self) -> _Contents:
        return _Contents(self)

    def release(self) -> None:
        memory.release(self)

    def setPurgeableState_(self, state: int) -> int:
        return memory.purge(self, int(state))

    def __int__(self) -> int:
        return self.id

    def __index__(self) -> int:
        return self.id


class _Contents:
    __slots__ = ("handle",)

    def __init__(self, handle: Handle) -> None:
        self.handle = handle

    def as_buffer(self, count: int) -> memoryview:
        return memory.map(self.handle, 0, int(count))


@dataclass
class _Segment:
    owner: str
    kind: str
    storage: str
    heap: Any
    buffer: Any
    size: int
    alignment: int
    mapping: OffsetMap | BumpMap
    queued: bool = False
    tokens: set[int] = field(default_factory=set)
    departed: int = 0


@dataclass
class _Span:
    segment: _Segment
    token: int
    offset: int
    length: int
    requested: int
    owner: str
    kind: str
    storage: str
    released: bool = False
    advised: bool = False


class Memory:
    """One device, placement heaps, and the handle table."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._device: Any = None
        self._metal: Any = None
        self._set: Any = None
        self._set_pool: Any = None
        self._queues: list[Any] = []
        self._watchers: list[Any] = []
        self._segments: list[_Segment] = []
        self._spans: dict[int, _Span] = {}
        self._next = 1
        self._live = 0
        self._pending = 0
        self._returning = 0
        self._reported: dict[str, int] = {}
        self._reclaim_mark = 0
        self._fence: Any = None
        self._wait_fence = False
        self._commands: dict[int, Any] = {}
        self._signalers: set[int] = set()
        self._deferred: list[tuple[_Segment, int, set[int]]] = []
        self._align = 256

    @property
    def device(self) -> Any:
        with self._lock:
            return self._device_locked()

    @property
    def live(self) -> int:
        with self._lock:
            return self._live

    @property
    def pending(self) -> int:
        with self._lock:
            self._retire_pending()
            return self._pending

    def budget(self) -> int:
        """Reclaimable pages, live spans, reported child bytes, and heaps not yet returned."""
        with self._lock:
            self._retire_pending()
            reported = sum(self._reported.values())
            return reclaimable_bytes() + self._live + self._pending + self._returning + reported

    def report(self, counts: dict[str, int]) -> None:
        """Remember bytes a pipeline process reported. This does not create a device."""
        with self._lock:
            for owner, nbytes in counts.items():
                if isinstance(owner, str) and owner:
                    self._reported[owner] = max(int(nbytes), 0)

    def forget(self, owners: Any) -> None:
        """Drop reported bytes for owners whose process has exited."""
        with self._lock:
            self._forget_locked(owners)

    def confirm_released(self, before: int, nbytes: int, *, timeout: float = _RECLAIM_WAIT) -> bool:
        """Wait until reclaimable pages cover ``nbytes`` that were wired at ``before``.

        ``before`` is the reclaimable count while those pages were still wired.
        The count is confirmed when that total has risen by ``nbytes``, within
        one heap slab. A count of one slab or less is already confirmed. The
        wait stops after ``timeout`` seconds.
        """
        if nbytes <= _SLAB:
            return True
        target = before + nbytes - _SLAB
        deadline = time.monotonic() + timeout
        while True:
            if reclaimable_bytes() >= target:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(_RECLAIM_PAUSE)

    def reported(self) -> dict[str, int]:
        with self._lock:
            return dict(self._reported)

    def owner_bytes(self) -> dict[str, int]:
        """Live bytes per owner. Reads the span table and does not create a device."""
        with self._lock:
            counts: dict[str, int] = {}
            for span in self._spans.values():
                if span.released or not span.owner:
                    continue
                counts[span.owner] = counts.get(span.owner, 0) + span.requested
            return counts

    def _forget_locked(self, owners: Any) -> None:
        if not owners:
            return
        for owner in owners:
            if isinstance(owner, str) and owner:
                self._reported.pop(owner, None)

    def make_queue(self) -> Any:
        with self._lock:
            device = self._device_locked()
            queue = device.newCommandQueue()
            if queue is None:
                message = "Metal command queue is missing"
                raise RuntimeError(message)
            self._attach_locked(queue)
            return queue

    def watch(self, residency: Any) -> None:
        """Publish later heaps into a pipeline residency set."""
        with self._lock:
            self._ensure_set()
            if residency not in self._watchers:
                self._watchers.append(residency)
            for segment in self._segments:
                residency.addAllocation_(segment.heap)
            residency.commit()
            residency.requestResidency()

    def allocate(
        self,
        nbytes: int,
        *,
        owner: str,
        heap: str = "resident",
        storage: str = "shared",
        zero: bool = False,
    ) -> Handle:
        kind = "private" if storage == "private" or heap == "private" else heap
        mode = "private" if kind == "private" or storage == "private" else "shared"
        if kind not in {"resident", "scratch", "private"}:
            message = f"unknown heap {heap}"
            raise ValueError(message)
        request = max(int(nbytes), 16)
        with self._lock:
            span = self._span_locked(request, owner, kind, mode)
            handle = Handle(span_id := self._next)
            self._next += 1
            self._spans[span_id] = span
            self._live += span.requested
        if zero:
            self.zero(handle, 0, span.requested)
        return handle

    def upload_group(self, parts: list[bytes | Any], *, owner: str) -> list[Handle]:
        """One heap for a quantized matrix. The first plane starts at offset zero.

        Blockwise tensor data reads the wrong bytes when its plane offset is
        not zero. The scale plane may follow it in the same heap.
        """
        raws = [_as_bytes(part) for part in parts]
        if not raws:
            return []
        alignment = self._align
        needs = [bin_size(max(len(raw), 16), alignment) for raw in raws]
        with self._lock:
            self._add_segment(sum(needs), owner, "quant", "shared")
            handles: list[Handle] = []
            for raw in raws:
                span = self._span_locked(max(len(raw), 16), owner, "quant", "shared")
                handle = Handle(span_id := self._next)
                self._next += 1
                self._spans[span_id] = span
                self._live += span.requested
                handles.append(handle)
        for handle, raw in zip(handles, raws, strict=True):
            if raw:
                self.store(handle, 0, raw)
        return handles

    def upload(self, data: bytes | Any, *, owner: str, storage: str = "shared") -> Handle:
        raw = _as_bytes(data)
        if storage == "private":
            staging = self.allocate(len(raw), owner=owner, heap="resident", storage="shared")
            self.store(staging, 0, raw)
            private = self.allocate(len(raw), owner=owner, heap="private", storage="private")
            self.copy(staging, 0, private, 0, len(raw))
            self.release(staging)
            self.complete()
            return private
        handle = self.allocate(len(raw), owner=owner, heap="resident", storage="shared")
        if raw:
            self.store(handle, 0, raw)
        return handle

    def store(self, handle: Handle, offset: int, data: bytes | Any) -> None:
        raw = _as_bytes(data)
        if not raw:
            return
        view = self.map(handle, int(offset), len(raw))
        view[: len(raw)] = raw

    def load(self, handle: Handle, offset: int, nbytes: int) -> bytes:
        if nbytes < 1:
            return b""
        return bytes(self.map(handle, int(offset), int(nbytes)))

    def zero(self, handle: Handle, offset: int, nbytes: int) -> None:
        self.fill(handle, offset, nbytes, 0)

    def fill(self, handle: Handle, offset: int, nbytes: int, value: int) -> None:
        count = int(nbytes)
        if count < 1:
            return
        span = self._require(handle)
        if span.storage == "shared":
            _fill_shared(handle, int(offset), count, int(value) & 0xFF)
            return
        self._blit(span, int(offset), count, int(value) & 0xFF)

    def copy(
        self,
        src: Handle,
        src_offset: int,
        dst: Handle,
        dst_offset: int,
        nbytes: int,
    ) -> None:
        count = int(nbytes)
        if count < 1:
            return
        source = self._require(src)
        dest = self._require(dst)
        self._copy(source, int(src_offset), dest, int(dst_offset), count)

    def map(self, handle: Handle, offset: int, nbytes: int) -> memoryview:
        span = self._require(handle)
        if span.storage != "shared":
            message = "a private span has no CPU mapping"
            raise RuntimeError(message)
        start = span.offset + int(offset)
        end = start + int(nbytes)
        if int(offset) < 0 or end > span.offset + span.length:
            message = f"map {offset}+{nbytes} is outside the span"
            raise ValueError(message)
        base = span.segment.buffer.contents().as_buffer(span.segment.size)
        return memoryview(base)[start:end]

    def unmap(self, handle: Handle) -> None:
        self._require(handle)

    def release_span(self, handle: Handle, offset: int, nbytes: int) -> None:
        self._advise(handle, int(offset), int(nbytes), 7)
        with self._lock:
            self._pending += int(nbytes)

    def commit_span(self, handle: Handle, offset: int, nbytes: int) -> None:
        self._advise(handle, int(offset), int(nbytes), 8)

    def discard(self, handle: Handle) -> None:
        """Mark one shared span reusable. The budget keeps it until the kernel reports it."""
        span = self._lookup(handle)
        if span is None or span.released or span.advised:
            return
        page = int(os.sysconf("SC_PAGE_SIZE"))
        aligned = span.requested - (span.requested % page)
        if aligned < page or span.storage != "shared":
            return
        self._advise(handle, 0, aligned, 7)
        with self._lock:
            if span.advised:
                return
            self._pending += aligned
            span.advised = True

    def purge(self, handle: Handle, state: int) -> int:
        if self._lookup(handle) is None:
            return _NONVOLATILE
        if int(state) == _EMPTY:
            self.discard(handle)
        elif int(state) == _NONVOLATILE:
            span = self._require(handle)
            page = int(os.sysconf("SC_PAGE_SIZE"))
            aligned = span.requested - (span.requested % page)
            if aligned >= page and span.storage == "shared":
                self.commit_span(handle, 0, aligned)
        return _NONVOLATILE

    def bind(self, encoder: Any, index: int, handle: Any, offset: int = 0) -> None:
        span = self._lookup(handle)
        if span is None:
            encoder.setBuffer_offset_atIndex_(handle, int(offset), index)
            return
        self._bind_span(encoder, index, span, int(offset))

    def bind_plane(self, attachments: Any, plane: int, handle: Any, offset: int = 0) -> None:
        span = self._lookup(handle)
        if span is None:
            attachments.setBuffer_offset_forPlane_(handle, int(offset), int(plane))
            return
        attachments.setBuffer_offset_forPlane_(
            span.segment.buffer, span.offset + int(offset), int(plane)
        )

    def bind_indirect(self, encoder: Any, handle: Any, offset: int, threads: Any) -> None:
        span = self._require(handle)
        encoder.dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup_(
            span.segment.buffer,
            span.offset + int(offset),
            threads,
        )

    def nbytes(self, handle: Handle) -> int:
        return self._require(handle).requested

    def kind(self, handle: Any) -> str:
        found = self._lookup(handle)
        if found is None:
            return ""
        return found.kind

    def heap_resource(self, handle: Any) -> Any:
        return self._require(handle).segment.heap

    def release(self, handle: Any) -> None:
        span = self._lookup(handle)
        if span is None or span.released:
            return
        with self._lock:
            self._free_span(span)
            self._drop_idle_segments()

    def owners(self) -> frozenset[str]:
        """Owner strings that still have a live span."""
        with self._lock:
            return frozenset(
                span.owner for span in self._spans.values() if not span.released and span.owner
            )

    def sweep(self, owner: str) -> None:
        with self._lock:
            self._forget_locked((owner,))
            self._retire_commands()
            spans = [span for span in self._spans.values() if span.owner == owner]
            for span in spans:
                self._free_span(span)
            self._drop_idle_segments()
            self._retire_pending()

    def sweep_new(self, kept: Any) -> None:
        """Sweep owners that were not in ``kept``. Owners in ``kept`` stay."""
        keep = {owner for owner in kept if isinstance(owner, str)}
        with self._lock:
            fresh = {
                span.owner
                for span in self._spans.values()
                if not span.released and span.owner and span.owner not in keep
            }
        for owner in fresh:
            self.sweep(owner)

    def complete(self) -> None:
        """Wait for tracked command buffers, then rewind scratch."""
        with self._lock:
            self._retire_commands()
            self._rewind_scratch()

    def command_buffer(self, queue: Any) -> Any:
        """A retained command buffer whose extra retain is drained in ``retire``."""
        import objc

        with self._lock:
            pool = objc.autorelease_pool()
            pool.__enter__()
            try:
                command = queue.commandBuffer()
            except Exception:
                pool.__exit__(None, None, None)
                raise
            pool.__exit__(None, None, None)
            if command is None:
                message = "Metal command buffer is missing"
                raise RuntimeError(message)
            self._commands[id(command)] = command
            return command

    def retire(self, command: Any) -> tuple[int, Any]:
        status_of = getattr(command, "status", None) if command is not None else None
        error_of = getattr(command, "error", None) if command is not None else None
        status = int(status_of()) if callable(status_of) else 0
        error = error_of() if callable(error_of) else None
        if command is not None and status in {1, 2, 3}:
            command.waitUntilCompleted()
            status = int(status_of()) if callable(status_of) else status
            error = error_of() if callable(error_of) else error
        with self._lock:
            self._drop_command(command)
            self._rewind_scratch()
        return status, error

    def handoff(self, encoder: Any, command: Any = None) -> None:
        """Update the fence before ``endEncoding`` so the next encoder can wait."""
        with self._lock:
            fence = self._fence_locked()
            encoder.updateFence_(fence)
            self._wait_fence = True
            if command is not None:
                self._signalers.add(id(command))

    def _fence_pending(self, encoder: Any) -> bool:
        """Wait for the command buffer that updated the fence, until it finishes.

        A finished signaler must not arm another wait: that wait would look for a
        future update, including one at the end of the encoder being recorded.
        """
        del encoder
        if not self._wait_fence or self._fence is None:
            return False
        for command in self._commands.values():
            if id(command) not in self._signalers:
                continue
            status_of = getattr(command, "status", None)
            status = int(status_of()) if callable(status_of) else 4
            if status in {0, 1, 2, 3}:
                return True
        self._wait_fence = False
        return False

    def _bind_span(self, encoder: Any, index: int, span: _Span, offset: int) -> None:
        with self._lock:
            if self._fence_pending(encoder):
                encoder.waitForFence_(self._fence)
                self._wait_fence = False
            if not span.segment.queued:
                encoder.useHeap_(span.segment.heap)
        encoder.setBuffer_offset_atIndex_(span.segment.buffer, span.offset + offset, index)

    def _device_locked(self) -> Any:
        if self._device is not None:
            return self._device
        import Metal

        device = Metal.MTLCreateSystemDefaultDevice()
        if device is None:
            message = "Metal device is missing"
            raise RuntimeError(message)
        self._metal = Metal
        self._device = device
        sample = device.heapBufferSizeAndAlignWithLength_options_(16, self._options("shared"))
        self._align = max(int(sample.align), 256)
        self._reclaim_mark = reclaimable_bytes()
        return device

    def _options(self, storage: str) -> int:
        metal = self._metal
        mode = (
            metal.MTLResourceStorageModePrivate
            if storage == "private"
            else metal.MTLResourceStorageModeShared
        )
        return int(mode) | int(metal.MTLResourceCPUCacheModeDefaultCache)

    def _span_locked(self, request: int, owner: str, kind: str, storage: str) -> _Span:
        self._device_locked()
        placed = self._place(request, owner, kind, storage)
        if placed is None:
            self._drop_idle_segments()
            self._add_segment(request, owner, kind, storage)
            placed = self._place(request, owner, kind, storage)
        if placed is None:
            message = f"no span for {request} bytes"
            raise RuntimeError(message)
        segment, token, offset, length = placed
        segment.tokens.add(token)
        return _Span(segment, token, offset, length, request, owner, kind, storage)

    def _place(
        self, request: int, owner: str, kind: str, storage: str
    ) -> tuple[_Segment, int, int, int] | None:
        for segment in self._segments:
            if segment.owner != owner or segment.kind != kind or segment.storage != storage:
                continue
            found = segment.mapping.allocate(request)
            if found is None:
                continue
            token, offset = found
            _offset, length = segment.mapping.placement(token)
            return segment, token, offset, length
        return None

    def _add_segment(self, request: int, owner: str, kind: str, storage: str) -> _Segment:
        import objc

        device = self._device_locked()
        metal = self._metal
        alignment = self._align
        floor = _SLAB if kind == "resident" else 0
        room = bin_size(max(request, _PAGE, floor), alignment)
        options = self._options(storage)
        info = device.heapBufferSizeAndAlignWithLength_options_(room, options)
        buf_len = int(info.size)
        heap_size = align(max(room, buf_len), alignment)
        descriptor = metal.MTLHeapDescriptor.alloc().init()
        descriptor.setSize_(heap_size)
        descriptor.setType_(metal.MTLHeapTypePlacement)
        descriptor.setStorageMode_(
            metal.MTLStorageModePrivate if storage == "private" else metal.MTLStorageModeShared
        )
        descriptor.setCpuCacheMode_(metal.MTLCPUCacheModeDefaultCache)
        pool = objc.autorelease_pool()
        pool.__enter__()
        try:
            heap = device.newHeapWithDescriptor_(descriptor)
            if heap is None:
                message = f"heap of {heap_size} bytes was refused"
                raise RuntimeError(message)
            buffer = heap.newBufferWithLength_options_offset_(buf_len, options, 0)
        finally:
            pool.__exit__(None, None, None)
        if buffer is None:
            message = f"heap buffer of {buf_len} bytes was refused"
            raise RuntimeError(message)
        mapping: OffsetMap | BumpMap
        if kind == "scratch":
            mapping = BumpMap(buf_len, alignment)
        else:
            mapping = OffsetMap(buf_len, alignment)
        segment = _Segment(owner, kind, storage, heap, buffer, buf_len, alignment, mapping)
        self._segments.append(segment)
        self._publish(segment)
        return segment

    def _publish(self, segment: _Segment) -> None:
        self._ensure_set()
        limit = int(self._device.recommendedMaxWorkingSetSize())
        used = sum(item.size for item in self._segments if item.queued)
        if self._set is not None and used + segment.size <= limit:
            self._set.addAllocation_(segment.heap)
            self._set.commit()
            self._set.requestResidency()
            segment.queued = True
        for watcher in self._watchers:
            watcher.addAllocation_(segment.heap)
            watcher.commit()
            watcher.requestResidency()

    def _ensure_set(self) -> None:
        if self._set is not None or self._device is None:
            return
        metal = self._metal
        descriptor = metal.MTLResidencySetDescriptor.alloc().init()
        created = self._device.newResidencySetWithDescriptor_error_(descriptor, None)
        residency, error = _pair(created)
        if residency is None:
            message = f"residency set failed: {error}"
            raise RuntimeError(message)
        self._set = residency
        for queue in self._queues:
            queue.addResidencySet_(residency)

    def _attach_locked(self, queue: Any) -> None:
        self._queues.append(queue)
        self._ensure_set()
        if self._set is not None:
            queue.addResidencySet_(self._set)
            limit = int(self._device.recommendedMaxWorkingSetSize())
            used = sum(item.size for item in self._segments if item.queued)
            for segment in self._segments:
                if segment.queued or segment.heap is None:
                    continue
                if used + segment.size > limit:
                    continue
                self._set.addAllocation_(segment.heap)
                segment.queued = True
                used += segment.size
            self._set.commit()
            self._set.requestResidency()

    def _free_span(self, span: _Span) -> None:
        if span.released:
            return
        span.released = True
        self._live -= span.requested
        key = self._id_of(span)
        if key is not None:
            self._spans.pop(key, None)
        if self._commands:
            self._deferred.append((span.segment, span.token, set(self._commands)))
            if not span.advised:
                self._pending += span.requested
                span.advised = True
            return
        span.segment.mapping.free(span.token)
        span.segment.tokens.discard(span.token)
        if not span.advised:
            span.segment.departed += span.requested

    def _id_of(self, span: _Span) -> int | None:
        for key, value in self._spans.items():
            if value is span:
                return key
        return None

    def _drop_idle_segments(self) -> None:
        held = {id(segment) for segment, _token, _waiting in self._deferred}
        idle = [
            segment
            for segment in self._segments
            if not segment.tokens and id(segment) not in held
        ]
        changed = False
        for segment in idle:
            self._segments.remove(segment)
            if segment.queued and self._set is not None and segment.heap is not None:
                self._set.removeAllocation_(segment.heap)
                segment.queued = False
                changed = True
            for watcher in self._watchers:
                if segment.heap is not None:
                    watcher.removeAllocation_(segment.heap)
                    changed = True
            self._release_once(segment.buffer)
            self._release_once(segment.heap)
            segment.buffer = None
            segment.heap = None
            self._returning += segment.departed
            segment.departed = 0
        if changed and self._set is not None:
            self._set.commit()
            if not any(segment.queued for segment in self._segments):
                self._set.endResidency()

    def _drop_object(self, obj: Any) -> None:
        """Drop the Python owner. The creation pool already consumed the extra retain."""
        del obj

    def _release_once(self, obj: Any) -> None:
        """Balance the extra retain on a heap or its buffer. The proxy keeps the last one."""
        if obj is None:
            return
        release = getattr(obj, "release", None)
        if callable(release):
            release()

    def _rewind_scratch(self) -> None:
        if self._commands:
            return
        for segment in self._segments:
            if segment.kind != "scratch":
                continue
            mapping = segment.mapping
            if isinstance(mapping, BumpMap):
                mapping.rewind()
            segment.tokens.clear()
        dead = [key for key, span in self._spans.items() if span.kind == "scratch"]
        for key in dead:
            span = self._spans.pop(key)
            if not span.released:
                self._live -= span.requested
            span.released = True

    def _retire_commands(self) -> None:
        commands = list(self._commands.values())
        for command in commands:
            status_of = getattr(command, "status", None)
            status = int(status_of()) if callable(status_of) else 0
            if status in {1, 2, 3}:
                command.waitUntilCompleted()
            self._drop_command(command)
        self._return_deferred()

    def _drop_command(self, command: Any) -> None:
        if command is None or id(command) not in self._commands:
            return
        self._signalers.discard(id(command))
        self._commands.pop(id(command), None)
        self._drop_object(command)
        self._return_deferred()

    def _return_deferred(self) -> None:
        alive = set(self._commands)
        still: list[tuple[_Segment, int, set[int]]] = []
        for segment, token, waiting in self._deferred:
            if waiting & alive:
                still.append((segment, token, waiting))
                continue
            segment.mapping.free(token)
            segment.tokens.discard(token)
        self._deferred = still
        self._drop_idle_segments()

    def _retire_pending(self) -> None:
        now = reclaimable_bytes()
        delta = now - self._reclaim_mark
        if delta > 0:
            take = min(self._returning, delta)
            self._returning -= take
            rest = delta - take
            if rest > 0 and self._pending:
                self._pending = max(0, self._pending - rest)
        if now > self._reclaim_mark:
            self._reclaim_mark = now

    def _require(self, handle: Any) -> _Span:
        found = self._lookup(handle)
        if found is None:
            message = "the handle is not a live span"
            raise KeyError(message)
        return found

    def _lookup(self, handle: Any) -> _Span | None:
        if isinstance(handle, Handle):
            return self._spans.get(handle.id)
        if isinstance(handle, int) and not isinstance(handle, bool):
            return self._spans.get(handle)
        return None

    def _fence_locked(self) -> Any:
        if self._fence is None:
            self._fence = self._device_locked().newFence()
        return self._fence

    def _advise(self, handle: Handle, offset: int, nbytes: int, advice: int) -> None:
        if nbytes < 1:
            return
        view = self.map(handle, offset, nbytes)
        holder = ctypes.c_char.from_buffer(view)
        try:
            address = ctypes.addressof(holder)
        finally:
            del holder
        libc = ctypes.CDLL("/usr/lib/libSystem.dylib", use_errno=True)
        libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        libc.madvise.restype = ctypes.c_int
        rc = libc.madvise(ctypes.c_void_p(address), ctypes.c_size_t(nbytes), ctypes.c_int(advice))
        if rc != 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))

    def _blit(self, span: _Span, offset: int, nbytes: int, value: int) -> None:
        queue = self.make_queue()
        command = self.command_buffer(queue)
        blit = command.blitCommandEncoder()
        blit.fillBuffer_range_value_(
            span.segment.buffer, _range(span.offset + offset, nbytes), value
        )
        blit.endEncoding()
        command.commit()
        status, error = self.retire(command)
        if status == 5:
            message = f"metal fill failed: {error}"
            raise RuntimeError(message)

    def _copy(
        self, source: _Span, src_offset: int, dest: _Span, dst_offset: int, nbytes: int
    ) -> None:
        queue = self.make_queue()
        command = self.command_buffer(queue)
        blit = command.blitCommandEncoder()
        blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(
            source.segment.buffer,
            source.offset + src_offset,
            dest.segment.buffer,
            dest.offset + dst_offset,
            nbytes,
        )
        blit.endEncoding()
        command.commit()
        status, error = self.retire(command)
        if status == 5:
            message = f"metal copy failed: {error}"
            raise RuntimeError(message)


def _pair(result: Any) -> tuple[Any, Any]:
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, None


def _range(location: int, length: int) -> Any:
    import Foundation

    return Foundation.NSRange(location, length)


def _fill_shared(handle: Handle, offset: int, nbytes: int, value: int) -> None:
    chunk = 8 << 20
    if value == 0:
        blank = b"\x00" * min(chunk, nbytes)
    else:
        blank = bytes([value]) * min(chunk, nbytes)
    cursor = 0
    while cursor < nbytes:
        take = min(chunk, nbytes - cursor)
        memory.store(handle, offset + cursor, blank[:take])
        cursor += take


def _as_bytes(data: bytes | Any) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    import numpy as np

    if isinstance(data, np.ndarray):
        return np.ascontiguousarray(data).tobytes()
    return bytes(data)


memory = Memory()
_lease = 0
_lease_lock = threading.Lock()


def mint_owner(stem: str) -> str:
    """An owner string for one opening. The token is a process counter.

    A pipeline that still stamps ``id()`` keeps that string. ``sweep`` matches
    the whole string.
    """
    global _lease
    with _lease_lock:
        _lease += 1
        token = _lease
    return f"{stem}:{token}"


def bind(encoder: Any, index: int, handle: Any, offset: int = 0) -> None:
    memory.bind(encoder, index, handle, offset)


def bind_plane(attachments: Any, plane: int, handle: Any, offset: int = 0) -> None:
    memory.bind_plane(attachments, plane, handle, offset)


def bind_indirect(encoder: Any, handle: Any, offset: int, threads: Any) -> None:
    memory.bind_indirect(encoder, handle, offset, threads)
