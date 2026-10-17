"""SSD prefix cache shared by every pipeline.

The file stores token ids and named regions. A pipeline adapter packs that
pipeline's device buffers into those regions and copies a matching file back.
This module does not know a key/value layout, a recurrent state, or a model.

One directory is ``~/.madmetal/prefix/<model>/<checkpoint>/<hash>.bin``. The
checkpoint id is the hex SHA-256 of ``manifest.json``, or of ``config.json``
when the record has no manifest. The file name is the hex SHA-256 of the
format, the model id, that digest, the cache tag, and the token ids. Regions
that are read back start on a 16 KB page. The write goes to a temporary name
in the same directory and replaces the destination after the header hash is
in place. A crash leaves the previous complete file. When the new token list
extends a file already in the directory, a region whose stored bytes are a
prefix of the new blob is copied forward and the new tail is written after
it. The partition for one model and one checkpoint is kept to 32 GiB when a
budget is set. Past that, the oldest files are deleted. The file just written
stays.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

PAGE = 16384
PREFERRED_IO = 1 << 20
DEFAULT_BUDGET = 32 << 30
_MAGIC = b"MADP"
_VERSION = 1

_F_PREALLOCATE = 42
_F_ALLOCATECONTIG = 0x00000002
_F_ALLOCATEALL = 0x00000004
_F_PEOFPOSMODE = 3
_CHUNK = 1 << 30
_FALLBACK = 32 << 20
_FSTORE = struct.Struct("=Iiqqq")

Blob = bytes | memoryview


class PrefixAdapter(Protocol):
    """Pipeline side of one live sequence.

    ``capture`` reads the device buffers that match the tokens already applied.
    ``install`` copies a stored header back onto those buffers and returns
    false when the regions do not fit. ``reset`` clears the device sequence.
    ``accept`` decides whether a stored header can continue ``prompt``.
    """

    def capture(self) -> list[Region]: ...

    def install(self, header: PrefixHeader) -> bool: ...

    def reset(self) -> None: ...

    def accept(self, header: PrefixHeader, prompt: list[int]) -> bool: ...


@dataclass(frozen=True)
class Region:
    """One named blob. The name is unique inside a record."""

    name: str
    data: Blob


@dataclass(frozen=True)
class PrefixRecord:
    tokens: list[int]
    regions: tuple[Region, ...]

    def __init__(self, tokens: Sequence[int], regions: Sequence[Region]) -> None:
        names = [region.name for region in regions]
        if len(names) != len(set(names)):
            message = "prefix region names are not unique"
            raise ValueError(message)
        object.__setattr__(self, "tokens", [int(token) for token in tokens])
        object.__setattr__(self, "regions", tuple(regions))


@dataclass(frozen=True)
class RegionRef:
    name: str
    offset: int
    length: int


@dataclass(frozen=True)
class PrefixHeader:
    path: Path
    model: str
    digest: bytes
    tag: str
    tokens: tuple[int, ...]
    regions: tuple[RegionRef, ...]
    file_size: int

    def region(self, name: str) -> RegionRef | None:
        for ref in self.regions:
            if ref.name == name:
                return ref
        return None


class PrefixStore:
    """One model and one checkpoint. Lookup reads headers, then one file."""

    def __init__(
        self,
        directory: Path,
        *,
        model: str,
        digest: bytes,
        tag: str,
        budget: int | None = None,
    ) -> None:
        digest_bytes = bytes(digest)
        if len(digest_bytes) != 32:
            message = "checkpoint digest must be 32 bytes"
            raise ValueError(message)
        if not model or not tag:
            message = "prefix model and tag are required"
            raise ValueError(message)
        self.model = model
        self.digest = digest_bytes
        self.tag = tag
        self.budget = budget
        self.directory = directory / model / digest_bytes.hex()
        self._clear_temps()

    def save(self, record: PrefixRecord) -> Path | None:
        """Write ``record``. An empty token list writes nothing."""
        if not record.tokens:
            return None
        self._clear_temps()
        self.directory.mkdir(parents=True, exist_ok=True)
        name = object_name(self.model, self.digest, self.tag, record.tokens)
        dest = self.directory / f"{name}.bin"
        priors = [
            header
            for header in self.matches(record.tokens)
            if len(header.tokens) < len(record.tokens)
        ]
        prior = priors[0] if priors else None
        temporary = self.directory / f".{name}.{os.getpid()}.tmp"
        fd = os.open(temporary, os.O_CREAT | os.O_RDWR | os.O_EXCL, 0o644)
        prior_fd = os.open(prior.path, os.O_RDONLY) if prior is not None else None
        try:
            _bypass_cache(fd)
            _write_file(fd, self, record, prior, prior_fd)
            os.fsync(fd)
            os.replace(temporary, dest)
        finally:
            os.close(fd)
            if prior_fd is not None:
                os.close(prior_fd)
            temporary.unlink(missing_ok=True)
        self._evict(dest)
        return dest

    def matches(self, tokens: Sequence[int]) -> list[PrefixHeader]:
        """Headers whose token lists are prefixes of ``tokens``, longest first."""
        if not tokens or not self.directory.is_dir():
            return []
        found: list[PrefixHeader] = []
        wanted = [int(token) for token in tokens]
        for path in self.directory.glob("*.bin"):
            header = read_header(path)
            if header is None or not self._owns(header):
                continue
            count = len(header.tokens)
            if count == 0 or count > len(wanted) or tuple(wanted[:count]) != header.tokens:
                continue
            found.append(header)
        found.sort(key=lambda header: len(header.tokens), reverse=True)
        return found

    def longest(self, tokens: Sequence[int]) -> PrefixRecord | None:
        """The longest stored prefix, with every region read into memory."""
        found = self.matches(tokens)
        if not found:
            return None
        return _materialize(found[0])

    def _owns(self, header: PrefixHeader) -> bool:
        return (
            header.model == self.model
            and header.digest == self.digest
            and header.tag == self.tag
        )

    def _evict(self, keep: Path) -> None:
        if self.budget is None:
            return
        files = [path for path in self.directory.glob("*.bin") if path.is_file()]
        files.sort(key=lambda path: path.stat().st_mtime)
        total = sum(path.stat().st_size for path in files)
        for path in files:
            if total <= self.budget or path == keep:
                continue
            size = path.stat().st_size
            path.unlink(missing_ok=True)
            total -= size

    def _clear_temps(self) -> None:
        if not self.directory.is_dir():
            return
        for path in self.directory.glob(".*.tmp"):
            path.unlink(missing_ok=True)
        for path in self.directory.glob("*.bin"):
            if read_header(path) is None:
                path.unlink(missing_ok=True)


class PrefixSession:
    """Live token list for one pipeline.

    A prompt that extends the live tokens keeps the device. Any other prompt
    writes the live sequence, clears the device, and loads the longest stored
    prefix the adapter accepts. The adapter's ``accept`` is the boundary: a
    pipeline that can continue an exact token list says so, and one that must
    step another token refuses a file as long as the prompt.
    """

    def __init__(self, store: PrefixStore, adapter: PrefixAdapter) -> None:
        self.store = store
        self.adapter = adapter
        self.tokens: list[int] = []
        self.dirty = False
        self.loaded = False

    def align(self, prompt: Sequence[int]) -> int:
        """Tokens already applied. ``loaded`` is true when they came from a file."""
        wanted = [int(token) for token in prompt]
        self.loaded = False
        if (
            self.tokens
            and wanted[: len(self.tokens)] == self.tokens
            and len(wanted) > len(self.tokens)
        ):
            return len(self.tokens)
        try:
            self.flush()
        except OSError:
            pass
        self.adapter.reset()
        self.tokens = []
        self.dirty = False
        for header in self.store.matches(wanted):
            if not self.adapter.accept(header, wanted):
                continue
            try:
                installed = self.adapter.install(header)
            except (OSError, RuntimeError):
                self.adapter.reset()
                continue
            if not installed:
                continue
            self.tokens = list(header.tokens)
            self.dirty = False
            self.loaded = True
            return len(self.tokens)
        return 0

    def note(self, tokens: Sequence[int]) -> None:
        """The device now matches ``tokens`` and the file is behind it."""
        self.tokens = [int(token) for token in tokens]
        self.dirty = True
        self.loaded = False

    def flush(self) -> Path | None:
        """Write the live sequence when ``note`` has moved it. Otherwise do nothing."""
        if not self.tokens or not self.dirty:
            return None
        path = self.store.save(PrefixRecord(self.tokens, self.adapter.capture()))
        if path is not None:
            self.dirty = False
        return path


def default_prefix_dir() -> Path:
    return Path.home() / ".madmetal" / "prefix"


def checkpoint_digest(root: Path) -> bytes:
    """SHA-256 of ``manifest.json``, or of ``config.json`` when there is no manifest."""
    manifest = root / "manifest.json"
    source = manifest if manifest.is_file() else root / "config.json"
    return hashlib.sha256(source.read_bytes()).digest()


def object_name(model: str, digest: bytes, tag: str, tokens: Sequence[int]) -> str:
    digest_bytes = bytes(digest)
    if len(digest_bytes) != 32:
        message = "checkpoint digest must be 32 bytes"
        raise ValueError(message)
    hashed = hashlib.sha256()
    hashed.update(_VERSION.to_bytes(4, "little"))
    hashed.update(model.encode())
    hashed.update(digest_bytes)
    hashed.update(tag.encode())
    hashed.update(struct.pack(f"<{len(tokens)}I", *[int(token) for token in tokens]))
    return hashed.hexdigest()


def stored_bytes(length: int) -> int:
    """Page multiple that holds ``length``. Zero stays zero."""
    if length <= 0:
        return 0
    return (length + PAGE - 1) & ~(PAGE - 1)


def read_header(path: Path) -> PrefixHeader | None:
    try:
        blob = path.read_bytes()
    except OSError:
        return None
    try:
        return _parse_header(path, blob)
    except (ValueError, struct.error, UnicodeDecodeError):
        return None


def read_tokens(path: Path) -> list[int] | None:
    header = read_header(path)
    if header is None:
        return None
    return list(header.tokens)


def read_regions(header: PrefixHeader, names: Sequence[str]) -> dict[str, bytes] | None:
    """One read of each named region. ``None`` when a name is missing."""
    refs: list[RegionRef] = []
    for name in names:
        ref = header.region(name)
        if ref is None:
            return None
        refs.append(ref)
    fd = os.open(header.path, os.O_RDONLY)
    try:
        return {ref.name: _read_ref(fd, ref) for ref in refs}
    finally:
        os.close(fd)


def _parse_header(path: Path, blob: bytes) -> PrefixHeader:
    if len(blob) < 12 or blob[:4] != _MAGIC:
        message = "prefix magic does not match"
        raise ValueError(message)
    version, header_size = struct.unpack_from("<II", blob, 4)
    if version != _VERSION or header_size > len(blob) or header_size < 44:
        message = "prefix header size does not match"
        raise ValueError(message)
    digest = hashlib.sha256(blob[: header_size - 32]).digest()
    if blob[header_size - 32 : header_size] != digest:
        message = "prefix header hash does not match"
        raise ValueError(message)
    pos = 12
    model, pos = _take_text(blob, pos)
    tag, pos = _take_text(blob, pos)
    file_digest = blob[pos : pos + 32]
    pos += 32
    count = struct.unpack_from("<I", blob, pos)[0]
    pos += 4
    tokens = struct.unpack_from(f"<{count}I", blob, pos)
    pos += 4 * count
    region_count = struct.unpack_from("<I", blob, pos)[0]
    pos += 4
    regions: list[RegionRef] = []
    file_size = struct.unpack_from("<Q", blob, header_size - 40)[0]
    if file_size != len(blob):
        message = "prefix file size does not match the header"
        raise ValueError(message)
    for _ in range(region_count):
        name, pos = _take_text(blob, pos)
        length, offset = struct.unpack_from("<QQ", blob, pos)
        pos += 16
        regions.append(RegionRef(name, *_checked(offset, length, file_size)))
    if pos != header_size - 40:
        message = "prefix header ended early"
        raise ValueError(message)
    return PrefixHeader(path, model, file_digest, tag, tokens, tuple(regions), file_size)


def _checked(offset: int, length: int, file_size: int) -> tuple[int, int]:
    if length == 0:
        if offset != 0:
            message = "empty prefix region has an offset"
            raise ValueError(message)
        return 0, 0
    if offset < PAGE or offset % PAGE:
        message = "prefix region is not page-aligned"
        raise ValueError(message)
    end = offset + stored_bytes(length)
    if end > file_size:
        message = "prefix region runs past the file"
        raise ValueError(message)
    return offset, length


def _take_text(blob: bytes, pos: int) -> tuple[str, int]:
    length = struct.unpack_from("<H", blob, pos)[0]
    pos += 2
    raw = blob[pos : pos + length]
    if len(raw) != length:
        message = "prefix text ended early"
        raise ValueError(message)
    return raw.decode(), pos + length


def _header_size(model: str, tag: str, tokens: Sequence[int], names: Sequence[str]) -> int:
    size = 12
    size += 2 + len(model.encode())
    size += 2 + len(tag.encode())
    size += 32
    size += 4 + 4 * len(tokens)
    size += 4
    for name in names:
        size += 2 + len(name.encode()) + 16
    size += 8 + 32
    return size


def _write_file(
    fd: int,
    store: PrefixStore,
    record: PrefixRecord,
    prior: PrefixHeader | None,
    prior_fd: int | None,
) -> None:
    ordered = tuple(sorted(record.regions, key=lambda region: region.name))
    names = [region.name for region in ordered]
    header_size = _header_size(store.model, store.tag, record.tokens, names)
    cursor = stored_bytes(header_size)
    refs: list[RegionRef] = []
    for region in ordered:
        length = len(region.data)
        if length == 0:
            refs.append(RegionRef(region.name, 0, 0))
            continue
        refs.append(RegionRef(region.name, cursor, length))
        cursor += stored_bytes(length)
    file_size = cursor
    header = _header_bytes(store, record.tokens, refs, header_size, file_size)
    if len(header) != header_size:
        message = f"prefix header is {len(header)} bytes, planned {header_size}"
        raise RuntimeError(message)
    _preallocate(fd, file_size)
    prior_regions = {ref.name: ref for ref in prior.regions} if prior is not None else {}
    for region, ref in zip(ordered, refs, strict=True):
        _write_region(fd, ref, region.data, prior_fd, prior_regions.get(region.name))
    os.pwrite(fd, header, 0)


def _header_bytes(
    store: PrefixStore,
    tokens: Sequence[int],
    regions: Sequence[RegionRef],
    header_size: int,
    file_size: int,
) -> bytes:
    parts = [
        _MAGIC,
        struct.pack("<II", _VERSION, header_size),
        _text(store.model),
        _text(store.tag),
        store.digest,
        struct.pack("<I", len(tokens)),
        struct.pack(f"<{len(tokens)}I", *[int(token) for token in tokens]),
        struct.pack("<I", len(regions)),
    ]
    for ref in regions:
        parts.append(_text(ref.name))
        parts.append(struct.pack("<QQ", ref.length, ref.offset))
    parts.append(struct.pack("<Q", file_size))
    blob = b"".join(parts)
    if len(blob) + 32 != header_size:
        message = "prefix header size does not match the planned size"
        raise RuntimeError(message)
    return blob + hashlib.sha256(blob).digest()


def _text(value: str) -> bytes:
    raw = value.encode()
    if len(raw) > 0xFFFF:
        message = f"prefix text {value!r} is too long"
        raise ValueError(message)
    return struct.pack("<H", len(raw)) + raw


def _write_region(
    fd: int,
    ref: RegionRef,
    blob: Blob,
    prior_fd: int | None,
    prior: RegionRef | None,
) -> None:
    if ref.length != len(blob):
        message = "prefix region length does not match the buffer"
        raise ValueError(message)
    stored = stored_bytes(ref.length)
    if stored == 0:
        return
    copied = 0
    if prior_fd is not None and prior is not None and _same_prefix(prior_fd, prior, blob):
        copied = prior.length & ~(PAGE - 1)
        _copy_pages(prior_fd, prior.offset, fd, ref.offset, copied)
    writer = _PageWriter(fd, ref.offset + copied, stored - copied) if copied < stored else None
    if writer is not None:
        if copied < ref.length:
            writer.push(memoryview(blob)[copied:])
        writer.finish()


def _same_prefix(fd: int, prior: RegionRef, blob: Blob) -> bool:
    if prior.length <= 0 or prior.length > len(blob):
        return False
    view = memoryview(blob)
    pos = 0
    while pos < prior.length:
        take = min(PREFERRED_IO, prior.length - pos)
        got = os.pread(fd, take, prior.offset + pos)
        if got != view[pos : pos + take]:
            return False
        pos += take
    return True


def _copy_pages(src: int, src_at: int, dst: int, dst_at: int, nbytes: int) -> None:
    pos = 0
    while pos < nbytes:
        take = min(PREFERRED_IO, nbytes - pos)
        got = os.pread(src, take, src_at + pos)
        if len(got) != take:
            message = "prefix file ended early"
            raise OSError(message)
        os.pwrite(dst, got, dst_at + pos)
        pos += take


def _materialize(header: PrefixHeader) -> PrefixRecord:
    fd = os.open(header.path, os.O_RDONLY)
    try:
        regions = [Region(ref.name, _read_ref(fd, ref)) for ref in header.regions]
    finally:
        os.close(fd)
    return PrefixRecord(list(header.tokens), regions)


def _read_ref(fd: int, ref: RegionRef) -> bytes:
    if ref.length == 0:
        return b""
    got = os.pread(fd, ref.length, ref.offset)
    if len(got) != ref.length:
        message = f"prefix region {ref.name} ended early"
        raise OSError(message)
    return got


class _PageWriter:
    """Page-sized writes so an uncached descriptor can take the transfer."""

    def __init__(self, fd: int, offset: int, stored: int) -> None:
        if stored < 0 or offset % PAGE or (stored and stored % PAGE):
            message = "prefix write is not page-aligned"
            raise ValueError(message)
        self.fd = fd
        self.dest = offset
        self.stored = stored
        self.buf = bytearray(PREFERRED_IO)
        self.filled = 0
        self.written = 0

    def push(self, data: Blob) -> None:
        view = memoryview(data)
        index = 0
        while index < len(view):
            room = PREFERRED_IO - self.filled
            take = min(room, len(view) - index)
            self.buf[self.filled : self.filled + take] = view[index : index + take]
            self.filled += take
            index += take
            if self.filled == PREFERRED_IO:
                self._emit(PREFERRED_IO)

    def finish(self) -> None:
        if self.stored == 0:
            return
        while self.written + self.filled < self.stored:
            room = PREFERRED_IO - self.filled
            need = self.stored - self.written - self.filled
            take = min(room, need)
            self.buf[self.filled : self.filled + take] = b"\0" * take
            self.filled += take
            if self.filled == PREFERRED_IO or self.written + self.filled == self.stored:
                self._emit(self.filled)
        if self.filled:
            self._emit(self.filled)
        if self.written != self.stored:
            message = "prefix region ended short"
            raise OSError(message)

    def _emit(self, count: int) -> None:
        if count % PAGE:
            message = "prefix write length is not a multiple of the page"
            raise OSError(message)
        os.pwrite(self.fd, memoryview(self.buf)[:count], self.dest)
        self.dest += count
        self.written += count
        self.filled = 0


def _preallocate(fd: int, nbytes: int) -> None:
    if nbytes <= 0 or nbytes % PAGE:
        message = f"allocation {nbytes} is not a positive multiple of {PAGE}"
        raise ValueError(message)
    os.ftruncate(fd, 0)
    filled = 0
    while filled < nbytes:
        take = min(_CHUNK, nbytes - filled)
        err = _claim(fd, take)
        if err is None:
            filled += take
            os.ftruncate(fd, filled)
            continue
        err = _claim_slices(fd, filled, nbytes)
        if err is not None:
            raise OSError(err, f"preallocate failed for {nbytes - filled} bytes")
        return


def _claim(fd: int, nbytes: int) -> int | None:
    err = _alloc_piece(fd, nbytes, contig=True)
    if err is None:
        return None
    other = _alloc_piece(fd, nbytes, contig=False)
    if other is None:
        return None
    return other or err


def _claim_slices(fd: int, filled: int, nbytes: int) -> int | None:
    while filled < nbytes:
        take = min(_FALLBACK, nbytes - filled)
        err = _claim(fd, take)
        if err is not None:
            return err
        filled += take
        os.ftruncate(fd, filled)
    return None


def _alloc_piece(fd: int, nbytes: int, *, contig: bool) -> int | None:
    flags = _F_ALLOCATEALL
    if contig:
        flags |= _F_ALLOCATECONTIG
    try:
        out = fcntl.fcntl(fd, _F_PREALLOCATE, _FSTORE.pack(flags, _F_PEOFPOSMODE, 0, nbytes, 0))
    except OSError as exc:
        return exc.errno or errno.EIO
    _flags, _mode, _off, _length, allocated = _FSTORE.unpack(out)
    if int(allocated) >= nbytes:
        return None
    return errno.ENOSPC


def _bypass_cache(fd: int) -> None:
    flag = getattr(fcntl, "F_NOCACHE", None)
    if flag is not None:
        fcntl.fcntl(fd, flag, 1)
