"""Finished prefills on disk.

A file is one boundary: the token ids, K and V, the indexer key, the recurrent
state, the convolution histories, and the last logits. The directory is
``~/.madmetal/prefix/<model>/<checkpoint>/<hash>.bin``. Regions that Metal
reads start on a 16 KB page. A crash leaves the previous complete file in place.
"""

from __future__ import annotations

import array
import fcntl
import hashlib
import os
import struct
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from qwen38_flash_next.allocate import preallocate
from qwen38_flash_next.formats import PAGE, PREFERRED_IO

MAGIC = b"MPX1"
FORMAT = 1
MODEL_ID = "qwen38-flash-next"
# One model and one checkpoint. The file just written is kept.
DEFAULT_BUDGET = 32 << 30
_HEADER_CAP = 8 << 20

Blob = bytes | memoryview


def default_prefix_dir() -> Path:
    return Path.home() / ".madmetal" / "prefix"


def stored_bytes(length: int) -> int:
    """Byte size of a region on disk. Empty regions take no pages."""
    if length <= 0:
        return 0
    return (length + PAGE - 1) & ~(PAGE - 1)


def checkpoint_digest(root: Path) -> bytes:
    """SHA-256 of the record manifest, or of ``config.json`` when it is absent."""
    manifest = root / "manifest.json"
    source = manifest if manifest.is_file() else root / "config.json"
    return hashlib.sha256(source.read_bytes()).digest()


def object_name(model: str, digest: bytes, tag: str, tokens: Sequence[int]) -> str:
    """Hex SHA-256 of the format, the model, the checkpoint, the cache tag, and the ids."""
    digest_bytes = bytes(digest)
    if len(digest_bytes) != 32:
        message = "checkpoint digest must be 32 bytes"
        raise ValueError(message)
    hasher = hashlib.sha256()
    hasher.update(struct.pack("<I", FORMAT))
    _hash_text(hasher, model)
    hasher.update(digest_bytes)
    _hash_text(hasher, tag)
    hasher.update(_token_bytes(tokens))
    return hasher.hexdigest()


@dataclass
class AttentionImage:
    layer: int
    key: Blob
    value: Blob
    key_tail: Blob
    value_tail: Blob
    indexer: Blob


@dataclass
class FixedImage:
    layer: int
    blob: Blob


@dataclass
class PrefixState:
    tokens: list[int]
    attention: list[AttentionImage]
    gdn: list[FixedImage]
    conv: list[FixedImage]
    ple: Blob
    logits: Blob


@dataclass(frozen=True)
class BlobRef:
    offset: int
    length: int


@dataclass(frozen=True)
class AttentionRef:
    layer: int
    key: BlobRef
    value: BlobRef
    key_tail: BlobRef
    value_tail: BlobRef
    indexer: BlobRef


@dataclass(frozen=True)
class FixedRef:
    layer: int
    blob: BlobRef


@dataclass(frozen=True)
class PrefixHeader:
    path: Path
    model: str
    digest: bytes
    tag: str
    tokens: tuple[int, ...]
    attention: tuple[AttentionRef, ...]
    gdn: tuple[FixedRef, ...]
    conv: tuple[FixedRef, ...]
    ple: BlobRef
    logits: BlobRef
    file_size: int


class PrefixStore:
    """One model and one checkpoint. Lookup reads headers, then one file."""

    def __init__(
        self,
        directory: Path,
        *,
        digest: bytes,
        tag: str,
        model: str = MODEL_ID,
        budget: int | None = None,
    ) -> None:
        digest_bytes = bytes(digest)
        if len(digest_bytes) != 32:
            message = "checkpoint digest must be 32 bytes"
            raise ValueError(message)
        self.model = model
        self.digest = digest_bytes
        self.tag = tag
        self.budget = budget
        self.directory = directory / model / digest_bytes.hex()
        self._clear_temps()

    def save(self, state: PrefixState) -> Path | None:
        if not state.tokens:
            return None
        self._clear_temps()
        self.directory.mkdir(parents=True, exist_ok=True)
        name = object_name(self.model, self.digest, self.tag, state.tokens)
        dest = self.directory / f"{name}.bin"
        existing = read_header(dest) if dest.is_file() else None
        if (
            existing is not None
            and self._owns(existing)
            and existing.tokens == tuple(state.tokens)
            and (existing.logits.length or not state.logits)
        ):
            self._evict(dest)
            return dest
        prior = self.match(state.tokens)
        if prior is not None and len(prior.tokens) == len(state.tokens):
            prior = None
        temporary = self.directory / f".{name}.{os.getpid()}.tmp"
        fd = os.open(temporary, os.O_CREAT | os.O_RDWR | os.O_EXCL, 0o644)
        try:
            _write_file(fd, self, state, prior)
            os.fsync(fd)
            os.replace(temporary, dest)
        finally:
            os.close(fd)
            temporary.unlink(missing_ok=True)
        self._evict(dest)
        return dest

    def matches(self, tokens: Sequence[int]) -> list[PrefixHeader]:
        """Stored headers whose token lists are prefixes of ``tokens``, longest first."""
        if not tokens or not self.directory.is_dir():
            return []
        found: list[PrefixHeader] = []
        for path in self.directory.glob("*.bin"):
            header = read_header(path)
            if header is None or not self._owns(header):
                continue
            count = len(header.tokens)
            if count == 0 or count > len(tokens) or tuple(tokens[:count]) != header.tokens:
                continue
            if count == len(tokens) and header.logits.length == 0:
                continue
            found.append(header)
        found.sort(key=lambda header: len(header.tokens), reverse=True)
        return found

    def match(self, tokens: Sequence[int]) -> PrefixHeader | None:
        found = self.matches(tokens)
        if not found:
            return None
        return found[0]

    def longest(self, tokens: list[int]) -> PrefixState | None:
        """The longest stored token list that is a prefix of ``tokens``.

        This reads every region into memory. The request path uses :meth:`match`
        and loads the regions with Metal IO.
        """
        header = self.match(tokens)
        if header is None:
            return None
        return read_state(header.path)

    def _owns(self, header: PrefixHeader) -> bool:
        if header.model != self.model or header.digest != self.digest or header.tag != self.tag:
            return False
        return header.path.stem == object_name(self.model, self.digest, self.tag, header.tokens)

    def _clear_temps(self) -> None:
        if not self.directory.is_dir():
            return
        for path in self.directory.glob(".*.tmp"):
            path.unlink(missing_ok=True)

    def _evict(self, keep: Path) -> None:
        budget = self.budget
        if budget is None or not self.directory.is_dir():
            return
        paths = [path for path in self.directory.glob("*.bin") if path.is_file()]
        paths.sort(key=lambda path: (path.stat().st_mtime, path.name))
        total = sum(path.stat().st_size for path in paths)
        for path in paths:
            if total <= budget:
                return
            if path == keep:
                continue
            size = path.stat().st_size
            path.unlink(missing_ok=True)
            total -= size


def read_header(path: Path) -> PrefixHeader | None:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            lead = handle.read(12)
            if len(lead) < 12 or lead[:4] != MAGIC:
                return None
            version, header_size = struct.unpack_from("<II", lead, 4)
            if (
                version != FORMAT
                or header_size < 44
                or header_size > _HEADER_CAP
                or header_size > size
            ):
                return None
            rest = handle.read(header_size - 12)
        if len(rest) != header_size - 12:
            return None
        blob = lead + rest
        if hashlib.sha256(blob[:-32]).digest() != blob[-32:]:
            return None
        header = _parse_header(path, blob, size)
    except (OSError, ValueError, UnicodeDecodeError, struct.error):
        return None
    if header.file_size != size or header.file_size % PAGE:
        return None
    return header


def read_tokens(path: Path) -> list[int] | None:
    header = read_header(path)
    if header is None:
        return None
    return list(header.tokens)


def read_state(path: Path) -> PrefixState | None:
    header = read_header(path)
    if header is None:
        return None
    try:
        return _materialize(header)
    except OSError:
        return None


def _materialize(header: PrefixHeader) -> PrefixState:
    fd = os.open(header.path, os.O_RDONLY)
    try:
        attention = [
            AttentionImage(
                image.layer,
                _read_ref(fd, image.key),
                _read_ref(fd, image.value),
                _read_ref(fd, image.key_tail),
                _read_ref(fd, image.value_tail),
                _read_ref(fd, image.indexer),
            )
            for image in header.attention
        ]
        gdn = [FixedImage(image.layer, _read_ref(fd, image.blob)) for image in header.gdn]
        conv = [FixedImage(image.layer, _read_ref(fd, image.blob)) for image in header.conv]
        ple = _read_ref(fd, header.ple)
        logits = _read_ref(fd, header.logits)
    finally:
        os.close(fd)
    return PrefixState(list(header.tokens), attention, gdn, conv, ple, logits)


def _hash_text(hasher: hashlib._Hash, text: str) -> None:
    raw = text.encode()
    hasher.update(struct.pack("<I", len(raw)))
    hasher.update(raw)


def _token_bytes(tokens: Sequence[int]) -> bytes:
    raw = array.array("I")
    for token in tokens:
        raw.append(token)
    if sys.byteorder != "little":
        raw.byteswap()
    return raw.tobytes()


def _tokens_from(blob: bytes) -> tuple[int, ...]:
    if len(blob) % 4:
        message = "token list is not a multiple of 4"
        raise ValueError(message)
    raw = array.array("I")
    raw.frombytes(blob)
    if sys.byteorder != "little":
        raw.byteswap()
    return tuple(raw)


def _write_file(
    fd: int, store: PrefixStore, state: PrefixState, prior: PrefixHeader | None
) -> None:
    attention = sorted(state.attention, key=lambda image: image.layer)
    gdn = sorted(state.gdn, key=lambda image: image.layer)
    conv = sorted(state.conv, key=lambda image: image.layer)
    token_blob = _token_bytes(state.tokens)
    header_len = _header_len(
        store.model, store.tag, len(token_blob), len(attention), len(gdn), len(conv)
    )
    cursor = stored_bytes(header_len)
    attention_refs: list[AttentionRef] = []
    for image in attention:
        ref, cursor = _attention_ref(image, cursor)
        attention_refs.append(ref)
    gdn_refs: list[FixedRef] = []
    for image in gdn:
        ref, cursor = _fixed_ref(image, cursor)
        gdn_refs.append(ref)
    conv_refs: list[FixedRef] = []
    for image in conv:
        ref, cursor = _fixed_ref(image, cursor)
        conv_refs.append(ref)
    ple = _take(len(state.ple), cursor)
    cursor = _advance(cursor, ple)
    logits = _take(len(state.logits), cursor)
    file_size = _advance(cursor, logits)
    header = _pack_header(
        store,
        token_blob,
        tuple(attention_refs),
        tuple(gdn_refs),
        tuple(conv_refs),
        ple,
        logits,
        file_size,
        header_len,
    )
    preallocate(fd, file_size)
    _bypass_cache(fd)
    _pwrite_aligned(fd, 0, header)
    prior_fd = None
    if prior is not None:
        prior_fd = os.open(prior.path, os.O_RDONLY)
        _bypass_cache(prior_fd)
    try:
        prior_attention = {image.layer: image for image in prior.attention} if prior else {}
        for image, ref in zip(attention, attention_refs, strict=True):
            old = prior_attention.get(image.layer)
            _write_region(fd, ref.key, image.key, prior_fd, old.key if old else None, True)
            _write_region(fd, ref.value, image.value, prior_fd, old.value if old else None, True)
            _write_region(fd, ref.key_tail, image.key_tail, prior_fd, None, False)
            _write_region(fd, ref.value_tail, image.value_tail, prior_fd, None, False)
            _write_region(
                fd, ref.indexer, image.indexer, prior_fd, old.indexer if old else None, True
            )
        for image, ref in zip(gdn, gdn_refs, strict=True):
            _write_region(fd, ref.blob, image.blob, prior_fd, None, False)
        for image, ref in zip(conv, conv_refs, strict=True):
            _write_region(fd, ref.blob, image.blob, prior_fd, None, False)
        _write_region(fd, ple, state.ple, prior_fd, None, False)
        _write_region(fd, logits, state.logits, prior_fd, None, False)
    finally:
        if prior_fd is not None:
            os.close(prior_fd)


def _header_len(
    model: str, tag: str, token_nbytes: int, attention: int, gdn: int, conv: int
) -> int:
    # magic, version, header size, model, digest, tag, count, tokens,
    # attention, gdn, conv, ple, logits, file size, hash.
    return (
        12
        + 4
        + len(model.encode())
        + 32
        + 4
        + len(tag.encode())
        + 4
        + token_nbytes
        + 4
        + attention * (4 + 5 * 16)
        + 4
        + gdn * 20
        + 4
        + conv * 20
        + 16
        + 16
        + 8
        + 32
    )


def _attention_ref(image: AttentionImage, cursor: int) -> tuple[AttentionRef, int]:
    key, cursor = _place(len(image.key), cursor)
    value, cursor = _place(len(image.value), cursor)
    key_tail, cursor = _place(len(image.key_tail), cursor)
    value_tail, cursor = _place(len(image.value_tail), cursor)
    indexer, cursor = _place(len(image.indexer), cursor)
    ref = AttentionRef(image.layer, key, value, key_tail, value_tail, indexer)
    return ref, cursor


def _fixed_ref(image: FixedImage, cursor: int) -> tuple[FixedRef, int]:
    blob, cursor = _place(len(image.blob), cursor)
    return FixedRef(image.layer, blob), cursor


def _place(length: int, cursor: int) -> tuple[BlobRef, int]:
    ref = _take(length, cursor)
    return ref, _advance(cursor, ref)


def _take(length: int, cursor: int) -> BlobRef:
    if length < 0:
        message = "prefix region length is negative"
        raise ValueError(message)
    if length == 0:
        return BlobRef(0, 0)
    if cursor % PAGE:
        message = "prefix region offset is not page-aligned"
        raise ValueError(message)
    return BlobRef(cursor, length)


def _advance(cursor: int, ref: BlobRef) -> int:
    if ref.length == 0:
        return cursor
    return ref.offset + stored_bytes(ref.length)


def _pack_header(
    store: PrefixStore,
    token_blob: bytes,
    attention: tuple[AttentionRef, ...],
    gdn: tuple[FixedRef, ...],
    conv: tuple[FixedRef, ...],
    ple: BlobRef,
    logits: BlobRef,
    file_size: int,
    header_len: int,
) -> bytes:
    buf = bytearray()
    buf += MAGIC
    buf += struct.pack("<I", FORMAT)
    buf += struct.pack("<I", header_len)
    _pack_text(buf, store.model)
    buf += store.digest
    _pack_text(buf, store.tag)
    buf += struct.pack("<I", len(token_blob) // 4)
    buf += token_blob
    buf += struct.pack("<I", len(attention))
    for ref in attention:
        buf += struct.pack("<I", ref.layer)
        for blob in (ref.key, ref.value, ref.key_tail, ref.value_tail, ref.indexer):
            buf += struct.pack("<QQ", blob.length, blob.offset)
    buf += struct.pack("<I", len(gdn))
    for ref in gdn:
        buf += struct.pack("<IQQ", ref.layer, ref.blob.length, ref.blob.offset)
    buf += struct.pack("<I", len(conv))
    for ref in conv:
        buf += struct.pack("<IQQ", ref.layer, ref.blob.length, ref.blob.offset)
    buf += struct.pack("<QQ", ple.length, ple.offset)
    buf += struct.pack("<QQ", logits.length, logits.offset)
    buf += struct.pack("<Q", file_size)
    if len(buf) + 32 != header_len:
        message = f"prefix header is {len(buf) + 32} bytes, planned {header_len}"
        raise RuntimeError(message)
    digest = hashlib.sha256(buf).digest()
    buf += digest
    return bytes(buf)


def _pack_text(buf: bytearray, text: str) -> None:
    raw = text.encode()
    buf += struct.pack("<I", len(raw))
    buf += raw


def _parse_header(path: Path, blob: bytes, size: int) -> PrefixHeader:
    pos = 12
    model, pos = _take_text(blob, pos)
    digest = blob[pos : pos + 32]
    pos += 32
    tag, pos = _take_text(blob, pos)
    count = struct.unpack_from("<I", blob, pos)[0]
    pos += 4
    token_blob = blob[pos : pos + count * 4]
    if len(token_blob) != count * 4:
        message = "prefix tokens ended early"
        raise ValueError(message)
    pos += count * 4
    attention_count = struct.unpack_from("<I", blob, pos)[0]
    pos += 4
    attention: list[AttentionRef] = []
    for _ in range(attention_count):
        layer = struct.unpack_from("<I", blob, pos)[0]
        pos += 4
        refs: list[BlobRef] = []
        for _blob in range(5):
            length, offset = struct.unpack_from("<QQ", blob, pos)
            pos += 16
            refs.append(_checked_ref(offset, length, size))
        attention.append(AttentionRef(layer, *refs))
    gdn, pos = _parse_fixed(blob, pos, size)
    conv, pos = _parse_fixed(blob, pos, size)
    ple_len, ple_off = struct.unpack_from("<QQ", blob, pos)
    pos += 16
    logits_len, logits_off = struct.unpack_from("<QQ", blob, pos)
    pos += 16
    file_size = struct.unpack_from("<Q", blob, pos)[0]
    pos += 8
    if pos + 32 != len(blob) or file_size != size:
        message = "prefix header size does not match the file"
        raise ValueError(message)
    return PrefixHeader(
        path,
        model,
        digest,
        tag,
        _tokens_from(token_blob),
        tuple(attention),
        gdn,
        conv,
        _checked_ref(ple_off, ple_len, size),
        _checked_ref(logits_off, logits_len, size),
        file_size,
    )


def _parse_fixed(blob: bytes, pos: int, size: int) -> tuple[tuple[FixedRef, ...], int]:
    count = struct.unpack_from("<I", blob, pos)[0]
    pos += 4
    images: list[FixedRef] = []
    for _ in range(count):
        layer, length, offset = struct.unpack_from("<IQQ", blob, pos)
        pos += 20
        images.append(FixedRef(layer, _checked_ref(offset, length, size)))
    return tuple(images), pos


def _checked_ref(offset: int, length: int, file_size: int) -> BlobRef:
    if length == 0:
        if offset != 0:
            message = "empty prefix region has an offset"
            raise ValueError(message)
        return BlobRef(0, 0)
    if offset < PAGE or offset % PAGE:
        message = "prefix region is not page-aligned"
        raise ValueError(message)
    end = offset + stored_bytes(length)
    if end > file_size:
        message = "prefix region runs past the file"
        raise ValueError(message)
    return BlobRef(offset, length)


def _take_text(blob: bytes, pos: int) -> tuple[str, int]:
    length = struct.unpack_from("<I", blob, pos)[0]
    pos += 4
    raw = blob[pos : pos + length]
    if len(raw) != length:
        message = "prefix text ended early"
        raise ValueError(message)
    return raw.decode(), pos + length


def _write_region(
    fd: int,
    ref: BlobRef,
    blob: Blob,
    prior_fd: int | None,
    prior: BlobRef | None,
    reuse: bool,
) -> None:
    if ref.length != len(blob):
        message = "prefix region length does not match the buffer"
        raise ValueError(message)
    stored = stored_bytes(ref.length)
    if stored == 0:
        return
    writer = _PageWriter(fd, ref.offset, stored)
    copied = 0
    if reuse and prior_fd is not None and prior is not None and 0 < prior.length <= ref.length:
        matched = True
        for pos, chunk in _logical_chunks(prior_fd, prior):
            if chunk != blob[pos : pos + len(chunk)]:
                matched = False
                break
            writer.push(chunk)
            copied = pos + len(chunk)
        if not matched:
            writer = _PageWriter(fd, ref.offset, stored)
            copied = 0
    if copied < ref.length:
        writer.push(blob[copied:])
    writer.finish()


def _logical_chunks(fd: int, ref: BlobRef):
    stored = stored_bytes(ref.length)
    pos = 0
    while pos < stored:
        take = min(PREFERRED_IO, stored - pos)
        got = os.pread(fd, take, ref.offset + pos)
        if len(got) != take:
            message = "prefix file ended early"
            raise OSError(message)
        if pos < ref.length:
            yield pos, got[: min(take, ref.length - pos)]
        pos += take


def _read_ref(fd: int, ref: BlobRef) -> bytes:
    if ref.length == 0:
        return b""
    out = bytearray()
    for _pos, chunk in _logical_chunks(fd, ref):
        out += chunk
    if len(out) != ref.length:
        message = "prefix region ended early"
        raise OSError(message)
    return bytes(out)


def _pwrite_aligned(fd: int, offset: int, blob: bytes) -> None:
    stored = stored_bytes(len(blob))
    region = bytearray(stored)
    region[: len(blob)] = blob
    pos = 0
    while pos < stored:
        take = min(PREFERRED_IO, stored - pos)
        os.pwrite(fd, memoryview(region)[pos : pos + take], offset + pos)
        pos += take


class _PageWriter:
    """Page-sized writes so an uncached descriptor can take the transfer."""

    def __init__(self, fd: int, offset: int, stored: int) -> None:
        if offset % PAGE or stored % PAGE or stored <= 0:
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
        while self.written + self.filled < self.stored:
            room = PREFERRED_IO - self.filled
            need = self.stored - self.written - self.filled
            take = min(room, need)
            self.buf[self.filled : self.filled + take] = b"\0" * take
            self.filled += take
            if self.filled == PREFERRED_IO or self.written + self.filled == self.stored:
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


def _bypass_cache(fd: int) -> None:
    flag = getattr(fcntl, "F_NOCACHE", None)
    if flag is not None:
        fcntl.fcntl(fd, flag, 1)
