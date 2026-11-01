"""Contiguous, page-aligned files for uncached SSD reads.

Apple's uncached path (`F_NOCACHE`, and Metal's uncached IO handle) keeps the
transfer off the bounce buffer when the file offset, the length, and the
destination are multiples of the VM page. `F_PREALLOCATE` with
`F_ALLOCATECONTIG` asks APFS for that storage as real extents, so a logical
read is not a chain of holes. A very large file may come back as several
1 GiB runs; each run is still one sequential DMA.
"""

from __future__ import annotations

import errno
import fcntl
import mmap
import os
import struct
from pathlib import Path

from qwen38_flash_next.formats import PAGE, PREFERRED_IO

# bsd/sys/fcntl.h. fstore is 32 bytes; log2phys is packed to 20.
_F_PREALLOCATE = 42
_F_ALLOCATECONTIG = 0x00000002
_F_ALLOCATEALL = 0x00000004
_F_PEOFPOSMODE = 3
_F_LOG2PHYS_EXT = 65
_CHUNK = 1 << 30
_FALLBACK = 32 << 20
_FSTORE = struct.Struct("=Iiqqq")
_LOG2PHYS = struct.Struct("=Iqq")


def round_preferred(nbytes: int) -> int:
    """Round a byte count up to the volume's preferred transfer size."""
    align = PREFERRED_IO
    return (nbytes + align - 1) & ~(align - 1)


def preallocate(fd: int, nbytes: int) -> None:
    """Allocate ``nbytes`` at the end of an empty file, then set that length.

    The file is truncated to zero first. Contiguous 1 GiB pieces are requested
    before a non-contiguous fallback, so a failed full-file request does not
    leave a sparse hole. A piece the volume refuses is claimed again in 32 MiB
    runs. Each piece is claimed with ``ftruncate`` so the next physical-end
    request starts after it.
    """
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


def contiguous_bytes(path: Path, offset: int, nbytes: int) -> tuple[int, int]:
    """Physical bytes and device offset at ``offset``. A hole returns device −1."""
    fd = os.open(path, os.O_RDONLY)
    try:
        out = fcntl.fcntl(fd, _F_LOG2PHYS_EXT, _LOG2PHYS.pack(0, nbytes, offset))
        _flags, contig, device = _LOG2PHYS.unpack(out)
        return int(contig), int(device)
    finally:
        os.close(fd)


def seal_contiguous(path: Path, nbytes: int) -> None:
    """Replace ``path`` with a preallocated copy of its first ``nbytes``."""
    if nbytes % PAGE:
        message = f"sealed length {nbytes} is not a multiple of {PAGE}"
        raise ValueError(message)
    tmp = path.with_name(path.name + ".seal")
    src = os.open(path, os.O_RDONLY)
    dst = os.open(tmp, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        preallocate(dst, nbytes)
        buf = mmap.mmap(-1, PREFERRED_IO)
        try:
            offset = 0
            while offset < nbytes:
                take = min(PREFERRED_IO, nbytes - offset)
                got = os.pread(src, take, offset)
                if len(got) != take:
                    message = f"short read sealing {path.name} at {offset}"
                    raise OSError(message)
                buf[:take] = got
                os.pwrite(dst, buf[:take], offset)
                offset += take
        finally:
            buf.close()
        os.fsync(dst)
    except Exception:
        os.close(src)
        os.close(dst)
        tmp.unlink(missing_ok=True)
        raise
    os.close(src)
    os.close(dst)
    os.replace(tmp, path)


def _claim(fd: int, nbytes: int) -> int | None:
    """Reserve ``nbytes`` at the physical end. ``None`` means the reservation held."""
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
    """``None`` on success. Otherwise the errno, with ``ENOSPC`` for a short grant."""
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
