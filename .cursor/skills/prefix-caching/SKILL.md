---
name: prefix-caching
description: Wire a Favlo pipeline to the shared SSD prefix cache in src/favlo/prefix.py. Use when building or changing a pipeline's sequence cache, prefix store, PrefixSession, or prefix adapter, or when a prompt should reuse tokens already applied on device or loaded from disk.
---

# Prefix cache

The file format lives in `src/favlo/prefix.py`. Read [docs/prefix.md](../../../docs/prefix.md) first. A pipeline imports that module. It does not copy the format, and it does not edit the module to special-case one model.

Another pipeline's adapter, region names, and pack functions are not a reference. Pack this pipeline's own buffers.

## Boundary

| Piece | Owner |
|---|---|
| Directory, header hash, page alignment, atomic replace, longest token prefix, copy-forward, 32 GiB budget | `favlo.prefix` |
| Which bytes are a region, and whether a header can continue a prompt | this pipeline's adapter |
| When the live sequence is written or loaded | `PrefixSession`, or this pipeline's `align` when it has extra device policy |

`longest()` reads every region into memory. The request path uses `matches()` and the adapter. A region name is unique inside one record. The store sorts names when it writes.

The directory is `~/.favlo/prefix/<model>/<checkpoint>/<hash>.bin`. `<model>` is this pipeline's flag stem. The tag is the active key/value mode. The digest is `checkpoint_digest(root)`. Pass `budget=DEFAULT_BUDGET`. A file whose header does not parse is removed when the store opens. A file for another model, digest, or tag is ignored.

## Adapter

Put `<package>/prefix.py` in this pipeline. Implement the four `PrefixAdapter` methods:

- `capture()` returns the regions for the tokens already applied. The device length is the count those regions cover.
- `install(header)` copies the header onto the device and returns false when a region is missing or the wrong length. Set the filled length to `len(header.tokens)`.
- `reset()` clears the device sequence.
- `accept(header, prompt)` is the only place that decides a header can continue `prompt`.

`read_regions(header, names)` reads those regions in one open. `header.region(name)` is the offset and length for a Metal read. Use that when the stored bytes are already the device layout. Pack on the host when the device buffer is strided by the full context: store positions `0 .. count-1` only.

An exact-length file is usable only when this pipeline can sample without stepping the last token again. A pipeline that must step the next token refuses `len(header.tokens) >= len(prompt)`. Stepping a token that is already in a recurrent state applies it twice.

## Session

A new pipeline uses `PrefixSession`. Open it after the device buffers exist:

```python
PrefixSession(
    PrefixStore(
        default_prefix_dir(),
        model="<pipeline>",
        digest=checkpoint_digest(runtime.root),
        tag=runtime.kv_cache,
        budget=DEFAULT_BUDGET,
    ),
    CacheAdapter(runtime),
)
```

Call order for one request:

1. `hit = session.align(prompt)`. Log `prefix load {hit} tokens` when `session.loaded` is true, then `cache hit {hit}` or `cache miss`.
2. Prefill `prompt[hit:]`. After the last prompt token is applied, `session.note(prompt)` and `session.flush()`.
3. Decode. `session.note` only the tokens the device has applied, then `session.flush()`.
4. On cancel before that prompt boundary, clear `session.tokens`, set `session.dirty` false, and `reset()`. After the boundary, `note` the applied tokens and `flush()`.
5. `close()` calls `flush()` while the device buffers still exist.

`flush()` writes only after `note()`. An `OSError` from the write is logged. The request still runs. The boundary that could not be written is not stored.

`align` keeps the device when `prompt` starts with the live tokens and is longer. Any other prompt flushes, resets, and installs the longest header `accept` allows. A later header is tried when `install` returns false or raises `OSError` or `RuntimeError`.

Use the store directly, and keep this pipeline's own `align`, when a load or a flush also has to preserve device state the session does not know. Flash Next does that for the expert atlas and for an exact file that still holds logits. The region names and the exact-file rule stay in that runtime. The bytes still go through `PrefixStore.save` and `PrefixStore.matches`.

## Docs and tests

This pipeline's `docs/<pipeline>/` note says which regions it stores and whether an exact-length file is sampled. Do not add that layout to `ARCHITECTURE.md` or to another pipeline's note.

A pack/unpack test uses numpy and imports only this package. It does not need the machine lock. A test that dispatches Metal, loads a checkpoint, or starts `serve` still takes the lock. Do not invent a cache-hit token rate.
