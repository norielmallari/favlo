"""Text, lyric, timbre, DiT, and Oobleck decode. Every neural op is a shader."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ace_step_15_turbo.hold import ClipHold, ClipPaused, interrupt
from ace_step_15_turbo.pcm import wav_bytes
from ace_step_15_turbo.record import Plane, plane
from ace_step_15_turbo.schedule import (
    HOP,
    LATENT_RATE,
    SAMPLE_RATE,
    STEPS,
    TEXT_LIMIT,
    TIMBRE_FRAMES,
    latent_frames,
    timestep_schedule,
)
from ace_step_15_turbo.schedule import condition_text

HEAD_DIM = 128
SLIDING = 128
CAUSAL = -2
PEAK = 10.0 ** (-1.0 / 20.0)
Progress = Callable[[str], None]


def generate(
    gpu: Any,
    weights: dict[str, Plane],
    tokenizer: Any,
    prompt: str,
    lyrics: str,
    seconds: float,
    seed: int | None,
    vocal_language: str,
    bpm: int | None,
    keyscale: str | None,
    timesignature: str | None,
    progress: Progress | None = None,
    hold: ClipHold | None = None,
) -> bytes:
    """One clip. The return is a 48 kHz stereo WAV.

    ``hold`` is a finished prefix. ``reached`` is stage order, so those stages
    are not run again. A reached stage with no host array is an error. A
    preempt or cancel at a boundary raises ``ClipPaused``.
    """
    if hold is None:
        found = getattr(progress, "hold", None)
        if isinstance(found, ClipHold):
            hold = found
    if interrupt(progress) == "cancel":
        raise ClipPaused("cancel", ClipHold(stage="start"))
    frames = latent_frames(seconds)
    if frames % 2 != 0:
        frames += 1
    audio = _array(hold, "decode", "audio", "the waveform")
    if audio is not None:
        return _wav(audio, seconds, progress)
    noise_host = _array(hold, "step", "noise", "the latent")
    if noise_host is not None and hold is not None and hold.step >= STEPS:
        noise = gpu.clip(gpu.f32(noise_host))
    else:
        if hold is None:
            _pause(progress, ClipHold(stage="start"))
        caption, lyric = condition_text(
            prompt,
            lyrics,
            seconds,
            vocal_language,
            bpm=bpm,
            keyscale=keyscale,
            timesignature=timesignature,
        )
        text_host = _array(hold, "text", "text", "the text states")
        if text_host is not None:
            text = gpu.clip(gpu.f32(text_host))
            text_len = int(text_host.shape[0])
        else:
            text_ids = _ids(tokenizer, caption, TEXT_LIMIT)
            text = _text(gpu, weights, text_ids, progress)
            text_len = len(text_ids)
            _boundary_device(gpu, progress, "text", text=(text, (text_len, 2048)))
        lyric_host = _array(hold, "lyric", "lyric", "the lyric states")
        if lyric_host is not None:
            lyric_state = gpu.clip(gpu.f32(lyric_host))
            lyric_len = int(lyric_host.shape[0])
        else:
            lyric_ids = _ids(tokenizer, lyric, 2048)
            lyric_state = _lyric(gpu, weights, lyric_ids, progress)
            lyric_len = len(lyric_ids)
            _boundary_device(
                gpu,
                progress,
                "lyric",
                text=(text, (text_len, 2048)),
                lyric=(lyric_state, (lyric_len, 2048)),
            )
        timbre_host = _array(hold, "timbre", "timbre", "the timbre vector")
        if timbre_host is not None:
            timbre = gpu.clip(gpu.f32(timbre_host))
        else:
            timbre = _timbre(gpu, weights, progress)
            _boundary_device(
                gpu,
                progress,
                "timbre",
                text=(text, (text_len, 2048)),
                lyric=(lyric_state, (lyric_len, 2048)),
                timbre=(timbre, (2048,)),
            )
        cond, cond_len = _pack(gpu, lyric_state, lyric_len, timbre, text, text_len)
        cond = _apply(gpu, cond, cond_len, plane(weights, "dit.condition_embedder.weight"))
        gpu.clip(cond)
        cross = _cross_cache(gpu, weights, cond, cond_len)
        gpu.reclaim(cond)
        silence = plane(weights, "silence_latent")
        taken = gpu.buffer(frames * 64 * 4)
        gpu.span(silence.data, taken, frames * 64, 0, 0)
        source = gpu.clip(gpu.transpose(taken, frames, 64))
        mask = gpu.clip(gpu.filled(1.0, 64 * frames))
        if noise_host is not None and hold is not None:
            noise = gpu.clip(gpu.f32(noise_host))
            start = hold.step
        else:
            rng = np.random.default_rng(0 if seed is None else seed)
            noise = gpu.clip(gpu.f32(rng.standard_normal((64, frames), dtype=np.float32)))
            start = 0
        tokens = frames // 2
        cos, sin = gpu.rope_table(tokens, HEAD_DIM, 1_000_000.0)
        gpu.clip(cos)
        gpu.clip(sin)
        nodes = list(timestep_schedule(STEPS)) + [0.0]
        hidden_size = 2560
        kept = {
            "text": (text, (text_len, 2048)),
            "lyric": (lyric_state, (lyric_len, 2048)),
            "timbre": (timbre, (2048,)),
        }
        for step in range(start, STEPS):
            packed = gpu.pack_latent(source, mask, noise, 64, frames)
            proj_in = plane(weights, "dit.proj_in_conv.weight")
            hidden, length = gpu.conv(
                packed,
                proj_in.data,
                proj_in.scale,
                proj_in.bias,
                192,
                2560,
                2,
                proj_in.inner,
                frames,
                0,
                2,
            )
            if length != tokens:
                message = f"patch length {length} is not {tokens}"
                raise RuntimeError(message)
            hidden = gpu.transpose(hidden, 2560, tokens)
            shift, scale, temb = _timestep(gpu, weights, float(nodes[step]))
            for index in range(32):
                hidden = _dit_layer(
                    gpu,
                    weights,
                    hidden,
                    tokens,
                    cos,
                    sin,
                    shift,
                    scale,
                    cross[index],
                    cond_len,
                    index,
                )
                gpu.reclaim(hidden)
                if progress is not None:
                    progress("dit")
            final = plane(weights, "dit.scale_shift_table")
            out_shift = _slice(gpu, final.data, 0, hidden_size)
            out_scale = _slice(gpu, final.data, hidden_size, hidden_size)
            gpu.add_vectors(out_shift, temb, hidden_size)
            gpu.add_vectors(out_scale, temb, hidden_size)
            normed = gpu.norm(hidden, plane(weights, "dit.norm_out.weight").data, tokens, hidden_size)
            hidden = gpu.adaln(normed, out_scale, out_shift, tokens, hidden_size)
            hidden = gpu.transpose(hidden, tokens, hidden_size)
            proj_out = plane(weights, "dit.proj_out_conv.weight")
            velocity, out_len = gpu.conv_transpose(
                hidden,
                proj_out.data,
                proj_out.scale,
                proj_out.bias,
                2560,
                64,
                2,
                proj_out.inner,
                tokens,
                0,
                2,
            )
            if out_len < frames:
                message = f"unpatch length {out_len} is shorter than {frames}"
                raise RuntimeError(message)
            gpu.euler(noise, velocity, float(nodes[step + 1] - nodes[step]), 64 * frames)
            gpu.reclaim()
            _boundary_device(gpu, progress, "step", step=step + 1, noise=(noise, (64, frames)), **kept)
    wave = _decode(gpu, weights, noise, frames, progress)
    gpu.normalize_peak(wave, 2 * frames * HOP, PEAK)
    gpu.reclaim(wave)
    gpu.wait()
    audio = gpu.read(wave, 2 * frames * HOP, np.dtype(np.float32)).reshape(2, frames * HOP)
    return _wav(audio, seconds, progress)


def _wav(audio: np.ndarray, seconds: float, progress: Progress | None = None) -> bytes:
    if audio.shape[1] != int(seconds * SAMPLE_RATE) and abs(audio.shape[1] - seconds * SAMPLE_RATE) > HOP:
        audio = audio[:, : int(round(seconds * SAMPLE_RATE))]
    _pause(progress, ClipHold(stage="decode", audio=audio))
    return wav_bytes(audio)


def _array(hold: ClipHold | None, stage: str, name: str, label: str) -> np.ndarray | None:
    """The host output of a finished stage. A later stage requires it."""
    if hold is None or not hold.reached(stage):
        return None
    array = getattr(hold, name)
    if not isinstance(array, np.ndarray):
        message = f"the paused {stage} is missing {label}"
        raise RuntimeError(message)
    return array


def _pause(progress: Progress | None, hold: ClipHold) -> None:
    """Stop when the listener asked. Cancel keeps no checkpoint."""
    reason = interrupt(progress)
    if reason is None:
        return
    if reason == "cancel":
        raise ClipPaused("cancel", ClipHold(stage="start"))
    hold.done = _done(progress)
    raise ClipPaused(reason, hold)


def _boundary_device(gpu: Any, progress: Progress | None, stage: str, step: int = 0, **named: tuple[Any, tuple[int, ...]]) -> None:
    """Copy the finished device buffers, then stop when the listener asked."""
    reason = interrupt(progress)
    if reason is None:
        return
    if reason == "cancel":
        raise ClipPaused("cancel", ClipHold(stage="start"))
    gpu.wait()
    host: dict[str, np.ndarray] = {}
    for name, (buf, shape) in named.items():
        count = int(np.prod(shape))
        host[name] = gpu.read(buf, count, np.dtype(np.float32)).reshape(shape)
    raise ClipPaused(reason, ClipHold(stage=stage, step=step, done=_done(progress), **host))


def _done(progress: Progress | None) -> int:
    return int(getattr(progress, "done", 0))


def _ids(tokenizer: Any, text: str, limit: int) -> list[int]:
    eos = tokenizer.token_to_id("<|endoftext|>")
    ids = list(tokenizer.encode(text, add_special_tokens=True).ids)
    if len(ids) > limit:
        if eos is not None and ids and ids[-1] == eos:
            ids = ids[: limit - 1] + [eos]
        else:
            ids = ids[:limit]
    if not ids:
        message = "tokenizer produced no tokens"
        raise RuntimeError(message)
    return ids


def _apply(gpu: Any, values: Any, rows: int, item: Plane) -> Any:
    return gpu.linear(
        values, rows, item.cols, item.data, item.scale, item.rows, item.inner, item.bias, item.kind
    )


def _slice(gpu: Any, values: Any, offset: int, count: int) -> Any:
    dest = gpu.buffer(count * 4)
    gpu.span(values, dest, count, offset, 0)
    return dest


def _text(gpu: Any, weights: dict[str, Plane], ids: list[int], progress: Progress | None) -> Any:
    table = plane(weights, "text.embed_tokens.weight")
    hidden = gpu.gather(table.data, table.scale, gpu.upload(np.asarray(ids, dtype=np.int32)), len(ids), table.cols, table.inner)
    cos, sin = gpu.rope_table(len(ids), HEAD_DIM, 1_000_000.0)
    for index in range(28):
        hidden = _encoder_layer(
            gpu,
            weights,
            hidden,
            len(ids),
            cos,
            sin,
            CAUSAL,
            f"text.layers.{index}",
            1024,
            3072,
            16,
            8,
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "self_attn.q_norm",
            "self_attn.k_norm",
            "input_layernorm",
            "post_attention_layernorm",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
        gpu.reclaim(hidden)
        if progress is not None:
            progress("text")
    hidden = gpu.norm(hidden, plane(weights, "text.norm.weight").data, len(ids), 1024)
    hidden = _apply(gpu, hidden, len(ids), plane(weights, "text_projector.weight"))
    return gpu.clip(hidden)


def _lyric(gpu: Any, weights: dict[str, Plane], ids: list[int], progress: Progress | None) -> Any:
    table = plane(weights, "text.embed_tokens.weight")
    raw = gpu.gather(table.data, table.scale, gpu.upload(np.asarray(ids, dtype=np.int32)), len(ids), table.cols, table.inner)
    hidden = _apply(gpu, raw, len(ids), plane(weights, "lyric_encoder.embed_tokens.weight"))
    cos, sin = gpu.rope_table(len(ids), HEAD_DIM, 1_000_000.0)
    for index in range(8):
        window = SLIDING if (index + 1) % 2 == 1 else -1
        hidden = _encoder_layer(
            gpu,
            weights,
            hidden,
            len(ids),
            cos,
            sin,
            window,
            f"lyric_encoder.layers.{index}",
            2048,
            6144,
            16,
            8,
            "self_attn.to_q",
            "self_attn.to_k",
            "self_attn.to_v",
            "self_attn.to_out.0",
            "self_attn.norm_q",
            "self_attn.norm_k",
            "input_layernorm",
            "post_attention_layernorm",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
        gpu.reclaim(hidden)
        if progress is not None:
            progress("lyric")
    hidden = gpu.norm(hidden, plane(weights, "lyric_encoder.norm.weight").data, len(ids), 2048)
    return gpu.clip(hidden)


def _timbre(gpu: Any, weights: dict[str, Plane], progress: Progress | None) -> Any:
    silence = plane(weights, "silence_latent")
    taken = _slice(gpu, silence.data, 0, TIMBRE_FRAMES * 64)
    hidden = _apply(gpu, taken, TIMBRE_FRAMES, plane(weights, "timbre_encoder.embed_tokens.weight"))
    cos, sin = gpu.rope_table(TIMBRE_FRAMES, HEAD_DIM, 1_000_000.0)
    for index in range(4):
        window = SLIDING if (index + 1) % 2 == 1 else -1
        hidden = _encoder_layer(
            gpu,
            weights,
            hidden,
            TIMBRE_FRAMES,
            cos,
            sin,
            window,
            f"timbre_encoder.layers.{index}",
            2048,
            6144,
            16,
            8,
            "self_attn.to_q",
            "self_attn.to_k",
            "self_attn.to_v",
            "self_attn.to_out.0",
            "self_attn.norm_q",
            "self_attn.norm_k",
            "input_layernorm",
            "post_attention_layernorm",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        )
        gpu.reclaim(hidden)
        if progress is not None:
            progress("timbre")
    hidden = gpu.norm(hidden, plane(weights, "timbre_encoder.norm.weight").data, TIMBRE_FRAMES, 2048)
    return gpu.clip(_slice(gpu, hidden, 0, 2048))


def _pack(gpu: Any, lyric: Any, lyric_len: int, timbre: Any, text: Any, text_len: int) -> tuple[Any, int]:
    total = lyric_len + 1 + text_len
    dest = gpu.buffer(total * 2048 * 4)
    gpu.span(lyric, dest, lyric_len * 2048, 0, 0)
    gpu.span(timbre, dest, 2048, 0, lyric_len * 2048)
    gpu.span(text, dest, text_len * 2048, 0, (lyric_len + 1) * 2048)
    return dest, total


def _encoder_layer(
    gpu: Any,
    weights: dict[str, Plane],
    hidden: Any,
    seq: int,
    cos: Any,
    sin: Any,
    window: int,
    prefix: str,
    width: int,
    intermediate: int,
    heads: int,
    kv_heads: int,
    q_name: str,
    k_name: str,
    v_name: str,
    o_name: str,
    qn_name: str,
    kn_name: str,
    norm1: str,
    norm2: str,
    gate: str,
    up: str,
    down: str,
) -> Any:
    normed = gpu.norm(hidden, plane(weights, f"{prefix}.{norm1}.weight").data, seq, width)
    query, key, value = _qkv(
        gpu,
        weights,
        normed,
        seq,
        width,
        heads,
        kv_heads,
        f"{prefix}.{q_name}.weight",
        f"{prefix}.{k_name}.weight",
        f"{prefix}.{v_name}.weight",
        f"{prefix}.{qn_name}.weight",
        f"{prefix}.{kn_name}.weight",
        cos,
        sin,
        True,
    )
    attended = gpu.attend(query, key[0], key[1], value[0], value[1], seq, seq, heads, kv_heads, HEAD_DIM, window)
    out = _apply(gpu, attended, seq, plane(weights, f"{prefix}.{o_name}.weight"))
    gpu.add(hidden, out, seq * width)
    normed = gpu.norm(hidden, plane(weights, f"{prefix}.{norm2}.weight").data, seq, width)
    mixed = _swiglu(gpu, weights, normed, seq, width, intermediate, f"{prefix}.{gate}.weight", f"{prefix}.{up}.weight", f"{prefix}.{down}.weight")
    gpu.add(hidden, mixed, seq * width)
    return hidden


def _qkv(
    gpu: Any,
    weights: dict[str, Plane],
    hidden: Any,
    seq: int,
    width: int,
    heads: int,
    kv_heads: int,
    q_key: str,
    k_key: str,
    v_key: str,
    qn_key: str,
    kn_key: str,
    cos: Any,
    sin: Any,
    use_rope: bool,
) -> tuple[Any, tuple[Any, Any], tuple[Any, Any]]:
    query = _apply(gpu, hidden, seq, plane(weights, q_key))
    key = _apply(gpu, hidden, seq, plane(weights, k_key))
    value = _apply(gpu, hidden, seq, plane(weights, v_key))
    query = gpu.norm(query, plane(weights, qn_key).data, seq * heads, HEAD_DIM)
    key = gpu.norm(key, plane(weights, kn_key).data, seq * kv_heads, HEAD_DIM)
    if use_rope:
        query = gpu.rope(query, cos, sin, seq, heads, HEAD_DIM)
        key = gpu.rope(key, cos, sin, seq, kv_heads, HEAD_DIM)
    key_q = gpu.quantize(key, seq * kv_heads * HEAD_DIM)
    value_q = gpu.quantize(value, seq * kv_heads * HEAD_DIM)
    return query, key_q, value_q


def _swiglu(
    gpu: Any,
    weights: dict[str, Plane],
    hidden: Any,
    seq: int,
    width: int,
    intermediate: int,
    gate: str,
    up: str,
    down: str,
) -> Any:
    del width
    gated = _apply(gpu, hidden, seq, plane(weights, gate))
    up_values = _apply(gpu, hidden, seq, plane(weights, up))
    mixed = gpu.silu_mul(gated, up_values, seq * intermediate)
    return _apply(gpu, mixed, seq, plane(weights, down))


def _cross_cache(gpu: Any, weights: dict[str, Plane], cond: Any, cond_len: int) -> list[tuple[Any, Any, Any, Any]]:
    cached = []
    for index in range(32):
        prefix = f"dit.layers.{index}.cross_attn"
        key = _apply(gpu, cond, cond_len, plane(weights, f"{prefix}.to_k.weight"))
        value = _apply(gpu, cond, cond_len, plane(weights, f"{prefix}.to_v.weight"))
        key = gpu.norm(key, plane(weights, f"{prefix}.norm_k.weight").data, cond_len * 8, HEAD_DIM)
        key_q = gpu.quantize(key, cond_len * 8 * HEAD_DIM)
        value_q = gpu.quantize(value, cond_len * 8 * HEAD_DIM)
        cached.append((gpu.clip(key_q[0]), gpu.clip(key_q[1]), gpu.clip(value_q[0]), gpu.clip(value_q[1])))
    return cached


def _timestep(gpu: Any, weights: dict[str, Plane], tick: float) -> tuple[Any, Any, Any]:
    temb_t, proj_t = _tower(gpu, weights, "dit.time_embed", tick)
    temb_r, proj_r = _tower(gpu, weights, "dit.time_embed_r", 0.0)
    gpu.add_vectors(temb_t, temb_r, 2560)
    gpu.add_vectors(proj_t, proj_r, 15360)
    return proj_t, None, temb_t


def _tower(gpu: Any, weights: dict[str, Plane], prefix: str, tick: float) -> tuple[Any, Any]:
    freq = gpu.sinusoid(tick)
    hidden = _apply(gpu, freq, 1, plane(weights, f"{prefix}.linear_1.weight"))
    hidden = gpu.silu(hidden, 2560)
    temb = _apply(gpu, hidden, 1, plane(weights, f"{prefix}.linear_2.weight"))
    hidden = gpu.silu(temb, 2560)
    proj = _apply(gpu, hidden, 1, plane(weights, f"{prefix}.time_proj.weight"))
    return temb, proj


def _dit_layer(
    gpu: Any,
    weights: dict[str, Plane],
    hidden: Any,
    seq: int,
    cos: Any,
    sin: Any,
    proj: Any,
    _scale: Any,
    cross: tuple[Any, Any, Any, Any],
    cond_len: int,
    index: int,
) -> Any:
    del _scale
    width = 2560
    table = plane(weights, f"dit.layers.{index}.scale_shift_table").data
    added = gpu.buffer(6 * width * 4)
    gpu.span(proj, added, 6 * width, 0, 0)
    gpu.add_vectors(added, table, 6 * width)
    chunks = [_slice(gpu, added, part * width, width) for part in range(6)]
    normed = gpu.norm(hidden, plane(weights, f"dit.layers.{index}.self_attn_norm.weight").data, seq, width)
    normed = gpu.adaln(normed, chunks[1], chunks[0], seq, width)
    query, key, value = _qkv(
        gpu,
        weights,
        normed,
        seq,
        width,
        32,
        8,
        f"dit.layers.{index}.self_attn.to_q.weight",
        f"dit.layers.{index}.self_attn.to_k.weight",
        f"dit.layers.{index}.self_attn.to_v.weight",
        f"dit.layers.{index}.self_attn.norm_q.weight",
        f"dit.layers.{index}.self_attn.norm_k.weight",
        cos,
        sin,
        True,
    )
    window = SLIDING if (index + 1) % 2 == 1 else -1
    attended = gpu.attend(query, key[0], key[1], value[0], value[1], seq, seq, 32, 8, HEAD_DIM, window)
    out = _apply(gpu, attended, seq, plane(weights, f"dit.layers.{index}.self_attn.to_out.0.weight"))
    gpu.add_gated(hidden, out, chunks[2], seq, width)
    normed = gpu.norm(hidden, plane(weights, f"dit.layers.{index}.cross_attn_norm.weight").data, seq, width)
    query = _apply(gpu, normed, seq, plane(weights, f"dit.layers.{index}.cross_attn.to_q.weight"))
    query = gpu.norm(query, plane(weights, f"dit.layers.{index}.cross_attn.norm_q.weight").data, seq * 32, HEAD_DIM)
    attended = gpu.attend(query, cross[0], cross[1], cross[2], cross[3], seq, cond_len, 32, 8, HEAD_DIM, -1)
    out = _apply(gpu, attended, seq, plane(weights, f"dit.layers.{index}.cross_attn.to_out.0.weight"))
    gpu.add(hidden, out, seq * width)
    normed = gpu.norm(hidden, plane(weights, f"dit.layers.{index}.mlp_norm.weight").data, seq, width)
    normed = gpu.adaln(normed, chunks[4], chunks[3], seq, width)
    mixed = _swiglu(
        gpu,
        weights,
        normed,
        seq,
        width,
        9728,
        f"dit.layers.{index}.mlp.gate_proj.weight",
        f"dit.layers.{index}.mlp.up_proj.weight",
        f"dit.layers.{index}.mlp.down_proj.weight",
    )
    gpu.add_gated(hidden, mixed, chunks[5], seq, width)
    return hidden


def _decode(gpu: Any, weights: dict[str, Plane], latent: Any, frames: int, progress: Progress | None) -> Any:
    channels = [2048, 1024, 512, 256, 128, 128]
    strides = [10, 6, 4, 4, 2]
    hidden, length = _conv(gpu, weights, latent, "decoder.conv1", 64, 2048, 7, frames, 3, 1, 1)
    if progress is not None:
        progress("vae")
    for index, stride in enumerate(strides):
        hidden = _snake(gpu, weights, hidden, f"decoder.block.{index}.snake1", channels[index], length)
        kernel = stride * 2
        hidden, length = _conv_t(
            gpu,
            weights,
            hidden,
            f"decoder.block.{index}.conv_t1",
            channels[index],
            channels[index + 1],
            kernel,
            length,
            stride // 2,
            stride,
        )
        for unit, dilation in ((1, 1), (2, 3), (3, 9)):
            hidden = _residual(gpu, weights, hidden, f"decoder.block.{index}.res_unit{unit}", channels[index + 1], length, dilation)
        if progress is not None:
            progress("vae")
    hidden = _snake(gpu, weights, hidden, "decoder.snake1", 128, length)
    hidden, length = _conv(gpu, weights, hidden, "decoder.conv2", 128, 2, 7, length, 3, 1, 1)
    if progress is not None:
        progress("vae")
    expected = frames * HOP
    if length != expected:
        message = f"decoder length {length} is not {expected}"
        raise RuntimeError(message)
    return hidden


def _snake(gpu: Any, weights: dict[str, Plane], values: Any, prefix: str, channels: int, length: int) -> Any:
    alpha = plane(weights, f"{prefix}.alpha").data
    beta_key = f"{prefix}.beta"
    beta = plane(weights, beta_key).data if beta_key in weights else alpha
    return gpu.snake(values, alpha, beta, channels, length)


def _residual(gpu: Any, weights: dict[str, Plane], values: Any, prefix: str, channels: int, length: int, dilation: int) -> Any:
    hidden = _snake(gpu, weights, values, f"{prefix}.snake1", channels, length)
    hidden, out_len = _conv(gpu, weights, hidden, f"{prefix}.conv1", channels, channels, 7, length, ((7 - 1) * dilation) // 2, 1, dilation)
    if out_len != length:
        message = f"{prefix} changed the length"
        raise RuntimeError(message)
    hidden = _snake(gpu, weights, hidden, f"{prefix}.snake2", channels, length)
    hidden, out_len = _conv(gpu, weights, hidden, f"{prefix}.conv2", channels, channels, 1, length, 0, 1, 1)
    if out_len != length:
        message = f"{prefix} pointwise changed the length"
        raise RuntimeError(message)
    gpu.add(values, hidden, channels * length)
    return values


def _conv(
    gpu: Any,
    weights: dict[str, Plane],
    values: Any,
    prefix: str,
    cin: int,
    cout: int,
    kernel: int,
    length: int,
    pad: int,
    stride: int,
    dilation: int,
) -> tuple[Any, int]:
    item = plane(weights, f"{prefix}.weight")
    return gpu.conv(values, item.data, item.scale, item.bias, cin, cout, kernel, item.inner, length, pad, stride, dilation)


def _conv_t(
    gpu: Any,
    weights: dict[str, Plane],
    values: Any,
    prefix: str,
    cin: int,
    cout: int,
    kernel: int,
    length: int,
    pad: int,
    stride: int,
) -> tuple[Any, int]:
    item = plane(weights, f"{prefix}.weight")
    return gpu.conv_transpose(
        values,
        item.data,
        item.scale,
        item.bias,
        cin,
        cout,
        kernel,
        item.inner,
        length,
        pad,
        stride,
    )
