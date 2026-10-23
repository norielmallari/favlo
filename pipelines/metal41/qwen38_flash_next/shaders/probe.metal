
#include <metal_stdlib>
using namespace metal;

kernel void copy_bytes(
    device const uchar* src [[buffer(0)]],
    device uchar* dst [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < count) {
        dst[gid] = src[gid];
    }
}

kernel void read_int8(
    device const char* src [[buffer(0)]],
    device int* dst [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) {
        return;
    }
    for (uint i = 0; i < 32; ++i) {
        dst[i] = int(src[i]);
    }
}

kernel void unpack_int4(
    device const uchar* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) {
        return;
    }
    for (uint group = 0; group < 2; ++group) {
        packed_vec<uchar, 8> bytes;
        for (uint i = 0; i < 8; ++i) {
            bytes[i] = src[group * 8 + i];
        }
        auto values = unpack<float>(packed_int4b<16>(bytes));
        for (uint i = 0; i < 16; ++i) {
            dst[group * 16 + i] = values[i];
        }
    }
}

kernel void unpack_e4m3(
    device const uchar* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) {
        return;
    }
    for (uint group = 0; group < 4; ++group) {
        packed_vec<uchar, 8> bytes;
        for (uint i = 0; i < 8; ++i) {
            bytes[i] = src[group * 8 + i];
        }
        auto values = unpack<float>(packed_metal_fp8_e4m3<8>(bytes));
        for (uint i = 0; i < 8; ++i) {
            dst[group * 8 + i] = values[i];
        }
    }
}

kernel void unpack_e5m2(
    device const uchar* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) {
        return;
    }
    for (uint group = 0; group < 4; ++group) {
        packed_vec<uchar, 8> bytes;
        for (uint i = 0; i < 8; ++i) {
            bytes[i] = src[group * 8 + i];
        }
        auto values = unpack<float>(packed_metal_fp8_e5m2<8>(bytes));
        for (uint i = 0; i < 8; ++i) {
            dst[group * 8 + i] = values[i];
        }
    }
}

kernel void reconstruct_int4(
    device const uchar* src [[buffer(0)]],
    device const uchar* scale [[buffer(1)]],
    device float* dst [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid != 0) {
        return;
    }
    float factor = exp2(float(scale[0]) - 127.0f);
    for (uint group = 0; group < 2; ++group) {
        packed_vec<uchar, 8> bytes;
        for (uint i = 0; i < 8; ++i) {
            bytes[i] = src[group * 8 + i];
        }
        auto values = unpack<float>(packed_int4b<16>(bytes));
        for (uint i = 0; i < 16; ++i) {
            dst[group * 16 + i] = values[i] * factor;
        }
    }
}
