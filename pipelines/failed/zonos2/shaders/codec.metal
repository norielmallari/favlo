#include <metal_stdlib>
using namespace metal;

kernel void conv1d(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &length [[buffer(4)]],
    constant uint &in_channels [[buffer(5)]],
    constant uint &out_channels [[buffer(6)]],
    constant uint &taps [[buffer(7)]],
    constant int &padding [[buffer(8)]],
    constant int &dilation [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint time = gid.x;
    uint channel = gid.y;
    if (time >= length || channel >= out_channels) {
        return;
    }
    float acc = bias[channel];
    for (uint incoming = 0; incoming < in_channels; ++incoming) {
        for (uint tap = 0; tap < taps; ++tap) {
            int index = int(time) + int(tap) * dilation - padding;
            if (index < 0 || index >= int(length)) {
                continue;
            }
            uint weight_index = (channel * in_channels + incoming) * taps + tap;
            acc += weight[weight_index] * input[uint(index) * in_channels + incoming];
        }
    }
    output[time * out_channels + channel] = acc;
}

kernel void conv_transpose1d(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &in_length [[buffer(4)]],
    constant uint &out_length [[buffer(5)]],
    constant uint &in_channels [[buffer(6)]],
    constant uint &out_channels [[buffer(7)]],
    constant uint &taps [[buffer(8)]],
    constant uint &stride [[buffer(9)]],
    constant int &padding [[buffer(10)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint time = gid.x;
    uint channel = gid.y;
    if (time >= out_length || channel >= out_channels) {
        return;
    }
    float acc = bias[channel];
    for (uint tap = 0; tap < taps; ++tap) {
        int numerator = int(time) - int(tap) + padding;
        if (numerator < 0 || numerator % int(stride) != 0) {
            continue;
        }
        int source = numerator / int(stride);
        if (source >= int(in_length)) {
            continue;
        }
        for (uint incoming = 0; incoming < in_channels; ++incoming) {
            uint weight_index = (incoming * out_channels + channel) * taps + tap;
            acc += weight[weight_index] * input[uint(source) * in_channels + incoming];
        }
    }
    output[time * out_channels + channel] = acc;
}

kernel void snake(
    device const float *values [[buffer(0)]],
    device const float *alpha [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint &channels [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    uint channel = id % channels;
    float scale = alpha[channel];
    float value = values[id];
    float wave = sin(scale * value);
    output[id] = value + wave * wave / scale;
}
