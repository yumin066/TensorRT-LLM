/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "tvm_ffi_utils.h"

namespace
{

constexpr DLDataType kBFloat16 = DLDataType{kDLBfloat, 16, 1};
constexpr DLDataType kFloat32 = DLDataType{kDLFloat, 32, 1};
constexpr DLDataType kUInt8 = DLDataType{kDLUInt, 8, 1};
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

__device__ __forceinline__ float reciprocal_approximate_ftz(float value)
{
    float result;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(result) : "f"(value));
    return result;
}

__device__ __forceinline__ uint8_t fp32_pair_to_e2m1(float even, float odd)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    uint32_t packed;
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;\n"
        "mov.b32 %0, {byte0, 0, 0, 0};\n"
        "}"
        : "=r"(packed)
        : "f"(even), "f"(odd));
    return static_cast<uint8_t>(packed);
#else
    return 0;
#endif
}

__device__ __forceinline__ float warp_sum(float value)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return __shfl_sync(0xffffffffu, value, 0);
}

template <bool kNormalizeAndActivate>
__global__ __launch_bounds__(kThreads, 4) void fused_preprocess_kernel(__nv_bfloat16 const* input,
    float const* global_scale, uint8_t* packed_output, uint8_t* scale_output, int batch, int channels, int depth,
    int height, int width, int64_t stride_n, int64_t stride_c, int64_t stride_d, int64_t stride_h, int64_t stride_w,
    bool causal, float eps)
{
    __shared__ float reduction[kWarps][32];
    int const lane = static_cast<int>(threadIdx.x) & 31;
    int const warp = static_cast<int>(threadIdx.x) >> 5;
    int const physical_depth = depth + 2;
    int const physical_height = height + 2;
    int const physical_width = width + 2;
    const int64_t physical_voxels = static_cast<int64_t>(physical_depth) * physical_height * physical_width;
    const int64_t physical_voxel = (static_cast<int64_t>(blockIdx.x) * kWarps) + warp;
    int const batch_idx = static_cast<int>(blockIdx.y);
    if (batch_idx >= batch || physical_voxel >= physical_voxels)
    {
        return;
    }

    int64_t coordinate = physical_voxel;
    int const physical_w = static_cast<int>(coordinate % physical_width);
    coordinate /= physical_width;
    int const physical_h = static_cast<int>(coordinate % physical_height);
    int const physical_d = static_cast<int>(coordinate / physical_height);
    int const source_w = physical_w - 1;
    int const source_h = physical_h - 1;
    bool const spatial_valid = source_h >= 0 && source_h < height && source_w >= 0 && source_w < width;

    int source_d;
    if (causal)
    {
        source_d = physical_d < 2 ? 0 : physical_d - 2;
    }
    else
    {
        source_d = physical_d == 0 ? 0 : (physical_d == depth + 1 ? depth - 1 : physical_d - 1);
    }
    const int64_t input_voxel_offset = static_cast<int64_t>(batch_idx) * stride_n
        + static_cast<int64_t>(source_d) * stride_d + static_cast<int64_t>(source_h) * stride_h
        + static_cast<int64_t>(source_w) * stride_w;

    float inverse_rms = 1.0f;
    if constexpr (kNormalizeAndActivate)
    {
        float sum_square = 0.0f;
        if (spatial_valid)
        {
            for (int channel = lane; channel < channels; channel += 32)
            {
                float const value = __bfloat162float(input[input_voxel_offset + channel * stride_c]);
                sum_square = fmaf(value, value, sum_square);
            }
        }
        sum_square = warp_sum(sum_square);
        inverse_rms = rsqrtf(sum_square / static_cast<float>(channels) + eps);
    }

    float const scale_multiplier = global_scale[0];
    int const half = lane >> 4;
    int const lane_in_half = lane & 15;
    const uint32_t half_mask = half == 0 ? 0x0000ffffu : 0xffff0000u;
    int const scale_groups = channels / 16;
    const int64_t packed_row = (static_cast<int64_t>(batch_idx) * physical_voxels + physical_voxel) * (channels / 2);
    const int64_t scale_row = (static_cast<int64_t>(batch_idx) * physical_voxels + physical_voxel) * scale_groups;

    for (int channel_base = 0; channel_base < channels; channel_base += 32)
    {
        int const channel = channel_base + lane;
        float value = 0.0f;
        if (spatial_valid)
        {
            value = __bfloat162float(input[input_voxel_offset + channel * stride_c]);
            if constexpr (kNormalizeAndActivate)
            {
                value *= inverse_rms;
                value = value * reciprocal_approximate_ftz(1.0f + __expf(-value));
            }
        }

        reduction[warp][lane] = fabsf(value);
        __syncwarp();
#pragma unroll
        for (int offset = 8; offset > 0; offset >>= 1)
        {
            if (lane_in_half < offset)
            {
                reduction[warp][lane] = fmaxf(reduction[warp][lane], reduction[warp][lane + offset]);
            }
            __syncwarp();
        }
        float const vector_max = reduction[warp][half * 16];
        float scale_value = scale_multiplier * (vector_max * reciprocal_approximate_ftz(6.0f));
        __nv_fp8_e4m3 narrowed_scale = __nv_fp8_e4m3(scale_value);
        const uint8_t scale_code = narrowed_scale.__x;
        scale_value = static_cast<float>(narrowed_scale);
        float const output_scale = vector_max != 0.0f
            ? reciprocal_approximate_ftz(scale_value * reciprocal_approximate_ftz(scale_multiplier))
            : 0.0f;

        if (lane_in_half == 0)
        {
            scale_output[scale_row + channel_base / 16 + half] = scale_code;
        }
        value *= output_scale;
        float const paired_value = __shfl_xor_sync(half_mask, value, 1);
        if ((lane_in_half & 1) == 0)
        {
            packed_output[packed_row + channel / 2] = fp32_pair_to_e2m1(value, paired_value);
        }
    }
}

void video_vae_ltx23_nvfp4_preprocess(TensorView input, TensorView global_scale, TensorView packed_output,
    TensorView scale_output, int64_t causal, int64_t normalize_and_activate, double eps)
{
    CHECK_CUDA(input);
    CHECK_INPUT_TYPE(input, kBFloat16);
    CHECK_INPUT_AND_TYPE(global_scale, kFloat32);
    CHECK_INPUT_AND_TYPE(packed_output, kUInt8);
    CHECK_INPUT_AND_TYPE(scale_output, kUInt8);
    CHECK_DIM(5, input);
    CHECK_DIM(1, global_scale);
    CHECK_DIM(5, packed_output);
    CHECK_DIM(5, scale_output);
    CHECK_DEVICE(input, global_scale);
    CHECK_DEVICE(input, packed_output);
    CHECK_DEVICE(input, scale_output);

    TVM_FFI_ICHECK_EQ(global_scale.size(0), 1);
    TVM_FFI_ICHECK(causal == 0 || causal == 1);
    TVM_FFI_ICHECK(normalize_and_activate == 0 || normalize_and_activate == 1);
    int const batch = static_cast<int>(input.size(0));
    int const channels = static_cast<int>(input.size(1));
    int const depth = static_cast<int>(input.size(2));
    int const height = static_cast<int>(input.size(3));
    int const width = static_cast<int>(input.size(4));
    TVM_FFI_ICHECK_GT(batch, 0);
    TVM_FFI_ICHECK_EQ(channels % 128, 0);
    TVM_FFI_ICHECK_GT(depth, 0);
    TVM_FFI_ICHECK_GT(height, 0);
    TVM_FFI_ICHECK_GT(width, 0);

    int const physical_depth = depth + 2;
    int const physical_height = height + 2;
    int const physical_width = width + 2;
    TVM_FFI_ICHECK_EQ(packed_output.size(0), batch);
    TVM_FFI_ICHECK_EQ(packed_output.size(1), physical_depth);
    TVM_FFI_ICHECK_EQ(packed_output.size(2), physical_height);
    TVM_FFI_ICHECK_EQ(packed_output.size(3), physical_width);
    TVM_FFI_ICHECK_EQ(packed_output.size(4), channels / 2);
    TVM_FFI_ICHECK_EQ(scale_output.size(0), batch);
    TVM_FFI_ICHECK_EQ(scale_output.size(1), physical_depth);
    TVM_FFI_ICHECK_EQ(scale_output.size(2), physical_height);
    TVM_FFI_ICHECK_EQ(scale_output.size(3), physical_width);
    TVM_FFI_ICHECK_EQ(scale_output.size(4), channels / 16);

    ffi::CUDADeviceGuard device_guard(input.device().device_id);
    const int64_t physical_voxels = static_cast<int64_t>(physical_depth) * physical_height * physical_width;
    const dim3 grid((physical_voxels + kWarps - 1) / kWarps, batch);
    const dim3 block(kThreads);
    cudaStream_t stream = get_stream(input.device());
    if (normalize_and_activate)
    {
        fused_preprocess_kernel<true><<<grid, block, 0, stream>>>(static_cast<__nv_bfloat16 const*>(input.data_ptr()),
            static_cast<float const*>(global_scale.data_ptr()), static_cast<uint8_t*>(packed_output.data_ptr()),
            static_cast<uint8_t*>(scale_output.data_ptr()), batch, channels, depth, height, width, input.stride(0),
            input.stride(1), input.stride(2), input.stride(3), input.stride(4), causal != 0, static_cast<float>(eps));
    }
    else
    {
        fused_preprocess_kernel<false><<<grid, block, 0, stream>>>(static_cast<__nv_bfloat16 const*>(input.data_ptr()),
            static_cast<float const*>(global_scale.data_ptr()), static_cast<uint8_t*>(packed_output.data_ptr()),
            static_cast<uint8_t*>(scale_output.data_ptr()), batch, channels, depth, height, width, input.stride(0),
            input.stride(1), input.stride(2), input.stride(3), input.stride(4), causal != 0, static_cast<float>(eps));
    }
    TVM_FFI_ICHECK_EQ(cudaGetLastError(), cudaSuccess);
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(video_vae_ltx23_nvfp4_preprocess, video_vae_ltx23_nvfp4_preprocess);
