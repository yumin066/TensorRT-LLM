#include "tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_tma_ws_launcher_no_macro.inl"
namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels_oss
{

    template void tma_warp_specialized_generic_moe_gemm_kernelLauncher<cutlass::arch::Sm90, __nv_fp8_e4m3,                                                                                                                                                                                                                  \
        cutlass::uint4b_t, __nv_bfloat16, void, tensorrt_llm::cutlass_extensions::EpilogueOpDefault,                                                                                                                                                                                                                        \
        EpilogueFusion::FINALIZE, cute::Shape<cute::Int<128>, cute::Int<16>, cute::Int<128>>,                                                                                                                                                                                                                      \
        cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>, false, false, false, true>(                                                                                                                                                                                                       \
        TmaWarpSpecializedGroupedGemmInput tma_ws_input, int num_experts, int const multi_processor_count,                                                                                                                                                                                                                  \
        cudaStream_t stream, int* kernel_occupancy, size_t* workspace_size,                                                                                                                                                                                                                                                 \
        cute::Shape<int32_t, int32_t, cute::_1> dynamic_cluster_shape,                                                                                                                                                                                                                                                      \
        cute::Shape<int32_t, int32_t, cute::_1> fallback_cluster_shape);

//        INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(Sm90, __nv_fp8_e4m3, uint4b_t, __nv_bfloat16, void, EpilogueOpDefault, FINALIZE, 128, 16, 128, 1, 1, 1, false, false, false, true);


} // namespace cutlass_kernels_oss
} // namespace kernels
} // namespace tensorrt_llm
