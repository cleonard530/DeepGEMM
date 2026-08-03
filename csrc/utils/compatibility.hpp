#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

// FP8 support requires PyTorch 2.1+, well below the minimum stable-ABI target version.
#define DG_FP8_COMPATIBLE 1

// `cuTensorMapEncodeTiled` is supported since CUDA Driver API 12.1
#define DG_TENSORMAP_COMPATIBLE (CUDA_VERSION >= 12010)

// `cublasGetErrorString` is supported since CUDA Runtime API 11.4.2
#define DG_CUBLAS_GET_ERROR_STRING_COMPATIBLE (CUDART_VERSION >= 11042)

// `CUBLASLT_MATMUL_DESC_FAST_ACCUM` and `CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET` are supported since CUDA Runtime API 11.8
#define DG_CUBLASLT_ADVANCED_FEATURES_COMPATIBLE (CUDART_VERSION >= 11080)