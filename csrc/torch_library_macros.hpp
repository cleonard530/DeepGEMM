#pragma once

#include <torch/library.h>

#include "torch_library_utils.hpp"

namespace deep_gemm::torch_registration {

using deep_gemm::torch_library_utils::list_to_optional_vector_int;
using deep_gemm::torch_library_utils::list_to_recipe2;
using deep_gemm::torch_library_utils::list_to_recipe3;
using deep_gemm::torch_library_utils::list_to_recipe_variant;
using deep_gemm::torch_library_utils::list_to_tuple3;

}  // namespace deep_gemm::torch_registration

#define DEEP_GEMM_IMPL(fn) TORCH_FN(deep_gemm::torch_registration::fn)
