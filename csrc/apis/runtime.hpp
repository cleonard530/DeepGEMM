#pragma once

#include <vector>

#if DG_TENSORMAP_COMPATIBLE
#include "../jit/compiler.hpp"
#include "../jit/kernel_runtime.hpp"
#endif
#include "../jit/device_runtime.hpp"
#include "../jit_kernels/heuristics/runtime.hpp"

#include <torch/csrc/stable/library.h>

namespace deep_gemm::torch_registration {

static void set_num_sms(const int64_t& new_num_sms) {
    device_runtime->set_num_sms(static_cast<int>(new_num_sms));
}

static int64_t get_num_sms() {
    return device_runtime->get_num_sms();
}

static void set_tc_util(const int64_t& new_tc_util) {
    device_runtime->set_tc_util(static_cast<int>(new_tc_util));
}

static int64_t get_tc_util() {
    return device_runtime->get_tc_util();
}

static void set_pdl(const bool& new_enable_pdl) {
    device_runtime->set_pdl(new_enable_pdl);
}

static bool get_pdl() {
    return device_runtime->get_pdl();
}

static void set_ignore_compile_dims(const bool& new_value) {
    heuristics_runtime->set_ignore_compile_dims(new_value);
}

static void set_block_size_multiple_of(const std::vector<int64_t>& value) {
    if (value.size() == 1) {
        const int v = static_cast<int>(value[0]);
        heuristics_runtime->set_block_size_multiple_of(v, v);
    } else {
        DG_HOST_ASSERT(value.size() == 2);
        heuristics_runtime->set_block_size_multiple_of(
            static_cast<int>(value[0]), static_cast<int>(value[1]));
    }
}

static void init(const std::string& library_root_path,
                 const std::string& cuda_home_path_by_python) {
#if DG_TENSORMAP_COMPATIBLE
        Compiler::prepare_init(library_root_path, cuda_home_path_by_python);
        KernelRuntime::prepare_init(cuda_home_path_by_python);
        IncludeParser::prepare_init(library_root_path);
#endif
}

}  // namespace deep_gemm::torch_registration

STABLE_TORCH_LIBRARY_FRAGMENT(deep_gemm, m) {
    m.def("set_num_sms(int new_num_sms) -> ()");
    m.def("get_num_sms() -> int");
    m.def("set_tc_util(int new_tc_util) -> ()");
    m.def("get_tc_util() -> int");
    m.def("set_pdl(bool new_enable_pdl) -> ()");
    m.def("get_pdl() -> bool");
    m.def("set_ignore_compile_dims(bool new_value) -> ()");
    m.def("set_block_size_multiple_of(int[] value) -> ()");
    m.def("init(str library_root_path, str cuda_home_path_by_python) -> ()");
}

// No Tensor args means no dispatch key set, so register under CompositeImplicitAutograd (the catch-all key).
STABLE_TORCH_LIBRARY_IMPL(deep_gemm, CompositeImplicitAutograd, m) {
    using namespace deep_gemm::torch_registration;

    m.impl("set_num_sms", TORCH_BOX(&set_num_sms));
    m.impl("get_num_sms", TORCH_BOX(&get_num_sms));
    m.impl("set_tc_util", TORCH_BOX(&set_tc_util));
    m.impl("get_tc_util", TORCH_BOX(&get_tc_util));
    m.impl("set_pdl", TORCH_BOX(&set_pdl));
    m.impl("get_pdl", TORCH_BOX(&get_pdl));
    m.impl("set_ignore_compile_dims", TORCH_BOX(&set_ignore_compile_dims));
    m.impl("set_block_size_multiple_of", TORCH_BOX(&set_block_size_multiple_of));
    m.impl("init", TORCH_BOX(&init));
}
