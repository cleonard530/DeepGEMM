#include <torch/all.h>
#include <torch/library.h>
#include "torch_library_utils.hpp"
#include "utils/registration.h"

#include <deep_jit/backend/cuda/backend.hpp>

#include "apis/config.hpp"
#include "apis/attention.hpp"
#include "apis/einsum.hpp"
#include "apis/hyperconnection.hpp"
#include "apis/gemm.hpp"
#include "apis/layout.hpp"
#include "apis/mega_moe.hpp"
#include "apis/nvfp4_mega_moe.hpp"
#include "apis/mega_mhc.hpp"
#include "apis/mega_gate.hpp"

REGISTER_EXTENSION(_C_extension)
