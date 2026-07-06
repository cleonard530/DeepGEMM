import os
import subprocess

# Load the C++ extension and re-export ops through deep_gemm._C.
from ._C import *  # noqa: F403
from . import _C

# Set some default environment provided at setup
try:
    # noinspection PyUnresolvedReferences
    from .envs import persistent_envs
    for key, value in persistent_envs.items():
        if key not in os.environ:
            os.environ[key] = value
except ImportError:
    pass

# Some alias for legacy supports
# TODO: remove these later
try:
    fp8_m_grouped_gemm_nt_masked = m_grouped_fp8_gemm_nt_masked
    bf16_m_grouped_gemm_nt_masked = m_grouped_bf16_gemm_nt_masked
except NameError:
    pass

# Mega kernels
from .mega import (
    SymmBuffer,
    get_symm_buffer_for_mega_moe,
    transform_weights_for_mega_moe,
    fp8_fp4_mega_moe,
    bf16_mega_moe,
)

# Some utils
from . import testing
from . import utils
from .utils import *

# Legacy Triton kernels for A100
try:
    from . import legacy
except Exception as e:
    print(f'Failed to load legacy DeepGEMM A100 Triton kernels: {e}')

# Initialize CPP modules
def _find_cuda_home() -> str:
    # TODO: reuse PyTorch API later
    # For some PyTorch versions, the original `_find_cuda_home` will initialize CUDA, which is incompatible with process forks
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # noinspection PyBroadException
        try:
            with open(os.devnull, 'w') as devnull:
                nvcc = subprocess.check_output(['which', 'nvcc'], stderr=devnull).decode().rstrip('\r\n')
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None
    assert cuda_home is not None
    return cuda_home


_C.init(
    os.path.dirname(os.path.abspath(__file__)), # Library root directory path
    _find_cuda_home()                           # CUDA home
)

__version__ = '2.6.1'
