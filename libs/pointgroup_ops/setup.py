import os
from distutils.sysconfig import get_config_vars
from sys import argv
import torch
from glob import glob

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

(opt,) = get_config_vars("OPT")
os.environ["OPT"] = " ".join(
    flag for flag in opt.split() if flag != "-Wstrict-prototypes"
)


def _argparse(pattern, argv, is_flag=True, is_list=False):
    if is_flag:
        found = pattern in argv
        if found:
            argv.remove(pattern)
        return found, argv
    else:
        arr = [arg for arg in argv if pattern == arg.split("=")[0]]
        if is_list:
            if len(arr) == 0:  # not found
                return False, argv
            else:
                assert "=" in arr[0], f"{arr[0]} requires a value."
                argv.remove(arr[0])
                val = arr[0].split("=")[1]
                if "," in val:
                    return val.split(","), argv
                else:
                    return [val], argv
        else:
            if len(arr) == 0:  # not found
                return False, argv
            else:
                assert "=" in arr[0], f"{arr[0]} requires a value."
                argv.remove(arr[0])
                return arr[0].split("=")[1], argv


def get_sources(surfix='*.c*'):
    src_dir = 'src'
    cuda_dir = os.path.join(src_dir, 'cuda')
    cpu_dir = os.path.join(src_dir, 'cpu')
    return glob(os.path.join(src_dir, surfix)) + glob(os.path.join(cuda_dir, surfix)) + glob(os.path.join(cpu_dir, surfix))


INCLUDE_DIRS, argv = _argparse("--include_dirs", argv, False, is_list=True)
include_dirs = []
if not (INCLUDE_DIRS is False):
    include_dirs += INCLUDE_DIRS


setup(
    name='pointgroup_ops',
    packages=["."],
    # package_dir={"pointgroup_ops": "functions"},
    ext_modules=[
        CUDAExtension(
            name="pointgroup_ops_cuda",
            sources=get_sources() + ["src/bfs_cluster/bfs_cluster.cpp", "src/bfs_cluster/bfs_cluster_kernel.cu"],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': [
                    '-D__CUDA_NO_HALF_OPERATORS__',
                    '-D__CUDA_NO_HALF_CONVERSIONS__',
                    '-D__CUDA_NO_HALF2_OPERATORS__',
                ],
            },
            define_macros=[('WITH_CUDA', None)])
    ],
    include_dirs=[*include_dirs],
    cmdclass={'build_ext': BuildExtension})
