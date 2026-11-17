# Isaac environment (simulation + cuRobo)

What gets installed:
- Python 3.11
- Isaac Sim 5.1.0
- Isaac Lab v2.3.2
- torch 2.7.0+cu128
- cuRobo at commit `ebb7170`, with our patch

On this PC (RTX 4080 SUPER) the result is the environment `/home/data/siddiquieu1/envs/env_a2world`.

`ENV` and `C` below are install locations; pick any. If conda refuses its `defaults` channel (the Anaconda terms
of service), keep `--override-channels -c conda-forge`.

```bash
ENV=/home/data/siddiquieu1/envs/env_a2world      # the Isaac environment
C=/home/data/siddiquieu1/envs/cuda128            # CUDA 12.8 toolkit, only for compiling cuRobo

# 1. Isaac Sim + torch
conda create -y -p $ENV --override-channels -c conda-forge python=3.11 && conda activate $ENV
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# 2. Isaac Lab
git clone --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab && cd ~/IsaacLab
pip install "setuptools<81" wheel && pip install flatdict==4.0.1 --no-build-isolation   # flatdict's build needs pkg_resources
OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh --install none && pip install "packaging==23.0"

# 3. CUDA toolkit with the same major version as torch (12.x), in its own environment
conda create -y -p $C --override-channels -c conda-forge \
    cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-libraries-dev=12.8 cuda-nvtx-dev=12.8 cuda-profiler-api=12.8 cuda-version=12.8

# 4. cuRobo (pinned commit + our patch), without changing any version Isaac pins
git clone https://github.com/NVlabs/curobo.git ~/src/curobo && cd ~/src/curobo
git checkout ebb71702f3f70e767f40fd8e050674af0288abe8
git apply <repo>/docs/env_patches/curobo_warp118.patch            # warp 1.18 dropped wp.torch.*
printf "torch==2.7.0+cu128\nnumpy==1.26.0\nwarp-lang==1.18.0\nscipy==1.15.3\ntrimesh==4.5.1\nnetworkx==3.3\nsetuptools==80.10.2\npackaging==23.0\n" > /tmp/c.txt
pip install -c /tmp/c.txt pybind11 numpy-quaternion "setuptools_scm>=6.2,<9" yourdfpy importlib_resources scikit-image ninja imageio
unset PYTHONPATH PYTHONHOME
export CUDA_HOME=$C PATH=$C/bin:$PATH CPATH=$C/targets/x86_64-linux/include LIBRARY_PATH=$C/targets/x86_64-linux/lib
export TORCH_CUDA_ARCH_LIST=8.9                                    # your GPU: 8.9 = RTX 40xx, 8.6 = RTX 30xx / A6000
pip install -e . --no-build-isolation --no-deps                    # ~20 min of CUDA compilation
```

## Check

Run these with the environment active and `A2W_DATA` set.

1. From `ur5e_sim/setup/`, run `python make_robot_cfg.py`. It must print `OK: cuRobo kinematics match the sim's ready pose`.
2. Do the one-time asset steps from the README.
3. Run one episode, as in the README.
