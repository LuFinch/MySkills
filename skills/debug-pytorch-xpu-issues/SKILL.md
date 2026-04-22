# PyTorch XPU Debug Skill

This skill documents the standard workflow for setting up the environment, building PyTorch with XPU support, and running/debugging XPU test cases.

> Compatible with: Claude Code, OpenCode, GitHub Copilot, Roo Code, and other AI coding assistants.

## Environment

### IMPORTANT: Confirm Environment with User First

Before executing any command in this skill, you **MUST** ask the user whether they are using the default environment settings below. If the user is NOT using the defaults, ask them to provide their actual values for any of the following:

- **Workspace**: `/home/sdp/fengqing`
- **PyTorch source**: `/home/sdp/fengqing/workspace/pytorch`
- **oneAPI SDK**: `/mnt/deep_learning_essentials/2025_3_2_36/setvars.sh`
- **Conda env**: `lfq-pt` (Python 3.10)
- **Env setup script**: `/home/sdp/fengqing/workspace/env.sh`

Example question to ask the user:
> "Are you using the default environment (workspace=`/home/sdp/fengqing`, pytorch=`/home/sdp/fengqing/workspace/pytorch`, oneAPI=`/mnt/deep_learning_essentials/2025_3_2_36/setvars.sh`, conda env=`lfq-pt`)? If not, please provide your workspace path, PyTorch source path, oneAPI setvars.sh path, and conda environment name."

Once the user confirms (or supplies) the values, substitute them in all subsequent commands. The defaults shown below assume the user accepted the defaults.

## Shell Command Pattern

Because most AI coding assistants execute commands in a non-interactive shell that does not source `.bashrc`, you **must** use this pattern for all commands that require the environment:

```bash
bash -c 'eval "$(conda shell.bash hook)" && source /home/sdp/fengqing/workspace/env.sh && cd /home/sdp/fengqing/workspace/pytorch && <YOUR_COMMAND>'
```

### Why?
- `eval "$(conda shell.bash hook)"` — initializes conda in a non-interactive bash shell
- `source /home/sdp/fengqing/workspace/env.sh` — sources oneAPI setvars.sh and activates conda env `lfq-pt`
- `cd /home/sdp/fengqing/workspace/pytorch` — switch to the PyTorch source directory

## Step 1: Setup Environment (No build)

If you only need to run Python/pytest commands and don't need to rebuild C++ code:

```bash
bash -c 'eval "$(conda shell.bash hook)" && source /home/sdp/fengqing/workspace/env.sh && cd /home/sdp/fengqing/workspace/pytorch && python -c "import torch; print(torch.__version__); print(torch.xpu.is_available())"'
```

## Step 2: Build PyTorch (Full build)

Full clean build (slow, ~1-2 hours):

```bash
bash -c 'eval "$(conda shell.bash hook)" && source /home/sdp/fengqing/workspace/env.sh && cd /home/sdp/fengqing/workspace/pytorch && export TORCH_XPU_ARCH_LIST="pvc,bmg" && export BUILD_SEPARATE_OPS=1 && git submodule sync && git submodule update --init --recursive && python setup.py clean && python setup.py bdist_wheel 2>&1'
```

## Step 3: Build PyTorch (Incremental / develop mode)

For iterative development (faster, recompiles only changed files):

```bash
bash -c 'eval "$(conda shell.bash hook)" && source /home/sdp/fengqing/workspace/env.sh && cd /home/sdp/fengqing/workspace/pytorch && export TORCH_XPU_ARCH_LIST="pvc,bmg" && export BUILD_SEPARATE_OPS=1 && python setup.py develop 2>&1'
```

> **Note**: Use `develop` mode for C++ changes during debugging. It's significantly faster than a full rebuild.

## Step 4: Run XPU Test Cases

### Run a specific test case
```bash
bash -c 'eval "$(conda shell.bash hook)" && source /home/sdp/fengqing/workspace/env.sh && cd /home/sdp/fengqing/workspace/pytorch && pytest -v third_party/torch-xpu-ops/test/xpu/<TEST_FILE>.py -k "<TEST_NAME>" 2>&1'
```

### Example
```bash
bash -c 'eval "$(conda shell.bash hook)" && source /home/sdp/fengqing/workspace/env.sh && cd /home/sdp/fengqing/workspace/pytorch && pytest -v third_party/torch-xpu-ops/test/xpu/test_transformers_xpu.py -k "test_transformerencoder_fastpath_use_torchscript_False_enable_nested_tensor_True_use_autocast_True_d_model_12_xpu" 2>&1'
```

### Run all tests in a file
```bash
bash -c 'eval "$(conda shell.bash hook)" && source /home/sdp/fengqing/workspace/env.sh && cd /home/sdp/fengqing/workspace/pytorch && pytest -v third_party/torch-xpu-ops/test/xpu/<TEST_FILE>.py 2>&1'
```

### Run tests with verbose oneDNN output (useful for debugging oneDNN kernels)
```bash
bash -c 'eval "$(conda shell.bash hook)" && source /home/sdp/fengqing/workspace/env.sh && cd /home/sdp/fengqing/workspace/pytorch && ONEDNN_VERBOSE=1 pytest -v third_party/torch-xpu-ops/test/xpu/<TEST_FILE>.py -k "<TEST_NAME>" 2>&1'
```

## Step 5: Quick Python-only Fix (no rebuild needed)

For Python-level changes (e.g., `torch/nn/modules/transformer.py`, `torch/nn/modules/activation.py`), you can edit both:

1. **Source files** in `workspace/pytorch/torch/nn/modules/`
2. **Installed files** in `/mnt/miniforge3/envs/lfq-pt/lib/python3.10/site-packages/torch/nn/modules/`

Editing the installed location takes effect immediately without a rebuild. This is useful for quick iteration.

```bash
# Copy a fixed source file to the installed location
cp workspace/pytorch/torch/nn/modules/transformer.py /mnt/miniforge3/envs/lfq-pt/lib/python3.10/site-packages/torch/nn/modules/transformer.py
```

## Key Directories

| Path | Description |
|------|-------------|
| `workspace/pytorch/` | PyTorch source root |
| `workspace/pytorch/torch/nn/modules/` | Python NN modules (transformer, activation, etc.) |
| `workspace/pytorch/aten/src/ATen/native/mkldnn/xpu/` | C++ oneDNN XPU attention/ops |
| `workspace/pytorch/aten/src/ATen/native/mkldnn/xpu/detail/` | oneDNN XPU utilities (Utils.h, oneDNN.h) |
| `workspace/pytorch/aten/src/ATen/native/transformers/` | C++ SDPA backend selection |
| `workspace/pytorch/aten/src/ATen/native/transformers/xpu/` | XPU-specific SDPA utils (flash attention) |
| `workspace/pytorch/third_party/torch-xpu-ops/test/xpu/` | XPU test files (mirrored from upstream test/) |
| `workspace/pytorch/test/` | Upstream PyTorch tests |
| `/mnt/miniforge3/envs/lfq-pt/lib/python3.10/site-packages/torch/` | Installed torch package |

## Common Debug Patterns

See [../pytorch-xpu-knowledge-base.md](../pytorch-xpu-knowledge-base.md) for accumulated debugging patterns, tips, and domain knowledge (SDPA backend selection, oneDNN, etc.). Add new patterns to that file as they are discovered.

