---

# Deeploy Integration Notes

**Date:** March 10 2026

---

## Progress Summary

### 1. Deeploy environment setup using Docker

To simplify dependency management, Deeploy was installed using the official container provided by the PULP platform.

```
docker pull ghcr.io/pulp-platform/deeploy:main
```

The container was launched with the project directory mounted:

```
docker run -it -v ~/revolution:/workspace ghcr.io/pulp-platform/deeploy:main
```

This allows Deeploy to operate on the local **ginger-v** repository while keeping the environment reproducible.

Inside the container, Deeploy and its dependencies were initialized:

```
git submodule update --init --recursive
pip install -e . --extra-index-url=https://pypi.ngc.nvidia.com
```

This provides the full compiler environment required for generating inference runtimes.

---

### 2. Integration of RedMulE backend (PR #67)

The RedMulE backend for Deeploy is currently available through an upstream pull request:

```
https://github.com/pulp-platform/Deeploy/pull/67
```

The repository was checked out locally and the PR branch was activated.

This backend introduces:

* RedMulE GEMM kernel mappings
* Siracusa cluster platform configuration
* tiling-aware code generation

This enables Deeploy to generate inference code targeting:

```
Siracusa_w_redmule
```

which represents a PULP cluster architecture with a RedMulE accelerator.

---

### 3. Deeploy test network execution

Several built-in Deeploy tests were executed to validate the compiler pipeline.

| Test          | Result                      | Notes                          |
| ------------- | --------------------------- | ------------------------------ |
| Adder         | ✓ success                   | basic runtime generation       |
| testFloatGEMM | ✓ success                   | RedMulE GEMM mapping confirmed |
| simpleCNN     | ✓ success                   | convolution network compiled   |
| testGEMM      | ✗ parser failure            | unsupported graph mapping      |
| mobilenetv2   | ✗ memory allocation failure | model exceeds memory budget    |

Successful tests confirmed that Deeploy can generate kernels for:

```
GEMM layers
convolution layers
```

which are sufficient for many TinyML workloads.

---

### 4. Successful RedMulE kernel generation

The `testFloatGEMM` example generated runtime code containing the RedMulE accelerator call:

```c
Gemm_fp32_fp32_fp32_fp32_Redmule(...)
```

This confirms the following pipeline:

```
ONNX model
↓
Deeploy graph parsing
↓
hardware mapping
↓
RedMulE GEMM kernel generation
```

The generated runtime also includes:

* DMA data movement
* scratchpad memory tiling
* cluster (Siracusa 8 core) execution scheduling

---

### 5. Generated runtime architecture

The generated inference runtime follows this execution structure:

```
RunNetwork()
   ↓
_closure_L3()
   ↓
_closure()
   ↓
_cluster_fork()
   ↓
_tiling_closure()
   ↓
RedMulE GEMM kernel
```

Key runtime components observed:

```
DMA transfers between memory levels
automatic tiling loops
cluster execution orchestration
accelerator invocation
```

This confirms that Deeploy performs **memory-aware code generation** for embedded systems.

---

### 6. Memory hierarchy used by generated runtime

The runtime assumes the following hierarchy:

```
L3 memory (model inputs / outputs)
↓
DMA transfers
↓
L1 scratchpad memory
↓
RedMulE accelerator execution
```

Key runtime primitives used:

```
dory_dma_memcpy_mindims_async
dory_dma_barrier
pi_cl_team_fork
```

These coordinate data movement and cluster execution.

---

### 7. Memory footprint observations

The generated example allocates:

```
L1 scratchpad arena: 16 KB
```

Example allocation:

```
DeeployNetwork_MEMORYARENA_L1 =
    pmsis_l1_malloc(sizeof(int8_t) * 16384);
```

This confirms that Deeploy targets **very small embedded memory budgets**, consistent with TinyML deployment scenarios.

---

## Key Milestone

The following deployment pipeline has been validated:

```
ONNX model
↓
Deeploy compiler
↓
C runtime generation
↓
RedMulE GEMM kernel invocation
```

This confirms that Deeploy can automatically map neural network layers to the RedMulE accelerator.

---

## Next Steps

* Run generated kernels with **RedMulE simulation**
* Extract **cycle counts and latency measurements**
* Select a **small nanopore basecalling model**
* Evaluate **tiling strategies and memory traffic**
* Integrate Deeploy-generated kernels with the standalone **RedMulE + CV32 host architecture**

---
