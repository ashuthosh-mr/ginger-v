# Ginger-V

Ginger-V is a project exploring architectural and memory-centric reasoning
for accelerating TinyML workloads on heterogeneous RISC-V systems.

This repository currently uses **RedMulE**, an  outer-product 
GEMM accelerator developed by the PULP platform, as the execution substrate
for experimentation and understanding.

The focus of Ginger-V is on **how accelerator behavior (e.g., memory traffic,
data reuse, and arithmetic intensity) can be analyzed and reasoned about**,
rather than on proposing a new accelerator design.

---

## Dependency: RedMulE

RedMulE is included as a **git submodule** and is not claimed as original work
in this project. The submodule points to a fork containing minor experimental
and instrumentation changes.

Original RedMulE repository:
https://github.com/pulp-platform/redmule

