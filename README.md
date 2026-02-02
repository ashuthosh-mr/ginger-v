<p align="center">
  <img src="assets/paragatolabs.png" width="100%">
  <em>ParaGato Labs · Edge genomics on RISC-V</em>
</p>

# Ginger-V

**Ginger-V** is a research project exploring **architecture- and memory-centric
design space exploration** for executing **DNN-based genomic basecalling
workloads** on **heterogeneous RISC-V systems** targeting the edge.

The primary application focus is **nanopore basecalling**, where neural
networks decode ionic current signals (“squiggles”) into DNA/RNA sequences
under **tight power, memory, and real-time throughput constraints**.

---

## Motivation

Modern nanopore basecallers rely on deep neural networks and are typically
accelerated using GPUs to sustain real-time sequencing rates.  
However, **edge and field deployments** demand alternatives that are:

- Energy-efficient  
- Memory-aware  
- Capable of sustained throughput matching signal generation rates  

**Key question:**

> *When and how can genomic basecalling workloads be efficiently executed on
FPGA-backed, heterogeneous RISC-V platforms?*

---

## Approach

Ginger-V studies this question through **workload-driven architectural
analysis**, rather than proposing a new accelerator.

The project decomposes basecalling inference into:

- **GEMM-dominant kernels**  
  → Offloaded to **RedMulE**, a systolic GEMM accelerator  
- **Non-GEMM operations** (control, activations, decoding, glue logic)  
  → Executed on a **RISC-V host core**, with selective use of custom
  instructions

The emphasis is on:
- Memory traffic and data movement  
- Tiling and reuse behavior  
- Arithmetic intensity and throughput limits  
- Real-time feasibility relative to nanopore signal rates  

---

## Execution Substrate

This repository leverages **RedMulE**, an outer-product systolic GEMM
accelerator developed by the **PULP platform**, as the primary acceleration
engine.

RedMulE is **not original work** of this project and is used strictly as an
experimental substrate to study accelerator behavior and system integration.

---

## Scope of Ginger-V

Ginger-V **is**:
- A design space exploration framework
- A workload-driven architectural study
- A bridge between genomics workloads and reconfigurable hardware

Ginger-V **is not**:
- A new accelerator proposal
- A full ML compiler or runtime
- A production basecaller

---

## Dependency: RedMulE

RedMulE is included as a **git submodule** and points to a fork containing
minor experimental and instrumentation changes used for analysis.

Original RedMulE repository:  
https://github.com/pulp-platform/redmule
