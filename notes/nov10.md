# Nov 10 2026 — First measurement data: CPU cores vs RedMulE (profiled)

## TL;DR
Ran `testFloatMatmul` (256 outputs, 16384 MACs) through Deeploy's **built-in
tiling profiler** on two platforms — the plain RISC-V cores and RedMulE — and
got the first real per-phase cycle breakdown. **RedMulE does the multiply
~2.85× faster than all 8 cores combined.** Both correct (0/256 errors).

## How (the profiling facility, no custom harness needed)
Deeploy has profiling built in:
- `--profileUntiled` → per-op cycles (`"<op> took N cycles"`). **Only in the
  untiled flow** — the tiled runners reject it.
- `--profileTiling L2` → instruments the tile loop at that level and prints, per
  tile: **Input DMA / Kernel / Output DMA** cycles. This is the one to use for
  tiled runs; it separates data movement from compute.

Commands (inside the `deeploy:redmule-local` container, `~/revolution` at /workspace):
```bash
cd /workspace/ginger-v/Deeploy/DeeployTest
source /app/install/pulp-sdk/configs/siracusa.sh
export PYTHONPATH=/workspace/ginger-v/Deeploy
# RedMulE
python3 testRunner_tiled_siracusa_w_redmule.py -t Tests/testFloatMatmul --cores=8 --l1 8000 --profileTiling L2
# Cores only (plain Siracusa, pulp FloatMatMul kernel)
python3 testRunner_tiled_siracusa.py          -t Tests/testFloatMatmul --cores=8 --l1 8000 --profileTiling L2
```
Note: the plain-Siracusa build is much slower to compile — it builds the whole
`pulp-nn-mixed` integer kernel library.

## Results — testFloatMatmul, 256 outputs, 16384 MACs, L1=8KB, single tile
| Phase            | 8 cores (FP32) | RedMulE (FP16) | Speedup |
|------------------|---------------:|---------------:|--------:|
| Input DMA        |            757 |            757 |   1.00× |
| **Kernel**       |       **4804** |       **1683** | **2.85×** |
| Output DMA       |            167 |            172 |  ~1.00× |
| Active (DMA+kern)|           5728 |           2612 |   2.19× |
| Full runtime     |           8937 |           5847 |   1.53× |

(RedMulE numbers also seen un-profiled earlier: 4701 cycles, 0 errors.)

## What it means (with the honest caveats)
1. **DMA is identical on both (757 / ~167).** Data movement doesn't care who
   computes → it's a **fixed tax RedMulE cannot reduce**. It's ~16% of the cores'
   active time but ~36% of RedMulE's (because RedMulE shrank the compute, the
   fixed DMA is now a bigger share). This IS the offload-crossover mechanism.
2. **The 2.85× is inflated by precision.** Cores ran **FP32**, RedMulE **FP16**.
   Part of the win is half-width data, not architecture. For any paper number:
   match precision or state "FP32 core vs FP16 RedMulE" explicitly.
3. **Tiny op → RedMulE is sandbagging.** 16384 MACs doesn't saturate RedMulE's
   array; advantage should grow with size. 2.85× is a floor, not a ceiling.

## Next step
Size sweep (and later force multi-tiling with smaller L1) → plot **cores vs
RedMulE across matrix sizes**. Gives (a) where RedMulE's advantage saturates and
(b) the small-size region where DMA overhead makes offload NOT worth it. That
crossover curve is the real publishable figure — same harness, run in a loop.
See [[nov9]] for the working-environment setup this builds on.
