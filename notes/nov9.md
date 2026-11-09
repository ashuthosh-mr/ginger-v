# Nov 9 2026 — First working GEMM on RedMulE (Deeploy → GVSoC)

## TL;DR
Got a **basic floating-point GEMM running on the RedMulE accelerator end-to-end**
in simulation. Correct result, **0 errors out of 1024**, **5304 cycles**. The
Deeploy → RedMulE → GVSoC pipeline now works on this machine.

```
HELLO WORLD:
Initialized
Output:
Runtime: 5304 cycles
Errors: 0 out of 1024
✅ No errors found in Tests/testFloatGEMM
```

## What was actually wrong (and it wasn't our code)
Deeploy's code generation for RedMulE worked on the first try. The blockers were
all in the Docker **environment**:

1. **`pulp-platform/deeploy:main` is incomplete for RedMulE.**
   - Its pulp-sdk is **missing the i3c driver sources** (`drivers/i3c/src/*.c`),
     which breaks the CMake build (`No SOURCES given to target: pulp-sdk`).
   - Its GVSoC has **no RedMulE model** → the program offloads the GEMM to RedMulE
     and waits forever. This is why the first run *hung for 40 min with zero output*
     (empty `tx_uart.log`, empty trace). Not slow — hung.

2. **The correct CI image is private.** Deeploy's CI (`CI.yml`) runs RedMulE tests on
   `ghcr.io/runwangdl/deeploy:redmule`, but that image is **private** (`unauthorized`).

## The fix (reproducible)
Build **just GVSoC with the RedMulE model** inside the public image — the public
image already ships the other build deps (xtensor/xtl/xsimd, LLVM, SDK).

- GVSoC source (public fork): `https://github.com/runwangdl/gvsoc.git`
- Pinned commit (from Deeploy `Makefile` `GVSOC_COMMIT_HASH`): `35d00d15d7249daaac0de61bd8485fba128e5959`
- Build target: `siracusa` (+ `pulp.snitch.snitch_cluster_single`), install over `/app/install/gvsoc`
- After building, GVSoC gains:
  `models/gen_pulp_chips_siracusa_light_redmule_cpp_*.so` (the RedMulE device model)

Also patched Deeploy to drop the 4 missing i3c sources:
`TargetLibraries/PULPOpen/cmake/pulp-sdk-siracusa.cmake` — commented out
`cdn_print.c`, `command_list.c`, `i3c_obj_if.c`, `cps_impl.c` (i3c = serial bus,
unused by compute tests; nothing references them).

Full steps are scripted in `experiments/setup_redmule_gvsoc.sh`.

## How to run the GEMM (inside the container)
```bash
cd /workspace/ginger-v/Deeploy/DeeployTest
source /app/install/pulp-sdk/configs/siracusa.sh
export PYTHONPATH=/workspace/ginger-v/Deeploy      # Deeploy imports without pip install -e .
python3 testRunner_tiled_siracusa_w_redmule.py -t Tests/testFloatGEMM --cores=8
```

## Gotcha
The run ends with `exitcode: 24` and a `gmake *** Error` — **harmless**. It's a GVSoC
exit-code quirk *after* the program finished. The computation passed: `Errors: 0 out of
1024`, Deeploy's checker prints `✅ No errors found`, and the runner exits 0.

## Saved artifacts
- Reusable Docker image: `deeploy:redmule-local` (committed from container `deeploy_dev`;
  has the built RedMulE GVSoC + g++). Spin up with the mount:
  `docker run -it -v ~/revolution:/workspace deeploy:redmule-local bash`
- `experiments/setup_redmule_gvsoc.sh` — rebuilds the env from the stock public image (~45 min).
- `experiments/make_mini_basecaller.py` — tiny Conv1d+Linear basecaller skeleton (next step).

## Next steps
1. Run the shipped `simpleCNN` test (a real multi-layer model, step up from a bare GEMM).
2. Run the mini-basecaller (`make_mini_basecaller.py`) → confirm Conv1d + Linear map to RedMulE.
3. Start measuring: per-layer cycles + RedMulE-vs-host op split (the feasibility data).
