# Nov 11 2026 — First multi-layer network on RedMulE (mini-basecaller)

## TL;DR
Ran a real **multi-layer network** end-to-end on RedMulE for the first time:
Conv→ReLU→Conv→ReLU→Flatten→FC (a tiny basecaller front-end). **0 errors out of
5**, **132,186 cycles**. Per-layer profile surfaced two findings a bare kernel
could never show: a **layout-transpose tax (~11%)** and the **FC turning
memory-bound**.

## What was built
- `experiments/make_mini_basecaller_onnx.py` — **pure-ONNX** generator (no torch;
  image has none). Emits `DeeployTest/Tests/miniBasecaller/{network.onnx,
  inputs.npz,outputs.npz}`, golden computed with onnxruntime.
- RedMulE conv constraints honored: Conv is **Conv2D only** (`PULPFPConv2DParser`),
  requires `group==1`, **all four pads equal**, and **exactly 2 inputs (no bias)**.
  So the 1D signal convs are Conv2D with a **width-1 dim, zero padding, no bias**.
- Net: in[1,1,300,1] → Conv 1→16 k[5,1] s[3,1] → ReLU → Conv 16→32 k[5,1] s[2,1]
  → ReLU → Flatten → MatMul(1536→5). Conv/MatMul on RedMulE; ReLU/Flatten on the
  PULP cluster engine (platform = RedmuleEngine + PULPClusterEngine).

## Bug hit + fix
First run crashed in the NCHW→NHWC lowering pass:
`TypeError: object of type 'NoneType' has no len()` (`_NCHWtoNHWC_fun`,
`len(inputNode.shape)`). Cause: intermediate tensors had **no shape** — the
hand-built ONNX only declared graph I/O shapes. Fix: run
`onnx.shape_inference.infer_shapes()` before saving so every tensor carries a
shape. (Lesson for all future hand-built test nets.)

## Run command
```bash
cd /workspace/ginger-v/Deeploy/DeeployTest
source /app/install/pulp-sdk/configs/siracusa.sh
export PYTHONPATH=/workspace/ginger-v/Deeploy
python3 testRunner_tiled_siracusa_w_redmule.py -t Tests/miniBasecaller --cores=8 --l1 16000 --profileTiling L2
```

## Per-layer profile (cycles, summed over tiles)
| Layer                 | ops    | tiles | in-DMA | kernel | out-DMA | total  |
|-----------------------|-------:|------:|-------:|-------:|--------:|-------:|
| transpose (conv1 in)  |      0 |     1 |    185 |    584 |     147 |    916 |
| conv1 (RedMulE)       |  15.8K |     2 |    287 |  15005 |    1037 |  16329 |
| transpose (conv1 out) |      0 |     1 |    780 |   1374 |     853 |   3007 |
| relu1 (cluster)       |   1584 |     1 |    785 |   1559 |     853 |   3197 |
| transpose (conv2 in)  |      0 |     1 |    827 |   1248 |     807 |   2882 |
| conv2 (RedMulE)       | 245.8K |     3 |   3404 |  29596 |     941 |  33941 |
| transpose (conv2 out) |      0 |     1 |    802 |   4291 |     806 |   5899 |
| relu2 (cluster)       |   1536 |     1 |    732 |   1514 |     829 |   3075 |
| fc (RedMulE)          |  15.4K |     5 |  11225 |  31862 |     240 |  43327 |
| **profiled total**    |        |       |        |        |         |**112573**|
| **full runtime**      |        |       |        |        |         |**132186**|

(~20k cycles unprofiled = init / kernel-launch glue / profiling instrumentation.)

## Findings (paper-relevant)
1. **Layout-transpose tax (~11%).** The NCHW→NHWC transpose nodes do ZERO useful
   compute but cost **12,704 cycles total**. RedMulE im2col wants NHWC, model is
   NCHW → every conv is bracketed by transposes. Scales with conv count → real
   inefficiency / optimization opportunity for conv-heavy basecallers.
2. **FC is bottleneck and going memory-bound.** fc = 43K cycles; its **input DMA
   alone is 11,225 (26%)** across 5 tiles (1536-wide activation streamed per
   tile). The "doesn't fit" pressure appearing on its own at small scale.
3. **Clean per-layer compute/memory split.** conv2 compute-bound (29.6K kernel vs
   4.3K DMA, RedMulE well-used); ReLUs tiny-compute but DMA-dominated on cluster.
   This is the per-layer feasibility data the paper needs — produced automatically.

## Next steps
- Scale the net toward a real basecaller (more/ wider convs, add the LSTM later on
  host) and watch findings #1 and #2 grow.
- Quantify the transpose tax vs conv count (is it fuseable/avoidable?).
- The off-chip ("doesn't fit") experiment: shrink L2 / widen tensors so data
  spills to L3/HyperRAM; measure off-chip traffic.
See [[nov9]] (env) and [[nov10]] (CPU-vs-RedMulE kernel data).
