#!/usr/bin/env python3
# make_mini_basecaller.py
# ---------------------------------------------------------------------------
# Phase-2 de-risk artifact for Ginger-V.
#
# Emits a Deeploy test folder (network.onnx + inputs.npz + outputs.npz) for a
# TINY, FLOAT, RedMulE-mappable "basecaller skeleton":
#
#     Conv1d -> Conv1d(stride) -> Conv1d(stride) -> Flatten -> Linear
#
# This is deliberately the part of a nanopore basecaller that the Deeploy
# RedMulE backend already lowers:
#   * Conv1d  -> RedmuleFloatConvIm2Col  (im2col -> GEMM on RedMulE)
#   * Linear  -> MatMul -> RedmuleGEMM
# We OMIT the LSTM (no recurrent binding in the backend) and the CTC/CRF
# decode (non-GEMM -> host). Those are later phases, on purpose.
#
# Run this INSIDE the Deeploy container (torch lives there), e.g.:
#   docker run -it -v ~/revolution:/workspace ghcr.io/pulp-platform/deeploy:main
#   cd /workspace/ginger-v/experiments
#   python make_mini_basecaller.py
#
# It writes to:  ../Deeploy/DeeployTest/Tests/miniBasecaller/
# Then:          cd ../Deeploy/DeeployTest
#                python testRunner_tiled_siracusa_w_redmule.py -t Tests/miniBasecaller --cores=8
# ---------------------------------------------------------------------------

import os
import argparse
import numpy as np

import torch
import torch.nn as nn

# ---- knobs (keep tiny so GVSoC runs fast on the first pass) ----------------
SIG_LEN   = 300     # samples in one "squiggle" window (nanopore chunk, scaled down)
C1, C2, C3 = 16, 16, 32
N_CLASSES = 5       # {A, C, G, T, blank}  (per-window logits for v0)
OPSET     = 13      # bump/lower if Deeploy's importer complains
SEED      = 0

class MiniBasecaller(nn.Module):
    """Conv feature extractor + linear head. Bonito-shaped, LSTM removed."""
    def __init__(self):
        super().__init__()
        # Bonito's first conv downsamples; we mirror that with strided convs.
        self.conv1 = nn.Conv1d(1,  C1, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(C1, C2, kernel_size=5, stride=3, padding=2)
        self.conv3 = nn.Conv1d(C2, C3, kernel_size=5, stride=2, padding=2)
        self.act   = nn.ReLU()
        # length after two strided convs (stride 3 then 2), padding 'same'-ish
        tout = (SIG_LEN + 3 - 1) // 3      # after conv2 stride 3
        tout = (tout + 2 - 1) // 2         # after conv3 stride 2
        self.head  = nn.Linear(C3 * tout, N_CLASSES)

    def forward(self, x):                   # x: [N, 1, SIG_LEN]
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = x.flatten(1)
        return self.head(x)                 # [N, N_CLASSES]  (logits)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "..", "Deeploy", "DeeployTest", "Tests", "miniBasecaller"))
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = MiniBasecaller().eval()

    # one fixed input window; float32 to stay on RedMulE's float path
    x = torch.randn(1, 1, SIG_LEN, dtype=torch.float32)
    with torch.no_grad():
        y = model(x)

    # ---- export ONNX. Tensor names 'input'/'output' mirror simpleCNN. -------
    torch.onnx.export(
        model, x, os.path.join(out, "network.onnx"),
        input_names=["input"], output_names=["output"],
        opset_version=OPSET, do_constant_folding=True,
        dynamic_axes=None,
    )

    # ---- golden vectors. npz KEYS are 'input'/'output' (matches your harness)
    np.savez(os.path.join(out, "inputs.npz"),  input=x.numpy().astype(np.float32))
    np.savez(os.path.join(out, "outputs.npz"), output=y.numpy().astype(np.float32))

    print("wrote:", out)
    print("  network.onnx, inputs.npz (key 'input'), outputs.npz (key 'output')")
    print("  input shape :", tuple(x.shape))
    print("  output shape:", tuple(y.shape))

if __name__ == "__main__":
    main()
