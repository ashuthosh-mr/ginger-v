#!/usr/bin/env python3
# make_mini_basecaller_onnx.py
# ---------------------------------------------------------------------------
# Pure-ONNX (no torch) generator for a tiny RedMulE-mappable basecaller
# skeleton. The Deeploy RedMulE backend lowers Conv as Conv2D only, and its
# conv parser requires: group==1, ALL FOUR pads equal, and exactly 2 inputs
# (no bias). So we express the 1D signal convs as Conv2D with a width-1 dim,
# zero padding, and no bias.
#
#   input [1,1,L,1] (squiggle window)
#     -> Conv2D 1->16  k[5,1] s[3,1] pad0   -> ReLU
#     -> Conv2D 16->32 k[5,1] s[2,1] pad0   -> ReLU
#     -> Flatten -> MatMul (-> N_CLASSES logits)
#
# ReLU/Flatten lower on the PULP cluster engine; Conv/MatMul on RedMulE.
#
# Run INSIDE the container:
#   docker exec gv_run python3 /workspace/ginger-v/experiments/make_mini_basecaller_onnx.py
# Emits: Deeploy/DeeployTest/Tests/miniBasecaller/{network.onnx,inputs.npz,outputs.npz}
# ---------------------------------------------------------------------------
import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

SIG_LEN = 300
C1, C2 = 16, 32
N_CLASSES = 5
SEED = 0
OUT = os.path.join(os.path.dirname(__file__), "..", "Deeploy", "DeeployTest", "Tests", "miniBasecaller")
OUT = os.path.abspath(OUT)


def conv_out(L, k, s):
    return (L - k) // s + 1


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    H1 = conv_out(SIG_LEN, 5, 3)   # 99
    H2 = conv_out(H1, 5, 2)        # 48
    flat = C2 * H2 * 1             # 1536

    # weights (float32, small)
    W1 = (rng.standard_normal((C1, 1, 5, 1)) * 0.1).astype(np.float32)
    W2 = (rng.standard_normal((C2, C1, 5, 1)) * 0.1).astype(np.float32)
    Wfc = (rng.standard_normal((flat, N_CLASSES)) * 0.1).astype(np.float32)

    inits = [
        numpy_helper.from_array(W1, "W1"),
        numpy_helper.from_array(W2, "W2"),
        numpy_helper.from_array(Wfc, "Wfc"),
    ]

    def conv(name, x, w, out, s):
        return helper.make_node("Conv", [x, w], [out], name=name,
                                kernel_shape=[5, 1], strides=[s, 1],
                                pads=[0, 0, 0, 0], dilations=[1, 1], group=1)

    nodes = [
        conv("conv1", "input", "W1", "c1", 3),
        helper.make_node("Relu", ["c1"], ["r1"], name="relu1"),
        conv("conv2", "r1", "W2", "c2", 2),
        helper.make_node("Relu", ["c2"], ["r2"], name="relu2"),
        helper.make_node("Flatten", ["r2"], ["flat"], name="flatten", axis=1),
        helper.make_node("MatMul", ["flat", "Wfc"], ["output"], name="fc"),
    ]

    g = helper.make_graph(
        nodes, "miniBasecaller", initializer=inits,
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, SIG_LEN, 1])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, N_CLASSES])],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    # Populate shapes for ALL intermediate tensors (Deeploy's layout passes need them)
    from onnx import shape_inference
    m = shape_inference.infer_shapes(m)
    onnx.save(m, os.path.join(OUT, "network.onnx"))

    # golden
    x = (rng.standard_normal((1, 1, SIG_LEN, 1)) * 1.0).astype(np.float32)
    sess = ort.InferenceSession(os.path.join(OUT, "network.onnx"), providers=["CPUExecutionProvider"])
    y = sess.run(["output"], {"input": x})[0].astype(np.float32)

    np.savez(os.path.join(OUT, "inputs.npz"), input=x)
    np.savez(os.path.join(OUT, "outputs.npz"), output=y)

    print("wrote:", OUT)
    print(f"  ops: Conv->ReLU->Conv->ReLU->Flatten->MatMul")
    print(f"  shapes: in{(1,1,SIG_LEN,1)} -> c1{(1,C1,H1,1)} -> c2{(1,C2,H2,1)} -> flat{(1,flat)} -> out{(1,N_CLASSES)}")
    print("  golden output:", y.ravel())


if __name__ == "__main__":
    main()
