"""Export the bilinear dependency-label scorer to public/model/bilinear.onnx.

For each word n and label k:

    scores[n, k] = head[n] @ weight[k] @ dep[n] + bias[k]

where head[n] is the head_tag vector of the word's predicted head and dep[n] the
word's own dept_tag vector (both [tag_dim]). weight is [n_deps, tag_dim,
tag_dim] and bias [n_deps], as written by model/onnx_export.py.

The graph is one MatMul against weight laid out as [tag_dim, n_deps * tag_dim],
then a per-label dot product with dep. Running it in onnxruntime-web is ~18x
faster than the equivalent JS loops.

Usage: python util/export_bilinear_onnx.py [weights_dir]   (default: web/)
"""

import pathlib
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "public/model/bilinear.onnx"


def build(weight, bias):
    n_deps, tag_dim, _ = weight.shape
    # w_mat[i, k * tag_dim + j] = weight[k, i, j]
    w_mat = np.ascontiguousarray(weight.transpose(1, 0, 2).reshape(tag_dim, n_deps * tag_dim))
    consts = [
        numpy_helper.from_array(w_mat, "weight"),
        numpy_helper.from_array(bias, "bias"),
        numpy_helper.from_array(np.array([-1, n_deps, tag_dim], np.int64), "per_label_shape"),
        numpy_helper.from_array(np.array([1], np.int64), "label_axis"),
        numpy_helper.from_array(np.array([2], np.int64), "tag_axis"),
    ]
    nodes = [
        helper.make_node("MatMul", ["head", "weight"], ["hw"]),                 # [N, K*D]
        helper.make_node("Reshape", ["hw", "per_label_shape"], ["hw_k"]),       # [N, K, D]
        helper.make_node("Unsqueeze", ["dep", "label_axis"], ["dep_k"]),        # [N, 1, D]
        helper.make_node("Mul", ["hw_k", "dep_k"], ["prod"]),
        helper.make_node("ReduceSum", ["prod", "tag_axis"], ["raw"], keepdims=0),  # [N, K]
        helper.make_node("Add", ["raw", "bias"], ["scores"]),
    ]
    graph = helper.make_graph(
        nodes,
        "bilinear_label_scorer",
        [
            helper.make_tensor_value_info("head", TensorProto.FLOAT, ["words", tag_dim]),
            helper.make_tensor_value_info("dep", TensorProto.FLOAT, ["words", tag_dim]),
        ],
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT, ["words", n_deps])],
        consts,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def check(model, weight, bias):
    """Compare against a direct einsum using onnx's reference evaluator."""
    from onnx.reference import ReferenceEvaluator

    rng = np.random.default_rng(0)
    tag_dim = weight.shape[1]
    head = rng.standard_normal((7, tag_dim), dtype=np.float32)
    dep = rng.standard_normal((7, tag_dim), dtype=np.float32)
    want = np.einsum("ni,kij,nj->nk", head, weight, dep) + bias
    (got,) = ReferenceEvaluator(model).run(None, {"head": head, "dep": dep})
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-3)


def main():
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "web"
    weight = np.load(src / "bilinear_weight.npy").astype(np.float32)
    bias = np.load(src / "bilinear_bias.npy").astype(np.float32)
    model = build(weight, bias)
    check(model, weight, bias)
    onnx.save(model, OUT)
    print(f"wrote {OUT} (n_deps={weight.shape[0]}, tag_dim={weight.shape[1]})")


if __name__ == "__main__":
    main()
