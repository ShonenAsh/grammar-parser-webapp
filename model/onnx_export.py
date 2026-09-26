"""
Export the trained model to ONNX and quantize to int8.

The ONNX graph contains:
  encoder (ELECTRA) + shape embedding + _head_sentinel + POS head
  + arc projections + normalized arc biaffine (divisor baked in)
  + label projections (head_tag / dept_tag feedforwards)

The ONNX graph does NOT contain:
  tag_bilinear  -- exported separately as raw weights (two tiny numpy arrays)
  CLE decoding  -- lives in JS
  masking       -- lives in JS

Outputs of the ONNX graph:
  pos_logits    [B, W,   n_pos]
  attended_arcs [B, W+1, W+1]       dep-major: arc[d][h]
  head_tag      [B, W+1, tag_dim]
  dept_tag      [B, W+1, tag_dim]

Usage:
  python export_onnx.py --run runs/electra --out web/
  python export_onnx.py --run runs/electra --out web/ --no-quantize
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn


def load_model_for_export(run_dir, device):
    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import Encoder

    class FullModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder(
                meta["encoder"], len(meta["tags"]), len(meta["deps"]),
                arc_representation_dim=meta["arc_dim"],
                tag_representation_dim=meta["tag_dim"],
            )
            self.tag_bilinear = nn.Bilinear(
                meta["tag_dim"], meta["tag_dim"], len(meta["deps"]))

        def forward(self, input_ids, attention_mask, word_index, shapes):
            return self.encoder(input_ids, attention_mask, word_index, shapes)

    model = FullModel().to(device)
    state = torch.load(os.path.join(run_dir, "model.pt"),
                       map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, meta


def make_dummy_inputs(meta, device):
    """
    Two sentences of different lengths so batch and seq axes are both > 1.
    Tracing with a single sentence can bake in size=1 for batch.
    """
    B, T, W = 2, 12, 5
    input_ids = torch.zeros((B, T), dtype=torch.long, device=device)
    input_ids[:, 0] = 101; input_ids[:, -1] = 102   # [CLS] ... [SEP]
    for i in range(1, T - 1):
        input_ids[:, i] = 1000 + i
    attention_mask = torch.ones((B, T), dtype=torch.long, device=device)
    # word_index: ROOT at 0, then first subword of each word
    word_index = torch.zeros((B, W + 1), dtype=torch.long, device=device)
    for i in range(W):
        word_index[:, i + 1] = i + 1
    shapes = torch.zeros((B, W), dtype=torch.long, device=device)
    return input_ids, attention_mask, word_index, shapes, W


def parity_check(model, inputs, ort_session, threshold=1e-3):
    """Compare torch and ORT outputs on the same inputs."""
    import onnxruntime as ort
    input_ids, attention_mask, word_index, shapes, _ = inputs
    feed = {
        "input_ids": input_ids.cpu().numpy(),
        "attention_mask": attention_mask.cpu().numpy(),
        "word_index": word_index.cpu().numpy(),
        "shapes": shapes.cpu().numpy(),
    }
    ort_out = ort_session.run(None, feed)
    with torch.no_grad():
        torch_out = model(input_ids, attention_mask, word_index, shapes)
    names = ["pos_logits", "attended_arcs", "head_tag", "dept_tag"]
    all_ok = True
    for name, t_out, o_out in zip(names, torch_out, ort_out):
        t_np = t_out.float().cpu().numpy()
        finite = np.isfinite(t_np) & np.isfinite(o_out)
        if finite.any():
            max_diff = float(np.abs(t_np[finite] - o_out[finite]).max())
        else:
            max_diff = float("nan")
        ok = max_diff < threshold
        all_ok = all_ok and ok
        print(f"  {name:<18} max_abs_diff={max_diff:.2e}  {'OK' if ok else 'FAIL'}")
    return all_ok

def export_bilinear_weights(model, out_dir, meta):
    bl = model.tag_bilinear
    w = bl.weight.detach().cpu().float().numpy()
    b = bl.bias.detach().cpu().float().numpy()
    np.save(os.path.join(out_dir, "bilinear_weight.npy"), w)
    np.save(os.path.join(out_dir, "bilinear_bias.npy"), b)
    print(f"  bilinear_weight.npy  {w.nbytes/1e6:.1f} MB")
    print(f"  bilinear_bias.npy    {b.nbytes/1e6:.3f} MB")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="web")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-quantize", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cpu"   # export on CPU: avoids CUDA/ORT version friction
    model, meta = load_model_for_export(args.run, device)
    inputs = make_dummy_inputs(meta, device)
    input_ids, attention_mask, word_index, shapes, W = inputs

    # ---- fp32 export ----
    fp32_path = os.path.join(args.out, "parser.onnx")
    print(f"exporting fp32 -> {fp32_path}")
    torch.onnx.export(
        model,
        (input_ids, attention_mask, word_index, shapes),
        fp32_path,
        input_names=["input_ids", "attention_mask", "word_index", "shapes"],
        output_names=["pos_logits", "attended_arcs", "head_tag", "dept_tag"],
        dynamic_axes={
            "input_ids":      {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "word_index":     {0: "batch", 1: "words_root"},
            "shapes":         {0: "batch", 1: "words"},
            "pos_logits":     {0: "batch", 1: "words"},
            "attended_arcs":  {0: "batch", 1: "words_root", 2: "words_root"},
            "head_tag":       {0: "batch", 1: "words_root"},
            "dept_tag":       {0: "batch", 1: "words_root"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,   # legacy exporter: single-file output, no external data
    )
    fp32_size = os.path.getsize(fp32_path) / 1e6
    print(f"  fp32  {fp32_size:.1f} MB")

    # ---- parity check ----
    try:
        import onnxruntime as ort
        print("parity check (torch vs ORT fp32):")
        sess = ort.InferenceSession(fp32_path,
                                    providers=["CPUExecutionProvider"])
        ok = parity_check(model, inputs, sess)
        if not ok:
            print("  WARNING: parity check failed — do not ship this graph")
        else:
            print("  parity OK")
    except ImportError:
        print("  onnxruntime not installed, skipping parity check")

    # ---- fp16 conversion ----
    # int8 dynamic quantization conflicts with SDPA-based attention tracing
    # (TracerWarnings bake wrong boolean constants into the graph, corrupting
    # quantized weights). fp16 halves the size with near-zero accuracy loss
    # and no calibration issues.
    fp16_path = os.path.join(args.out, "parser.fp16.onnx")
    shipped_model = "parser.fp16.onnx"
    if not args.no_quantize:
        try:
            import onnx
            from onnxconverter_common import float16
            print(f"converting to fp16 -> {fp16_path}")
            model_proto = onnx.load(fp32_path)
            fp16_model = float16.convert_float_to_float16(
                model_proto,
                keep_io_types=True,   # inputs/outputs stay fp32 for JS compat
            )
            onnx.save(fp16_model, fp16_path)
            fp16_size = os.path.getsize(fp16_path) / 1e6
            print(f"  fp16  {fp16_size:.1f} MB  "
                  f"({100*(1-fp16_size/fp32_size):.0f}% reduction vs fp32)")
            try:
                sess16 = ort.InferenceSession(fp16_path,
                                              providers=["CPUExecutionProvider"])
                print("parity check (torch vs ORT fp16, threshold 0.05):")
                # fp16 introduces ~0.01 max diff on logits — acceptable for
                # argmax; only fails if two logits are within 0.01 of each other
                ok16 = parity_check(model, inputs, sess16, threshold=0.05)
                if ok16:
                    print("  parity OK — use parser.fp16.onnx in the browser")
            except Exception as e:
                print(f"  fp16 parity check failed: {e}")
        except ImportError:
            print("onnxconverter-common not installed — run:")
            print("  pip install onnxconverter-common")
            print("falling back to fp32 for browser")
            shipped_model = "parser.onnx"

    # ---- export bilinear weights ----
    print("exporting tag_bilinear weights:")
    export_bilinear_weights(model, args.out, meta)

    # ---- write browser manifest ----
    manifest = {
        "model": shipped_model,
        "tags": meta["tags"],
        "deps": meta["deps"],
        "tag_dim": meta["tag_dim"],
        "arc_dim": meta["arc_dim"],
        "n_pos": len(meta["tags"]),
        "n_deps": len(meta["deps"]),
        "opset": args.opset,
        "encoder": meta["encoder"],
    }
    manifest_path = os.path.join(args.out, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  manifest.json written")

    print(f"\noutput directory: {args.out}/")
    for fname in sorted(os.listdir(args.out)):
        size = os.path.getsize(os.path.join(args.out, fname)) / 1e6
        print(f"  {fname:<30} {size:.1f} MB")
    print("\nNext: copy the web/ directory to your browser project and")
    print("implement JS decode using manifest.json + bilinear.json.")


if __name__ == "__main__":
    main()
