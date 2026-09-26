"""Quantize the parser to uint8 for the browser: public/model/parser.uint8.onnx.

Dynamic quantization: MatMul weights are stored as uint8 and activations are
quantized at run time. The word embeddings (a Gather, not a MatMul) stay fp32,
so the file is about the size of the fp16 model. In onnxruntime-web (WASM) it
ran 1.5x faster than the fp16 model, for POS -0.03, UAS -0.11, LAS -0.08
points on 1,000 wiki sentences.

Input is the fp32 model written by model/onnx_export.py (parser.onnx; run it
with --no-quantize to skip the fp16 conversion).

Usage: python util/quantize_parser.py path/to/parser.onnx
"""

import os
import pathlib
import sys
import tempfile

from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

OUT = pathlib.Path(__file__).resolve().parent.parent / "public/model/parser.uint8.onnx"


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    with tempfile.TemporaryDirectory() as tmp:
        pre = os.path.join(tmp, "parser.pre.onnx")
        quant_pre_process(src, pre, skip_symbolic_shape=True)
        quantize_dynamic(pre, str(OUT), weight_type=QuantType.QUInt8, op_types_to_quantize=["MatMul"])
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
