"""CPU int8 Kitsune: ONNX Runtime dynamic quantisation of the MatMuls only, per
channel (convolutions stay fp32) - the model author's model_int8 recipe.

Usage (SubPlz venv): python tools/quantize_kitsune.py models/kitsune-p0.05b [...]
"""
import sys
from onnxruntime.quantization import QuantType, quantize_dynamic

for d in sys.argv[1:]:
    quantize_dynamic(f"{d}/model.fp32.onnx", f"{d}/model.int8.onnx", weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul"], per_channel=True)
    print(f"wrote {d}/model.int8.onnx")
