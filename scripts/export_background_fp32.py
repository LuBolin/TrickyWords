"""Prepare a full-FP32 background model for the isolated performance test only."""
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'public/models'
manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
source = OUT / manifest['models']['background']['file']
target = OUT / 'background-birefnet-f32-v1.onnx'
model = onnx.load(source)


def promote(tensor):
    if tensor.data_type == onnx.TensorProto.FLOAT16:
        tensor.CopyFrom(onnx.numpy_helper.from_array(onnx.numpy_helper.to_array(tensor).astype(np.float32), tensor.name))


for tensor in model.graph.initializer:
    promote(tensor)
for value in [*model.graph.input, *model.graph.output, *model.graph.value_info]:
    if value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
        value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
for node in model.graph.node:
    for attribute in node.attribute:
        if attribute.type == onnx.AttributeProto.GRAPH:
            raise ValueError('Nested graphs need an explicit precision conversion.')
        if attribute.type == onnx.AttributeProto.TENSOR:
            promote(attribute.t)
        if node.op_type == 'Cast' and attribute.name == 'to' and attribute.i == onnx.TensorProto.FLOAT16:
            attribute.i = onnx.TensorProto.FLOAT
onnx.checker.check_model(model)
onnx.save(model, target)
del model

options = ort.SessionOptions()
options.intra_op_num_threads = 6
options.log_severity_level = 3
reference = ort.InferenceSession(str(source), sess_options=options, providers=['CPUExecutionProvider'])
candidate = ort.InferenceSession(str(target), sess_options=options, providers=['CPUExecutionProvider'])
checks = []
for name in ['slipper', 'teapot', 'orange']:
    rgba = np.asarray(Image.open(ROOT / f'public/samples/{name}.png').convert('RGBA'), dtype=np.float32) / 255
    rgb = rgba[:, :, :3] * rgba[:, :, 3:] + 1 - rgba[:, :, 3:]
    pixels = ((rgb - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)).transpose(2, 0, 1)[None]
    old = reference.run(None, {'input_image': pixels})[0]
    new = candidate.run(None, {'input_image': pixels})[0]
    assert np.isfinite(new).all(), 'The FP32 model returned non-finite logits.'
    old_mask, new_mask = old >= 0, new >= 0
    iou = float(np.logical_and(old_mask, new_mask).sum() / max(np.logical_or(old_mask, new_mask).sum(), 1))
    alpha_error = float(np.abs(1 / (1 + np.exp(-np.clip(old, -80, 80))) - 1 / (1 + np.exp(-np.clip(new, -80, 80)))).mean())
    assert iou > .995 and alpha_error < .005, 'The precision change changed the cutout too much.'
    checks.append({'sample': name, 'maskIntersectionOverUnion': iou, 'meanAbsoluteAlphaError': alpha_error})
    print(checks[-1], flush=True)

metadata = {**manifest['models']['background'], 'file': target.name, 'bytes': target.stat().st_size,
            'precision': 'FP32 weights and activations', 'padding_implementation': 'FP32 padding and surrounding tensors',
            'precisionChecks': checks}
metadata.pop('externalData', None)
metadata.pop('originalFile', None)
(OUT / 'background-fp32-test.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
(ROOT / 'artifacts/background-fp32-parity.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
print(f'Prepared {target.name}: {target.stat().st_size / 1e6:.1f} MB. Main manifest unchanged.', flush=True)
