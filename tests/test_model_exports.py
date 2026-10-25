import json
import unittest
from pathlib import Path

import onnx
import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]


class ModelExportTests(unittest.TestCase):
    def test_compact_exports_keep_large_weights_outside_the_graph(self):
        manifest = json.loads((ROOT / 'public/models/manifest-compact.json').read_text(encoding='utf-8'))
        total = 0
        self.assertEqual(manifest['resolution'], 512)
        self.assertEqual(manifest['latent_shape'], [1, 32, 16, 16])
        for name, info in manifest['models'].items():
            with self.subTest(model=name):
                path = ROOT / 'public/models' / info['file']
                self.assertEqual(path.stat().st_size, info['bytes'])
                self.assertLess(info['bytes'], 10_000_000)
                self.assertTrue(info['externalData'])
                total += info['bytes']
                for part in info['externalData']:
                    self.assertEqual((path.parent / part['file']).stat().st_size, part['bytes'])
                    total += part['bytes']
                graph = onnx.load(path, load_external_data=False)
                self.assertTrue(any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.graph.initializer))
                if name in ['text', 'dit', 'vae']:
                    self.assertGreaterEqual(sum(n.op_type == 'MatMulNBits' for n in graph.graph.node), 50 if name == 'vae' else 100)
                if name in ['dit', 'vae']:
                    shape = [d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim]
                    self.assertEqual(shape, manifest['latent_shape'])
                if name == 'vae':
                    self.assertEqual(len(info['reconstructionChecks']), 3)
                    self.assertTrue(all(c['meanAbsolutePixelError'] < .02 and c['psnrDb'] > 30 for c in info['reconstructionChecks']))
        self.assertLess(total, 900_000_000, 'The four native compact models must stay under 900 MB.')

    def test_pointwise_lowering_preserves_convolution_and_bias(self):
        from scripts.export_compact import lower_pointwise

        rng = np.random.default_rng(4)
        for kernel in [1, 3]:
            for bias in [False, True]:
                with self.subTest(kernel=kernel, bias=bias):
                    inputs = ['input', 'weight'] + (['bias'] if bias else [])
                    weights = [onnx.numpy_helper.from_array(rng.normal(size=(8, 4, kernel, kernel)).astype('float32'), 'weight')]
                    if bias:
                        weights.append(onnx.numpy_helper.from_array(rng.normal(size=8).astype('float32'), 'bias'))
                    graph = onnx.helper.make_graph([onnx.helper.make_node('Conv', inputs, ['output'])], 'pointwise',
                        [onnx.helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [2, 4, 6, 6])],
                        [onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [2, 8, 7-kernel, 7-kernel])], weights)
                    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid('', 17)], ir_version=10)
                    values = rng.normal(size=(2, 4, 6, 6)).astype('float32')
                    expected = ort.InferenceSession(model.SerializeToString()).run(None, {'input': values})[0]
                    self.assertEqual(lower_pointwise(model), int(kernel == 1))
                    onnx.checker.check_model(model)
                    actual = ort.InferenceSession(model.SerializeToString()).run(None, {'input': values})[0]
                    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)

    def test_padding_promotion_preserves_half_precision_values(self):
        from scripts.export_models import promote_fp16_pads

        for fill in [None, np.float16(1.5)]:
            with self.subTest(fill=fill):
                constants = [onnx.numpy_helper.from_array(np.array([1, 1, 1, 1], dtype=np.int64), 'pads')]
                inputs = ['input', 'pads', '']
                if fill is not None:
                    constants.append(onnx.numpy_helper.from_array(np.array(fill), 'fill'))
                    inputs[2] = 'fill'
                graph = onnx.helper.make_graph(
                    [onnx.helper.make_node('Pad', inputs, ['output'], name='window/Pad', mode='constant')],
                    'padding-values',
                    [onnx.helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT16, [2, 2])],
                    [onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT16, [4, 4])], constants)
                model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid('', 17)], ir_version=10)
                values = np.array([[65504, -65504], [np.inf, np.nan]], dtype=np.float16)
                expected = ort.InferenceSession(model.SerializeToString()).run(None, {'input': values})[0]
                self.assertEqual(promote_fp16_pads(model), 1)
                onnx.checker.check_model(model)
                actual = ort.InferenceSession(model.SerializeToString()).run(None, {'input': values})[0]
                np.testing.assert_array_equal(actual, expected)

    def test_background_export_avoids_half_precision_pad_shader(self):
        manifest = json.loads((ROOT / 'public/models/manifest.json').read_text(encoding='utf-8'))
        model = onnx.load(ROOT / 'public/models' / manifest['models']['background']['file'])
        types = {v.name: v.type.tensor_type.elem_type for v in [*model.graph.input, *model.graph.value_info]}
        types.update({n.output[0]: a.i for n in model.graph.node if n.op_type == 'Cast' for a in n.attribute if a.name == 'to'})
        pads = [n.name for n in model.graph.node if n.op_type == 'Pad' and types.get(n.input[0]) == onnx.TensorProto.FLOAT16]
        self.assertEqual(pads, [], 'Firefox rejects the packed-f16 Pad shader.')

    def test_clip_replacement_preserves_half_precision_boundaries(self):
        from scripts.export_models import replace_diffusion_clips

        bounds = [onnx.numpy_helper.from_array(np.array(value, dtype=np.float16), name)
                  for value, name in [(-65504, 'lower'), (65504, 'upper')]]
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node('Clip', ['input', 'lower', 'upper'], ['output'], name='attention/Clip')],
            'clip-boundaries',
            [onnx.helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT16, [9])],
            [onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT16, [9])], bounds)
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid('', 17)], ir_version=10)
        values = np.array([-np.inf, -65504, -1, -0., 0., 1, 65504, np.inf, np.nan], dtype=np.float16)
        expected = ort.InferenceSession(model.SerializeToString()).run(None, {'input': values})[0]
        self.assertEqual(replace_diffusion_clips(model), 1)
        onnx.checker.check_model(model)
        actual = ort.InferenceSession(model.SerializeToString()).run(None, {'input': values})[0]
        np.testing.assert_array_equal(actual, expected)

    def test_diffusion_export_avoids_rejected_clip_shader(self):
        manifest = json.loads((ROOT / 'public/models/manifest.json').read_text(encoding='utf-8'))
        model = onnx.load(ROOT / 'public/models' / manifest['models']['dit']['file'])
        clips = [node.name for node in model.graph.node if node.op_type == 'Clip']
        self.assertEqual(clips, [], 'The external browser rejects the diffusion Clip shader.')


if __name__ == '__main__':
    unittest.main()
