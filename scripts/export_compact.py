"""Quantize Mobile-O constant weights; keep FP16 activations and browser fixes."""
import json
import logging
from pathlib import Path

import onnx
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'public/models'


def quantize_embedding(data, block=32):
    assert data.ndim == 2 and data.shape[1] % block == 0
    values = data.astype(np.float32).reshape(data.shape[0], -1, block)
    indices = np.abs(values).argmax(axis=-1, keepdims=True)
    extreme = np.take_along_axis(values, indices, axis=-1).squeeze(-1)
    scales = (extreme / -128).astype(data.dtype)
    divisor = np.where(scales == 0, 1, scales).astype(np.float32)
    quantized = np.rint(values / divisor[..., None]).clip(-128, 127).astype(np.int8)
    restored = quantized.astype(np.float32) * scales.astype(np.float32)[..., None]
    assert np.isfinite(restored).all()
    assert np.max(np.abs(restored - values) - np.abs(scales.astype(np.float32)[..., None])) < 1e-5
    return quantized.reshape(data.shape), scales


def embedding_nodes(graph):
    weights = {tensor.name: tensor for tensor in graph.graph.initializer}
    count = 0
    for node in graph.graph.node:
        source = weights.get(node.input[0]) if node.op_type == 'Gather' else None
        if source is None or source.data_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16) or len(source.dims) != 2:
            continue
        quantized, scales = quantize_embedding(onnx.numpy_helper.to_array(source))
        name, scale_name, zero_name = (source.name + suffix for suffix in ['_Q8', '_scales', '_zero_points'])
        # GatherBlockQuantized uses unsigned storage with an explicit midpoint for signed INT8 values.
        unsigned = (quantized.astype(np.int16) + 128).astype(np.uint8)
        graph.graph.initializer.extend([onnx.numpy_helper.from_array(unsigned, name), onnx.numpy_helper.from_array(scales, scale_name),
                                       onnx.numpy_helper.from_array(np.full(scales.shape, 128, dtype=np.uint8), zero_name)])
        axis = next((attribute.i for attribute in node.attribute if attribute.name == 'axis'), 0)
        node.CopyFrom(onnx.helper.make_node('GatherBlockQuantized', [name, node.input[1], scale_name, zero_name], list(node.output),
                      name=node.name + '_Q8', domain='com.microsoft', gather_axis=axis, quantize_axis=1, block_size=32, bits=8))
        count += 1
    used = {name for node in graph.graph.node for name in node.input} | {value.name for value in graph.graph.output}
    retained = [tensor for tensor in graph.graph.initializer if tensor.name in used]
    del graph.graph.initializer[:]
    graph.graph.initializer.extend(retained)
    return count


def lower_pointwise(graph):
    weights = {t.name: t for t in graph.graph.initializer}
    nodes, count = [], 0
    for node in graph.graph.node:
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        weight = weights.get(node.input[1]) if len(node.input) > 1 else None
        if (node.op_type != 'Conv' or weight is None or list(weight.dims[2:]) != [1, 1]
                or attrs.get('group', 1) != 1 or attrs.get('strides', [1, 1]) != [1, 1]
                or attrs.get('pads', [0, 0, 0, 0]) != [0, 0, 0, 0]
                or attrs.get('auto_pad', b'NOTSET') != b'NOTSET'):
            nodes.append(node)
            continue
        prefix = node.output[0] + '/pointwise'
        matrix = onnx.numpy_helper.to_array(weight)[:, :, 0, 0].T.copy()
        graph.graph.initializer.append(onnx.numpy_helper.from_array(matrix, prefix + '/weight'))
        graph.graph.initializer.extend([
            onnx.numpy_helper.from_array(np.array([0, 1, 2], dtype=np.int64), prefix + '/shape_indices'),
            onnx.numpy_helper.from_array(np.array([weight.dims[0]], dtype=np.int64), prefix + '/channels'),
        ])
        nodes.append(onnx.helper.make_node('Transpose', [node.input[0]], [prefix + '/input'], perm=[0, 2, 3, 1]))
        # Merge spatial rows so MatMulNBits uses its large-M tiled kernel instead of many tiny batches.
        nodes.append(onnx.helper.make_node('Flatten', [prefix + '/input'], [prefix + '/flat_input'], axis=3))
        nodes.append(onnx.helper.make_node('MatMul', [prefix + '/flat_input', prefix + '/weight'], [prefix + '/product']))
        output = prefix + '/product'
        if len(node.input) > 2:
            output = prefix + '/biased'
            nodes.append(onnx.helper.make_node('Add', [prefix + '/product', node.input[2]], [output]))
        nodes.append(onnx.helper.make_node('Shape', [prefix + '/input'], [prefix + '/input_shape']))
        nodes.append(onnx.helper.make_node('Gather', [prefix + '/input_shape', prefix + '/shape_indices'], [prefix + '/spatial_shape'], axis=0))
        nodes.append(onnx.helper.make_node('Concat', [prefix + '/spatial_shape', prefix + '/channels'], [prefix + '/output_shape'], axis=0))
        nodes.append(onnx.helper.make_node('Reshape', [output, prefix + '/output_shape'], [prefix + '/reshaped']))
        nodes.append(onnx.helper.make_node('Transpose', [prefix + '/reshaped'], list(node.output), perm=[0, 3, 1, 2]))
        count += 1
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    used = {name for n in nodes for name in n.input} | {v.name for v in graph.graph.output}
    retained = [t for t in graph.graph.initializer if t.name in used]
    del graph.graph.initializer[:]
    graph.graph.initializer.extend(retained)
    return count


def main():
    logging.getLogger('onnxruntime.quantization.matmul_nbits_quantizer').setLevel(logging.WARNING)
    manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['resolution'] == 512 and manifest['latent_shape'] == [1, 32, 16, 16]
    for name in ['text', 'dit', 'vae']:
        info = manifest['models'][name]
        source = OUT / info.get('originalFile', info.get('source', info['file']))
        graph = onnx.load(source)
        if name == 'text':
            graph = onnx.version_converter.convert_version(graph, 21)
            assert embedding_nodes(graph) == 1
        else:
            print(f'{name}: lowered {lower_pointwise(graph)} pointwise convolutions.', flush=True)
        config = DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, bits=8, op_types_to_quantize=('MatMul',))
        quantizer = MatMulNBitsQuantizer(graph, algo_config=config)
        del graph
        quantizer.process()
        onnx.checker.check_model(quantizer.model.model)
        for node in quantizer.model.model.graph.node:
            if node.op_type in ('MatMulNBits', 'GatherBlockQuantized'):
                assert next(attribute.i for attribute in node.attribute if attribute.name == 'bits') == 8
        count = sum(n.op_type in ('MatMulNBits', 'GatherBlockQuantized') for n in quantizer.model.model.graph.node)
        assert count >= (50 if name == 'vae' else 100) and quantizer.model.model.ByteSize() < source.stat().st_size * .7
        manifest['models'][name] = {'originalFile': source.name, 'quantizedOperators': count, 'blockSize': 32, 'bits': 8}
        target = f'{name}-q8.onnx'
        externalize(quantizer.model.model, OUT / target, manifest['models'][name])
        del quantizer
        if name == 'vae':
            manifest['models'][name]['reconstructionChecks'] = vae_parity(source, OUT / target)
            manifest['models'][name]['reconstructionCheckMethod'] = 'Direct FP16 comparison; measured error is also its upper bound.'
    info = manifest['models']['background']
    externalize(OUT / info.get('originalFile', info['file']), OUT / 'background-compact-v2.onnx', info)
    cpu_file = OUT / 'background-fp32-test.json'
    cpu = json.loads(cpu_file.read_text(encoding='utf-8'))
    externalize(OUT / cpu.get('originalFile', cpu['file']), OUT / 'background-compact-cpu-v2.onnx', cpu)
    cpu_file.write_text(json.dumps(cpu, indent=2), encoding='utf-8')
    manifest['precision'] = 'INT8 constant matrix and embedding weights; FP16 remaining weights and activations; original FP32 attention'
    for file in ['manifest-compact.json', 'manifest.json']:
        (OUT / file).write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def externalize(source, target, info):
    graph = onnx.load(source) if isinstance(source, Path) else source
    data = target.with_suffix('.data')
    # ONNX appends to existing external data files, so replace only this export's known data file.
    data.unlink(missing_ok=True)
    onnx.save_model(graph, target, save_as_external_data=True, all_tensors_to_one_file=True,
                    location=data.name, size_threshold=1024)
    onnx.checker.check_model(str(target))
    if isinstance(source, Path):
        info.setdefault('originalFile', source.name)
    info.update(file=target.name, bytes=target.stat().st_size,
                externalData=[{'file': data.name, 'bytes': data.stat().st_size}])
    print(f'{target.name}: {info["bytes"] / 1e6:.1f} MB graph + {data.stat().st_size / 1e6:.1f} MB weights', flush=True)


def vae_parity(source, target):
    options = ort.SessionOptions(); options.intra_op_num_threads = 6; options.log_severity_level = 3
    reference = ort.InferenceSession(str(source), options, providers=['CPUExecutionProvider'])
    candidate = ort.InferenceSession(str(target), options, providers=['CPUExecutionProvider'])
    checks = []
    for scale in [.2, 1, 2]:
        latent = np.random.default_rng(42).normal(size=(1, 32, 16, 16)).astype('float32') * scale
        wanted = reference.run(None, {'latent': latent})[0]
        actual = candidate.run(None, {'latent': latent})[0]
        assert actual.shape == (1, 3, 512, 512) and np.isfinite(actual).all()
        error = actual - wanted
        mae = float(np.abs(error).mean())
        psnr = float(-10 * np.log10(max(float(np.mean(error ** 2)), 1e-12)))
        assert mae < .02 and psnr > 30, 'Decoder quantization changed reconstruction too much.'
        checks.append({'scale': scale, 'meanAbsolutePixelErrorUpperBound': mae, 'psnrDbLowerBound': psnr})
    print(f'VAE reconstruction: {checks}', flush=True)
    return checks


if __name__ == '__main__':
    main()
