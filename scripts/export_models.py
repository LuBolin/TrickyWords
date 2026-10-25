"""Export Mobile-O's actual generation weights, without loading its vision tower.

The upstream model and connector are CC BY-NC-SA 4.0. See THIRD_PARTY.md.
Half-precision module export preserves explicit FP32 attention operations.
"""
import argparse
import gc
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from diffusers import AutoencoderDC, DPMSolverMultistepScheduler, SanaTransformer2DModel
from huggingface_hub import snapshot_download, HfApi
from safetensors import safe_open
from onnxruntime.transformers.onnx_model import OnnxModel
from transformers import AutoTokenizer, Qwen2Config, Qwen2Model
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'public/models'
CHECKPOINT = ROOT / 'checkpoints/Mobile-O'
BASE = ROOT / 'checkpoints/Sana-config'
MODEL_ID = 'Amshaker/Mobile-O-0.5B'
SANA_ID = 'Efficient-Large-Model/Sana_600M_512px_diffusers'
STEPS = 8


def prompt(word):
    caption = f'A single {word}, 2D side view, flat cartoon game sprite, centered, fully visible, plain white background, no shadow, no text.'
    return ('<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n'
            '<|im_start|>user\nPlease generate image based on the following caption: '
            + caption + '<|im_end|>\n<|im_start|>assistant\n')


def read_weights(module, prefix):
    expected = module.state_dict()
    with safe_open(CHECKPOINT / 'model.safetensors', framework='pt', device='cpu') as file:
        available = set(file.keys())
        missing = [prefix + key for key in expected if prefix + key not in available]
        if missing:
            raise ValueError(f'Missing checkpoint tensors: {missing[:5]}')
        weights = {key: file.get_tensor(prefix + key) for key in expected}
    module.load_state_dict(weights, strict=True, assign=True)
    return module.half().eval()


def scheduler_plan(config, steps=STEPS):
    scheduler = DPMSolverMultistepScheduler.from_config(config)
    if scheduler.config.thresholding:
        raise ValueError('Symbolic schedule requires thresholding=False.')
    scheduler.set_timesteps(steps)
    sample = torch.zeros(steps + 1)
    sample[0] = 1
    coefficients = []
    for index, time in enumerate(scheduler.timesteps):
        prediction = torch.zeros(steps + 1)
        prediction[index + 1] = 1
        sample = scheduler.step(prediction, time, sample).prev_sample
        coefficients.append(sample.tolist())
    # Validate the exported linear plan against independent normal scheduler calls.
    rng = torch.Generator().manual_seed(7)
    initial = torch.randn(9, generator=rng)
    predictions = [torch.randn(9, generator=rng) for _ in range(steps)]
    reference = DPMSolverMultistepScheduler.from_config(config)
    reference.set_timesteps(steps)
    value = initial.clone()
    for index, time in enumerate(reference.timesteps):
        value = reference.step(predictions[index], time, value).prev_sample
        reconstructed = sum(c * x for c, x in zip(coefficients[index], [initial, *predictions]))
        torch.testing.assert_close(reconstructed, value, atol=2e-5, rtol=2e-5)
    return {'timesteps': scheduler.timesteps.tolist(), 'coefficients': coefficients}


class TextConditioning(torch.nn.Module):
    def __init__(self, model, connector):
        super().__init__()
        self.model, self.connector = model, connector

    def forward(self, input_ids):
        length = input_ids.shape[1]
        positions = torch.arange(length, device=input_ids.device)
        causal = positions[:, None] >= positions[None, :]
        mask = torch.where(causal, 0., torch.finfo(torch.float16).min).half()[None, None]
        states = self.model(input_ids=input_ids, attention_mask=mask,
                            position_ids=positions[None], use_cache=False,
                            output_hidden_states=True, return_dict=True).hidden_states
        selected = states[-self.connector.num_layers:]
        conditional = self.connector(selected)
        unconditional = self.connector(tuple(torch.zeros_like(x) for x in selected))
        return conditional.float(), unconditional.float()


class Denoiser(torch.nn.Module):
    def __init__(self, model, guidance):
        super().__init__()
        self.model = model
        self.guidance = guidance

    def forward(self, latent, timestep, conditioning):
        prediction = self.model(hidden_states=torch.cat([latent, latent]).half(), timestep=timestep.expand(2),
                          encoder_hidden_states=conditioning.half(),
                          encoder_attention_mask=None, return_dict=False)[0].float()
        unconditional, conditional = prediction.chunk(2)
        return unconditional + self.guidance * (conditional - unconditional)


class Decoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, latent):
        decoded = self.model.decode((latent / self.model.config.scaling_factor).half(), return_dict=False)[0]
        return (decoded.float() / 2 + .5).clamp(0, 1)


def replace_diffusion_clips(model):
    nodes = []
    count = 0
    for node in model.graph.node:
        if node.op_type != 'Clip':
            nodes.append(node)
            continue
        if len(node.input) != 3 or not all(node.input):
            raise ValueError(f'Expected both diffusion Clip bounds: {node.name}')
        # Some browser compilers reject ORT's packed-f16 Clip shader.
        lower_bounded = node.output[0] + '/lower_bounded'
        nodes.append(onnx.helper.make_node('Max', [node.input[0], node.input[1]], [lower_bounded], name=node.name + '/Max'))
        nodes.append(onnx.helper.make_node('Min', [lower_bounded, node.input[2]], list(node.output), name=node.name + '/Min'))
        count += 1
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return count


def promote_fp16_pads(model):
    types = {v.name: v.type.tensor_type.elem_type for v in [*model.graph.input, *model.graph.value_info]}
    nodes = []
    count = 0
    for node in model.graph.node:
        if node.op_type != 'Pad':
            nodes.append(node)
            continue
        dtype = types.get(node.input[0])
        if dtype is None:
            raise ValueError(f'Unknown Pad input dtype: {node.name}')
        if dtype != onnx.TensorProto.FLOAT16:
            nodes.append(node)
            continue
        inputs = list(node.input)
        inputs[0] = node.output[0] + '/input_fp32'
        nodes.append(onnx.helper.make_node('Cast', [node.input[0]], [inputs[0]], name=node.name + '/InputCast', to=onnx.TensorProto.FLOAT))
        if len(inputs) > 2 and inputs[2]:
            inputs[2] = node.output[0] + '/value_fp32'
            nodes.append(onnx.helper.make_node('Cast', [node.input[2]], [inputs[2]], name=node.name + '/ValueCast', to=onnx.TensorProto.FLOAT))
        padded = onnx.NodeProto()
        padded.CopyFrom(node)
        padded.name += '/FP32'
        padded.input[:] = inputs
        padded.output[0] = node.output[0] + '/padded_fp32'
        nodes.append(padded)
        nodes.append(onnx.helper.make_node('Cast', [padded.output[0]], list(node.output), name=node.name + '/OutputCast', to=onnx.TensorProto.FLOAT16))
        count += 1
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return count


def export(module, inputs, name, input_names, output_names, dynamic_axes=None, verify_existing=False):
    target = OUT / f'{name}.onnx'
    print(f'Exporting {name}...', flush=True)
    with torch.inference_mode():
        expected = module(*inputs)
        expected = expected if isinstance(expected, tuple) else (expected,)
        if not verify_existing:
            torch.onnx.export(module, inputs, str(target), input_names=input_names,
                              output_names=output_names, dynamic_axes=dynamic_axes,
                              opset_version=17, dynamo=False, do_constant_folding=True)
    graph = onnx.load(str(target))
    if name.startswith('dit'):
        print(f'Replaced {replace_diffusion_clips(graph)} diffusion Clip shaders with Max/Min.', flush=True)
    # PyTorch movedim can emit negative axes; ONNX Transpose requires positive axes.
    for node in graph.graph.node:
        if node.op_type == 'Transpose':
            for attribute in node.attribute:
                if attribute.name == 'perm':
                    rank = len(attribute.ints)
                    attribute.ints[:] = [axis + rank if axis < 0 else axis for axis in attribute.ints]
                    if sorted(attribute.ints) != list(range(rank)):
                        raise ValueError(f'Invalid transpose axes in {node.name}')
    onnx.save(graph, str(target))
    del graph
    onnx.checker.check_model(str(target))
    session = ort.InferenceSession(str(target), providers=['CPUExecutionProvider'])
    outputs = session.run(None, {name: value.numpy() for name, value in zip(input_names, inputs)})
    checks = []
    for actual, wanted in zip(outputs, expected):
        wanted = wanted.numpy()
        if not np.isfinite(actual).all():
            raise ValueError(f'{name}: non-finite ONNX output')
        relative = float(np.linalg.norm(actual.astype(np.float32) - wanted) / max(np.linalg.norm(wanted), 1e-6))
        if relative > .08:
            raise ValueError(f'{name}: ONNX relative error too large: {relative}')
        checks.append({'shape': list(actual.shape), 'relative_l2_error': relative})
    print(f'{name}: {target.stat().st_size / 1e6:.1f} MB; parity {checks}', flush=True)
    del session
    gc.collect()
    return {'file': f'{name}.onnx', 'bytes': target.stat().st_size, 'parity': checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', choices=['text', 'dit', 'vae', 'background'])
    parser.add_argument('--verify-existing', action='store_true', help='Recheck saved graphs without retracing them')
    args = parser.parse_args()
    torch.set_num_threads(6)
    torch.manual_seed(42)
    OUT.mkdir(parents=True, exist_ok=True)
    snapshot_download(MODEL_ID, local_dir=CHECKPOINT,
                      allow_patterns=['*.json', 'model.safetensors', 'merges.txt', 'vocab.json'])
    snapshot_download(SANA_ID, local_dir=BASE,
                      allow_patterns=['transformer/config.json', 'vae/config.json', 'scheduler/scheduler_config.json'])
    raw = json.loads((CHECKPOINT / 'config.json').read_text())
    configs = {name: json.loads((BASE / name / file).read_text()) for name, file in
               [('transformer', 'config.json'), ('vae', 'config.json'), ('scheduler', 'scheduler_config.json')]}
    manifest_path = OUT / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'models': {}}
    manifest.update({'model': MODEL_ID, 'source_revision': HfApi().model_info(MODEL_ID).sha,
                     'precision': 'FP16 weights with original FP32 attention calculations',
                     'scheduler': scheduler_plan(configs['scheduler']), 'guidance': 1.5,
                     'latent_shape': [1, 32, 16, 16], 'resolution': 512})
    tokenizer_dir = OUT / 'tokenizer'
    tokenizer_dir.mkdir(exist_ok=True)
    for file in CHECKPOINT.glob('*.json'):
        if file.name != 'config.json':
            shutil.copy2(file, tokenizer_dir / file.name)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    ids = tokenizer(prompt('slipper'), return_tensors='pt').input_ids
    manifest['tokenizer_check'] = {'text': prompt('slipper'), 'ids': ids[0].tolist()}
    for name in (['text', 'dit', 'vae', 'background'] if not args.only else [args.only]):
        if name == 'text':
            spec = importlib.util.spec_from_file_location('mobile_block', ROOT / 'vendor/Mobile-O/mobileo/model/mobile_block.py')
            block = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(block)
            config = Qwen2Config(**raw)
            config._attn_implementation = 'eager'
            with torch.device('meta'):
                qwen = Qwen2Model(config)
                connector = block.MobileConditioningProjector(input_dim=raw['hidden_size'], num_layers=raw['vlm_num_layers'])
            read_weights(qwen, 'model.')
            qwen.rotary_emb = Qwen2RotaryEmbedding(config=config)
            read_weights(connector, 'model.diffusion_connector.')
            module = TextConditioning(qwen, connector)
            result = export(module, (ids,), name, ['input_ids'], ['conditional', 'unconditional'],
                            {'input_ids': {1: 'sequence'}, 'conditional': {1: 'sequence'}, 'unconditional': {1: 'sequence'}}, args.verify_existing)
        elif name == 'dit':
            with torch.device('meta'):
                model = SanaTransformer2DModel.from_config(configs['transformer'])
            read_weights(model, 'model.dit.')
            module = Denoiser(model, manifest['guidance'])
            result = export(module, (torch.randn(1, 32, 16, 16), torch.tensor([700.]), torch.randn(2, ids.shape[1], 2304)),
                            'dit-cfg-minmax', ['latent', 'timestep', 'conditioning'], ['prediction'], {'conditioning': {1: 'sequence'}}, args.verify_existing)
            result['classifier_free_guidance'] = 'batched'
            result['clip_implementation'] = 'Max followed by Min, with original bounds and dtype'
        elif name == 'vae':
            with torch.device('meta'):
                model = AutoencoderDC.from_config(configs['vae'])
            model.encoder = None
            read_weights(model, 'model.vae.')
            module = Decoder(model)
            result = export(module, (torch.randn(1, 32, 16, 16) * .2,), name, ['latent'], ['image'], verify_existing=args.verify_existing)
        else:
            background_dir = ROOT / 'checkpoints/birefnet'
            snapshot_download('studioludens/birefnet-lite-512', local_dir=background_dir,
                              allow_patterns=['onnx/model_fp16.onnx', 'preprocessor_config.json', 'config.json', 'README.md', 'LICENSE'])
            target = OUT / 'background-birefnet-pad-f32.onnx'
            shutil.copy2(background_dir / 'onnx/model_fp16.onnx', target)
            graph = OnnxModel(onnx.load(str(target)))
            graph.topological_sort()
            print(f'Promoted {promote_fp16_pads(graph.model)} background Pad shaders to FP32.', flush=True)
            onnx.save(graph.model, str(target))
            onnx.checker.check_model(str(target))
            result = {'file': target.name, 'bytes': target.stat().st_size,
                      'padding_implementation': 'FP32 padding with FP16 surrounding tensors',
                      'source': 'studioludens/birefnet-lite-512', 'license': 'MIT',
                      'revision': HfApi().model_info('studioludens/birefnet-lite-512').sha}
            shutil.copy2(background_dir / 'LICENSE', ROOT / 'public/birefnet-LICENSE.txt')
        manifest['models'][name] = result
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False))
        if name != 'background':
            del module
        if name == 'text':
            del qwen, connector
        elif name != 'background':
            del model
        gc.collect()
    print('Export complete.', flush=True)


if __name__ == '__main__':
    main()
