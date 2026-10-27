import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { modelBytes, downloadModel, selectModelFiles } from '../public/model-loading.mjs';

test('release manifests select eight-bit weights within the download budget', () => {
  const read = name => JSON.parse(readFileSync(new URL(`../public/models/${name}`, import.meta.url), 'utf8'));
  const manifest = read('manifest.json');
  assert.deepEqual(manifest, read('manifest-compact.json'));
  for (const name of ['text', 'dit', 'vae']) {
    assert.equal(manifest.models[name].bits, 8);
    assert.equal(manifest.models[name].file, `${name}-q8.onnx`);
  }
  const files = new Map();
  for (const model of [...Object.values(manifest.models), read('background-fp32-test.json')]) {
    for (const part of [model, ...model.externalData]) {
      assert.ok(Number.isSafeInteger(part.bytes) && part.bytes > 0);
      files.set(part.file, part.bytes);
    }
  }
  assert.equal(files.size, 10);
  assert.ok([...files.values()].reduce((a, b) => a + b, 0) < 1_700_000_000);
});

test('model loader counts and checks both graph and external weights', async t => {
  const model = { file: 'graph.onnx', bytes: 2, externalData: [{ file: 'weights.data', bytes: 3 }] };
  assert.equal(modelBytes(model), 5);
  const files = { 'graph.onnx': [1, 2], 'weights.data': [3, 4, 5] };
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    assert.equal(url.href, new URL(`../public/models/${url.pathname.split('/').at(-1)}`, import.meta.url).href);
    options.signal?.throwIfAborted();
    return new Response(new Uint8Array(files[url.pathname.split('/').at(-1)]));
  });
  const progress = [];
  const loaded = await downloadModel(model, undefined, (done, total) => progress.push([done, total]));
  assert.deepEqual([...loaded.bytes], [1, 2]);
  assert.equal(loaded.externalData[0].path, 'weights.data');
  assert.ok(loaded.externalData[0].data instanceof Blob);
  assert.deepEqual([...new Uint8Array(await loaded.externalData[0].data.arrayBuffer())], [3, 4, 5]);
  assert.deepEqual(progress.at(-1), [5, 5]);
  await assert.rejects(downloadModel({ ...model, externalData: [{ file: 'weights.data', bytes: 4 }] }), /does not match/);
  await assert.rejects(downloadModel({ ...model, bytes: 3 }), /incomplete/);
  await assert.rejects(downloadModel({ ...model, bytes: 1 }), /larger/);
  await assert.rejects(downloadModel({ ...model, bytes: -1 }), /Invalid/);
  const controller = new AbortController(); controller.abort();
  await assert.rejects(downloadModel(model, controller.signal), { name: 'AbortError' });
  t.mock.method(globalThis, 'fetch', async () => new Response(null, { status: 404 }));
  await assert.rejects(downloadModel(model), /HTTP 404/);
});

const model = { file: 'graph.onnx', bytes: 2, externalData: [{ file: 'weights.data', bytes: 3 }] };
const graph = () => new File([new Uint8Array([1, 2])], 'graph.onnx');
const weights = () => new File([new Uint8Array([3, 4, 5])], 'weights.data');

test('folder selection checks every required model and ignores unrelated files', () => {
  const cpu = { file: 'cpu.onnx', bytes: 1 };
  const files = [graph(), weights(), new File(['a'], 'cpu.onnx'), new File(['read me'], 'README.txt')];
  assert.deepEqual([...selectModelFiles(files, [model, cpu]).keys()], ['graph.onnx', 'weights.data', 'cpu.onnx']);
  assert.throws(() => selectModelFiles(files.slice(0, 2), [model, cpu]), /Missing cpu.onnx/);
  assert.throws(() => selectModelFiles([], [model]), /Missing graph.onnx/);
  assert.throws(() => selectModelFiles([graph(), weights(), graph()], [model]), /More than one graph.onnx/);
  assert.throws(() => selectModelFiles([graph(), new File(['bad-size'], 'weights.data')], [model]), /wrong file size/);
});

test('local model loading reads selected files without network or copying external weights', async t => {
  t.mock.method(globalThis, 'fetch', () => { throw new Error('Local models must never fall back to a network fetch.'); });
  const external = weights();
  const files = selectModelFiles([graph(), external], [model]);
  const progress = [];
  const loaded = await downloadModel(model, undefined, (done, total) => progress.push([done, total]), files);
  assert.deepEqual([...loaded.bytes], [1, 2]);
  assert.equal(loaded.externalData[0].path, 'weights.data');
  assert.equal(loaded.externalData[0].data, external);
  assert.deepEqual(progress.at(-1), [5, 5]);
  await assert.rejects(downloadModel(model, undefined, undefined, new Map()), /Missing graph.onnx/);
  await assert.rejects(downloadModel(model, undefined, undefined, new Map([['graph.onnx', weights()]])), /does not match/);
});

test('local model reads honor cancellation before and during loading', async () => {
  const files = selectModelFiles([graph(), weights()], [model]);
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(downloadModel(model, controller.signal, undefined, files), { name: 'AbortError' });
  const during = new AbortController();
  await assert.rejects(downloadModel(model, during.signal, () => during.abort(), files), { name: 'AbortError' });
});
