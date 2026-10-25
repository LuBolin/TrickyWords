import test from 'node:test';
import assert from 'node:assert/strict';
import { assetURL, fetchAsset, workerAssetURLs } from '../public/assets.mjs';
import { runtimeAssetPaths } from '../public/runtime-choice.mjs';

test('standalone metadata and runtime assets need no external fetch or URL base', async t => {
  const fetchBlob = globalThis.fetch;
  t.mock.method(globalThis, 'fetch', () => { throw new Error('Unexpected network request'); });
  globalThis.trickyWordsAssets = {
    './models/test.json': { type: 'application/json', text: '{"value":3}' },
    './test-engine.wasm': { type: 'application/wasm', base64: 'AGFzbQ==' },
    './test-module.mjs': { type: 'text/javascript', text: 'export default 4;' },
  };
  t.after(() => delete globalThis.trickyWordsAssets);
  const json = await fetchAsset('./models/test.json');
  assert.deepEqual(await json.json(), { value: 3 });
  assert.match(json.headers.get('Content-Type'), /application\/json/);
  const wasm = assetURL('./test-engine.wasm');
  assert.equal(assetURL('./test-engine.wasm'), wasm);
  assert.deepEqual([...new Uint8Array(await (await fetchBlob(wasm)).arrayBuffer())], [0, 97, 115, 109]);
  const workerAssets = structuredClone(workerAssetURLs(['./test-module.mjs', './test-engine.wasm']));
  assert.deepEqual(workerAssets['./test-engine.wasm'], { url: wasm });
  assert.equal(await (await fetchBlob(workerAssets['./test-module.mjs'].url)).text(), 'export default 4;');
  await assert.rejects(fetchAsset('./missing.json'), /Missing packaged asset/);
  assert.throws(() => assetURL('./missing.wasm'), /Missing packaged asset/);
  for (const item of Object.values(workerAssets)) URL.revokeObjectURL(item.url);
});

test('website assets retain their module-relative paths and cancellation', async t => {
  const controller = new AbortController();
  t.mock.method(globalThis, 'fetch', (url, options) => {
    assert.equal(url.href, new URL('../public/models/test.json', import.meta.url).href);
    assert.equal(options.signal, controller.signal);
    return new Response('{}');
  });
  assert.ok((await fetchAsset('./models/test.json', { signal: controller.signal })).ok);
  assert.equal(assetURL('./calibration-worker.mjs'), new URL('../public/calibration-worker.mjs', import.meta.url).href);
  assert.equal(workerAssetURLs(['./test']), undefined);
  controller.abort();
  await assert.rejects(fetchAsset('./models/test.json', { signal: controller.signal }), { name: 'AbortError' });
});

test('every supported runtime has an entry point, glue module and matching WASM binary', () => {
  for (const [runtime, suffix] of [['asyncify', '.asyncify'], ['jspi', '.jspi'], ['wasm', '']]) {
    const paths = runtimeAssetPaths(runtime);
    assert.equal(paths.length, 3);
    assert.ok(paths[1].endsWith(`ort-wasm-simd-threaded${suffix}.mjs`));
    assert.ok(paths[2].endsWith(`ort-wasm-simd-threaded${suffix}.wasm`));
  }
  assert.throws(() => runtimeAssetPaths('../unknown'), /Unknown runtime/);
});
