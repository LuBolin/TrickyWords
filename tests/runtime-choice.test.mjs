import test from 'node:test';
import assert from 'node:assert/strict';
import { maskAgreement, fastestRuntime, loadRuntime, benchmarkBackground } from '../public/runtime-choice.mjs';

test('runtime choice rejects faster failures, invalid timings and changed masks', () => {
  const item = (runtime, runsMs, agreement = { iou: 1, meanAbsoluteError: 0 }) => ({ runtime, runsMs, agreement });
  assert.equal(fastestRuntime([
    item('asyncify', [14000, 15000]), item('wasm', [2500, 2600]),
    item('bad-mask', [1, 1], { iou: .8, meanAbsoluteError: .02 }),
    item('invalid-time', [NaN, 2]), { ...item('failed', [1, 2]), error: 'Shader failed' },
  ]), 'wasm');
  assert.throws(() => fastestRuntime([item('invalid', [0, 0])]));
});

test('mask agreement catches a changed silhouette and invalid output', async () => {
  assert.deepEqual(maskAgreement([0, 1, 1, 0], [0, 1, 1, 0]), { iou: 1, meanAbsoluteError: 0 });
  assert.equal(maskAgreement([0, 1, 1, 0], [0, 1, 0, 0]).iou, .5);
  assert.throws(() => maskAgreement([0], [NaN]));
  assert.throws(() => maskAgreement([0], []));
  await assert.rejects(loadRuntime('../unknown'), /Unknown runtime/);
});

test('calibration releases its worker after success, failure and cancellation', async t => {
  const previous = globalThis.Worker;
  const workers = [];
  let mode = 'success';
  globalThis.Worker = class {
    constructor() { workers.push(this); }
    postMessage(message) {
      this.message = structuredClone(message);
      if (mode === 'pending') return;
      queueMicrotask(() => mode === 'script' ? this.onerror({ message: 'Script failed' })
        : this.onmessage({ data: mode === 'error' ? { error: 'Invalid mask' } : { runsMs: [1, 2], alpha: new Float32Array([0, 1]) } }));
    }
    terminate() { this.terminated = true; }
  };
  t.after(() => { if (previous === undefined) delete globalThis.Worker; else globalThis.Worker = previous; });
  const files = new Map([['model.data', new File(['local weights'], 'model.data')]]);
  assert.deepEqual((await benchmarkBackground({}, 'wasm', new Float32Array(), undefined, files)).runsMs, [1, 2]);
  assert.equal(await workers[0].message.files.get('model.data').text(), 'local weights');
  mode = 'error'; await assert.rejects(benchmarkBackground({}, 'wasm', new Float32Array()), /Invalid mask/);
  mode = 'script'; await assert.rejects(benchmarkBackground({}, 'wasm', new Float32Array()), /Script failed/);
  mode = 'pending';
  const controller = new AbortController();
  const pending = benchmarkBackground({}, 'wasm', new Float32Array(), controller.signal);
  controller.abort();
  await assert.rejects(pending, { name: 'AbortError' });
  await assert.rejects(benchmarkBackground({}, 'wasm', new Float32Array(), controller.signal), { name: 'AbortError' });
  assert.ok(workers.every(worker => worker.terminated), 'No trial runtime may survive success, failure or stop.');
});
