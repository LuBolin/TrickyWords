import { backgroundInput } from './core.mjs';
import { assetURL, fetchAsset, workerAssetURLs } from './assets.mjs';

export const runtimeFiles = { asyncify: 'ort.webgpu.min.mjs', jspi: 'ort.jspi.min.mjs', wasm: 'ort.wasm.min.mjs' };
const wasmSuffixes = { asyncify: '.asyncify', jspi: '.jspi', wasm: '' };

export function runtimeAssetPaths(runtime) {
  if (!Object.hasOwn(runtimeFiles, runtime)) throw new Error('Unknown runtime.');
  const root = './runtime/ort-1.30.0/';
  const wasm = `${root}ort-wasm-simd-threaded${wasmSuffixes[runtime]}`;
  return [root + runtimeFiles[runtime], `${wasm}.mjs`, `${wasm}.wasm`];
}
export const runtimeChoiceKey = 'mobile-o-background-runtime-1.30.0-v11';
// Immediate release can destroy buffers before submit; deferred release avoids retaining a pool.
export const webgpuProvider = { name: 'webgpu', storageBufferCacheMode: 'lazyRelease' };

export function maskAgreement(reference, actual) {
  if (reference.length !== actual.length || !actual.length) throw new Error('Mask dimensions do not match.');
  let intersection = 0, union = 0, difference = 0;
  for (let i = 0; i < actual.length; i++) {
    if (!Number.isFinite(reference[i]) || !Number.isFinite(actual[i])) throw new Error('Invalid mask value.');
    const a = reference[i] >= .5, b = actual[i] >= .5;
    intersection += a && b; union += a || b;
    difference += Math.abs(reference[i] - actual[i]);
  }
  return { iou: union ? intersection / union : 1, meanAbsoluteError: difference / actual.length };
}

export function fastestRuntime(measurements) {
  const valid = measurements.filter(item => !item.error && item.runsMs?.length === 2
    && item.runsMs.every(ms => Number.isFinite(ms) && ms > 0)
    && item.agreement?.iou >= .995 && item.agreement.meanAbsoluteError <= .005);
  if (!valid.length) throw new Error('No runtime produced a valid background mask.');
  return valid.reduce((best, item) => item.runsMs[0] + item.runsMs[1] < best.runsMs[0] + best.runsMs[1] ? item : best).runtime;
}

export async function loadRuntime(runtime) {
  const [entry, module, wasm] = runtimeAssetPaths(runtime);
  const ort = await import(assetURL(entry));
  ort.env.wasm.wasmPaths = globalThis.trickyWordsAssets ? { mjs: assetURL(module), wasm: assetURL(wasm) }
    : new URL('./runtime/ort-1.30.0/', import.meta.url).href;
  ort.env.wasm.numThreads = runtime === 'wasm' && crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  ort.env.wasm.proxy = runtime === 'wasm';
  ort.env.webgpu.powerPreference = 'high-performance';
  return ort;
}

export async function benchmarkBackground(model, runtime, pixels, signal, files) {
  const worker = new Worker(assetURL('./calibration-worker.mjs'), { type: 'module' });
  let onAbort;
  try {
    return await new Promise((resolve, reject) => {
      onAbort = () => reject(new DOMException('Stopped by user.', 'AbortError'));
      if (signal?.aborted) { onAbort(); return; }
      signal?.addEventListener('abort', onAbort, { once: true });
      worker.onerror = error => reject(new Error(error.message || 'Calibration worker failed.'));
      worker.onmessage = ({ data }) => data.error ? reject(new Error(data.error)) : resolve(data);
      worker.postMessage({ model, runtime, pixels, files, assets: workerAssetURLs(runtimeAssetPaths(runtime)) });
    });
  } finally {
    signal?.removeEventListener('abort', onAbort);
    worker.terminate();
  }
}

export async function calibrateRuntime(model, preferred, status, checkStop, signal, files) {
  const started = performance.now();
  const image = new Image(); image.src = assetURL('./samples/slipper.png'); await image.decode();
  const canvas = document.createElement('canvas'); canvas.width = canvas.height = 512;
  const context = canvas.getContext('2d');
  context.fillStyle = 'white'; context.fillRect(0, 0, 512, 512); context.drawImage(image, 0, 0);
  const pixels = backgroundInput(context.getImageData(0, 0, 512, 512).data);
  const measurements = [];
  let reference;
  for (const runtime of [preferred, 'wasm']) {
    checkStop();
    status(`Checking ${runtime} performance during setup…`);
    const result = { runtime, runsMs: [] };
    try {
      let candidateModel = model;
      if (runtime === 'wasm') {
        const variant = await fetchAsset('./models/background-fp32-test.json', { cache: 'no-store', signal });
        if (!variant.ok) throw new Error('The FP32 CPU fallback has not been prepared.');
        candidateModel = await variant.json();
      }
      result.model = candidateModel;
      const { alpha, runsMs } = await benchmarkBackground(candidateModel, runtime, pixels, signal, files);
      result.runsMs = runsMs;
      reference ||= alpha;
      result.agreement = maskAgreement(reference, alpha);
      status(`${runtime}: ${result.runsMs.map(ms => (ms / 1000).toFixed(2) + ' s').join(', ')}.`);
    } catch (error) {
      if (error.name === 'AbortError') throw error;
      result.error = error.message || String(error);
      status(`${runtime} unavailable: ${result.error.split('\n')[0].slice(0, 180)}`);
    }
    measurements.push(result);
  }
  checkStop();
  const runtime = fastestRuntime(measurements);
  return { runtime, backgroundModel: measurements.find(item => item.runtime === runtime).model, measurements, durationMs: performance.now() - started,
    model: model.file, browser: navigator.userAgent };
}
