import { backgroundInput, sigmoidMask } from './core.mjs';
import { downloadModel } from './model-loading.mjs';
import { webgpuProvider } from './runtime-choice.mjs';

const $ = id => document.getElementById(id);
const seconds = ms => `${(ms / 1000).toFixed(2)} s`;
const precisionTest = new URLSearchParams(location.search).get('precision');
const runtimeTest = new URLSearchParams(location.search).get('runtime') || 'asyncify';
let adapter;
function log(message) {
  $('diagnostic-log').textContent += `${new Date().toLocaleTimeString()}  ${message}\n`;
}
document.addEventListener('visibilitychange', () => log(`Page visibility: ${document.visibilityState}`));

$('run').addEventListener('click', async () => {
  $('run').disabled = true;
  $('save-diagnostic').hidden = true;
  $('diagnostic-log').textContent = '';
  for (const name of ['load', 'warmup', 'run-1', 'run-2', 'run-3']) $('diagnostic-' + name).textContent = '—';
  $('diagnostic-image').getContext('2d').clearRect(0, 0, 512, 512);
  const report = { date: new Date().toISOString(), test: 'background-only', outcome: 'failed',
    browser: navigator.userAgent, runtimeVersion: '1.30.0', runtimeBuild: runtimeTest, provider: runtimeTest === 'wasm' ? 'wasm' : 'webgpu',
    jspiSupported: typeof WebAssembly.Suspending === 'function' && typeof WebAssembly.promising === 'function',
    input: 'Saved slipper composited onto white, 512x512', otherDemoTabsClosed: 'Not detectable by this page',
    visibilityAtStart: document.visibilityState, visibilityChanges: [], runsMs: [] };
  const visibilityChanged = () => report.visibilityChanges.push({ time: performance.now(), state: document.visibilityState });
  document.addEventListener('visibilitychange', visibilityChanged);
  let session, input, warmInput;
  try {
    if (!['asyncify', 'jspi', 'jsep', 'wasm'].includes(runtimeTest)) throw new Error('Unknown runtime experiment.');
    if (runtimeTest === 'jspi' && !report.jspiSupported) throw new Error('This browser does not support JSPI.');
    const runtimeFile = { jspi: 'ort.jspi.min.mjs', asyncify: 'ort.webgpu.min.mjs', jsep: 'ort.min.mjs', wasm: 'ort.wasm.min.mjs' }[runtimeTest];
    const ort = await import(`./runtime/ort-1.30.0/${runtimeFile}`);
    ort.env.wasm.wasmPaths = new URL('./runtime/ort-1.30.0/', location.href).href;
    ort.env.wasm.numThreads = runtimeTest === 'wasm' && crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    report.numThreads = ort.env.wasm.numThreads;
    ort.env.wasm.proxy = runtimeTest === 'wasm';
    report.proxyWorker = ort.env.wasm.proxy;
    log(`Runtime: ${runtimeTest}; JSPI available: ${report.jspiSupported}`);
    if (!navigator.gpu) throw new Error('This browser does not expose WebGPU.');
    if (!adapter) {
      adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter?.features.has('shader-f16')) throw new Error('The GPU must support WebGPU shader-f16.');
      if (report.provider === 'webgpu') ort.env.webgpu.adapter = adapter;
    }
    const info = adapter.info || {};
    report.device = { vendor: info.vendor, architecture: info.architecture, description: info.description,
      fallback: info.isFallbackAdapter || false, features: [...adapter.features].sort(),
      maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize };
    log(JSON.stringify(report.device, null, 2));
    const manifestResponse = await fetch(new URL('./models/manifest.json', import.meta.url), { cache: 'no-store' });
    if (!manifestResponse.ok) throw new Error('The model manifest is unavailable.');
    const manifest = await manifestResponse.json();
    report.backgroundModel = manifest.models.background;
    if (precisionTest) {
      if (precisionTest !== 'fp32') throw new Error('Unknown precision experiment.');
      const variant = await fetch(new URL('./models/background-fp32-test.json', import.meta.url), { cache: 'no-store' });
      if (!variant.ok) throw new Error('The FP32 experiment has not been exported.');
      report.backgroundModel = await variant.json();
    }
    if (!report.backgroundModel) throw new Error('The background model has not been exported.');
    log(`Model: ${report.backgroundModel.file}`);
    const image = new Image();
    image.src = new URL('./samples/slipper.png', import.meta.url).href;
    await image.decode();
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 512;
    const context = canvas.getContext('2d');
    context.fillStyle = 'white'; context.fillRect(0, 0, 512, 512); context.drawImage(image, 0, 0);
    const pixels = context.getImageData(0, 0, 512, 512);
    input = new ort.Tensor('float32', backgroundInput(pixels.data), [1, 3, 512, 512]);
    warmInput = new ort.Tensor('float32', backgroundInput(new Uint8ClampedArray(512 * 512 * 4)), [1, 3, 512, 512]);
    $('diagnostic-status').textContent = 'Loading only the background model…';
    const loadStart = performance.now();
    const downloaded = await downloadModel(report.backgroundModel);
    const nativeGpu = report.provider === 'webgpu' && runtimeTest !== 'jsep';
    session = await ort.InferenceSession.create(downloaded.bytes, { executionProviders: [nativeGpu ? webgpuProvider : report.provider], graphOptimizationLevel: 'all', externalData: downloaded.externalData });
    report.webgpuStorageBufferCacheMode = nativeGpu ? webgpuProvider.storageBufferCacheMode : null;
    report.loadMs = performance.now() - loadStart;
    $('diagnostic-load').textContent = seconds(report.loadMs);
    if (report.provider === 'webgpu') {
      const activeDevice = await ort.env.webgpu.device;
      report.activeDeviceFeatures = [...activeDevice.features].sort();
      const deviceInfo = activeDevice.adapterInfo;
      if (deviceInfo) report.deviceAfterSession = { vendor: deviceInfo.vendor, architecture: deviceInfo.architecture, description: deviceInfo.description };
    }
    async function infer(tensor) {
      const start = performance.now();
      const runInput = report.proxyWorker ? new ort.Tensor('float32', tensor.data.slice(), tensor.dims) : tensor;
      let outputs;
      try {
        outputs = await session.run({ input_image: runInput });
        const logits = Object.values(outputs)[0].data;
        const elapsed = performance.now() - start;
        if (logits.length !== 512 * 512 || !logits.every(Number.isFinite)) throw new Error('The model returned an invalid mask.');
        return { elapsed, logits: logits.slice() };
      } finally {
        for (const tensor of Object.values(outputs || {})) tensor.dispose();
        if (runInput !== tensor) runInput.dispose();
      }
    }
    $('diagnostic-status').textContent = 'Warming the background remover once…';
    report.warmupMs = (await infer(warmInput)).elapsed;
    $('diagnostic-warmup').textContent = seconds(report.warmupMs);
    log(`Warmup: ${seconds(report.warmupMs)}`);
    let result;
    for (let i = 0; i < 3; i++) {
      $('diagnostic-status').textContent = `Measuring background inference ${i + 1} / 3…`;
      result = await infer(input);
      report.runsMs.push(result.elapsed);
      $('diagnostic-run-' + (i + 1)).textContent = seconds(result.elapsed);
      log(`Run ${i + 1}: ${seconds(result.elapsed)}`);
    }
    const alpha = sigmoidMask(result.logits);
    const solidPixels = alpha.reduce((sum, value) => sum + (value >= .5), 0);
    if (solidPixels < 1000 || solidPixels > 250000) throw new Error('The fixture mask is empty or covers the whole image.');
    report.solidPixels = solidPixels;
    for (let i = 0; i < alpha.length; i++) pixels.data[i * 4 + 3] = Math.round(alpha[i] * 255);
    $('diagnostic-image').getContext('2d').putImageData(pixels, 0, 0);
    report.outcome = 'success';
    $('diagnostic-status').textContent = 'Done. Three inference measurements are ready.';
  } catch (error) {
    report.error = error.message || String(error);
    $('diagnostic-status').textContent = `Test failed: ${report.error}`;
    log(report.error);
    console.error(error);
  } finally {
    input?.dispose(); warmInput?.dispose();
    try { await session?.release(); } catch (error) { log(`Model release failed: ${error.message}`); }
    document.removeEventListener('visibilitychange', visibilityChanged);
    report.visibilityAtEnd = document.visibilityState;
    report.runtimeResources = performance.getEntriesByType('resource').map(entry => entry.name)
      .filter(name => /ort[^/]*\.(mjs|wasm)$/.test(name));
    report.log = $('diagnostic-log').textContent;
    $('save-diagnostic').href = `data:application/json;charset=utf-8,${encodeURIComponent(JSON.stringify(report, null, 2))}`;
    $('save-diagnostic').download = runtimeTest !== 'asyncify' ? `background-${runtimeTest}-diagnostic.json`
      : precisionTest === 'fp32' ? 'background-fp32-diagnostic.json' : 'background-diagnostic.json';
    $('save-diagnostic').hidden = false;
    $('run').disabled = false;
  }
});
