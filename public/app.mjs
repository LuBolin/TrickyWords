import { Tokenizer } from './runtime/tokenizers.mjs';
import { validateWord, wordKey, makePrompt, normalNoise, schedulerStep, packConditioning, imagePixels, applyMask, backgroundInput, sigmoidMask } from './core.mjs';
import { showCollision, clearCollisionPreview, collisionFailed } from './collision-preview.mjs';
import { runtimeFiles, runtimeChoiceKey, loadRuntime, calibrateRuntime, webgpuProvider } from './runtime-choice.mjs';
import { modelBytes, downloadModel, selectModelFiles } from './model-loading.mjs';
import { fetchAsset } from './assets.mjs';
import { generationProgress } from './generation-progress.mjs';

const $ = id => document.getElementById(id);
const stages = generationProgress(document);
const mb = bytes => `${(bytes / 1e6).toFixed(1)} MB`;
const seconds = ms => `${(ms / 1000).toFixed(2)} s`;
let manifest, tokenizer, current, setupTimings, ort, backgroundOrt, runtimeBuild, backgroundRuntimeBuild, runtimeCalibration;
let modelFiles, requiredModels;
const forcedRuntime = new URLSearchParams(location.search).get('runtime');
const forcedBackgroundRuntime = new URLSearchParams(location.search).get('backgroundRuntime');
const sessions = new Map();
const usedWords = new Set();
const ruleIds = ['single-word', 'no-repeat'];
// Keep rule choices when this tab is reloaded.
try {
  const saved = JSON.parse(sessionStorage.getItem('tricky-words-rules-v1'));
  for (const id of ruleIds) if (typeof saved?.[id] === 'boolean') $(id).checked = saved[id];
} catch { /* Rules still work when browser storage is unavailable. */ }
const wordOptions = () => ({ singleWord: $('single-word').checked, usedWords: $('no-repeat').checked ? usedWords : undefined });
function updateWordValidity() {
  let message = '';
  try { if ($('word').value) validateWord($('word').value, wordOptions()); }
  catch (error) { message = error.message; }
  $('word').setCustomValidity(message);
  $('word').setAttribute('aria-invalid', String(!!message));
  $('word-error').textContent = message;
  $('word-error').hidden = !message;
}
$('word').addEventListener('input', updateWordValidity);
for (const id of ruleIds) $(id).addEventListener('change', () => {
  try { sessionStorage.setItem('tricky-words-rules-v1', JSON.stringify(Object.fromEntries(ruleIds.map(id => [id, $(id).checked])))); }
  catch { /* A private tab can change rules without saving them. */ }
  updateWordValidity();
});
document.addEventListener('word-history-reset', () => {
  usedWords.clear();
  updateWordValidity();
});
updateWordValidity();
const inferenceState = busy => {
  for (const id of ['word', ...ruleIds]) $(id).disabled = busy;
  $('model-folder').disabled = busy || sessions.size > 0;
  if (busy) {
    const ready = sessions.size === 4;
    stages.reset(ready);
    $('loading-title').textContent = ready ? 'Making your object' : 'Loading the AI';
    $('loading-note').textContent = ready ? 'Your tower is paused. The object will appear when it is ready.' :
      'One-time setup. Checking this browser, then loading and warming the models.';
    $('status').textContent = ready ? 'Preparing your object…' : 'Preparing model setup…';
    $('elapsed').textContent = '0.00 s';
    $('progress').removeAttribute('value');
    $('progress').setAttribute('aria-label', ready ? 'Object generation progress' : 'AI setup progress');
    $('stop').textContent = ready ? 'Stop generation' : 'Stop setup';
    $('stop').hidden = false;
    $('loading-screen').showModal();
  } else {
    $('loading-screen').close();
    $('result-status').textContent = $('status').textContent;
    $('result-status').classList.toggle('error', $('status').classList.contains('error'));
  }
  document.dispatchEvent(new CustomEvent('inference-state', { detail: { busy, ready: sessions.size === 4 } }));
};
$('loading-screen').addEventListener('cancel', event => event.preventDefault());

$('model-folder').addEventListener('change', () => {
  if (!$('model-folder').files.length) return;
  modelFiles = undefined;
  $('model-folder-error').hidden = true;
  try {
    modelFiles = selectModelFiles($('model-folder').files, requiredModels);
    const bytes = [...modelFiles.values()].reduce((sum, file) => sum + file.size, 0);
    $('model-folder-status').textContent = `Model pack ready · ${modelFiles.size} files · ${(bytes / 1e9).toFixed(2)} GB`;
    status('Model folder checked. Click Load AI models to start setup.', 0);
  } catch (error) {
    $('model-folder').value = '';
    $('model-folder-status').textContent = 'No valid model pack selected.';
    $('model-folder-error').textContent = error.message;
    $('model-folder-error').hidden = false;
  }
  $('model-folder').setAttribute('aria-invalid', String(!modelFiles));
  $('prepare').disabled = !modelFiles;
  inferenceState(false);
});

function showModelPayload() {
  const bytes = Object.values(manifest.models).reduce((sum, model) => sum + modelBytes(model), 0);
  $('payload').textContent = `${(bytes / 1e9).toFixed(2)} GB`;
  $('load-note').textContent = `Active model files: ${(bytes / 1e9).toFixed(2)} GB. Models are read from your selected folder. First setup compares GPU and CPU background models. Nothing is uploaded.`;
}
function log(message) {
  $('log').textContent += `${new Date().toLocaleTimeString()}  ${message}\n`;
  $('log').scrollTop = $('log').scrollHeight;
}
function status(message, fraction, stage) {
  if (stage) stages.stage(stage);
  $('status').textContent = message;
  if (!current) $('result-status').textContent = message;
  if (fraction !== undefined) $('progress').value = fraction;
  log(message);
}
function startupStage(message, step) {
  status(message);
  document.dispatchEvent(new CustomEvent('startup-stage', { detail: { message, step } }));
}
function checkStop() {
  if (current.controller.signal.aborted) throw new DOMException('Stopped by user.', 'AbortError');
}
function dispose(tensors) {
  for (const tensor of Object.values(tensors)) tensor.dispose();
}
async function sessionFor(name) {
  checkStop();
  if (sessions.has(name)) return sessions.get(name);
  const model = manifest.models[name];
  const started = performance.now();
  status(`Loading ${name} · ${mb(modelBytes(model))}…`);
  let downloaded = await downloadModel(model, current.controller.signal, (done, total) => {
    $('status').textContent = `Loading ${name} · ${mb(done)} / ${mb(total)}`;
  }, modelFiles);
  checkStop();
  status(`Compiling ${name} for WebGPU…`);
  const modelRuntime = name === 'background' ? backgroundOrt : ort;
  const session = await modelRuntime.InferenceSession.create(downloaded.bytes, {
    executionProviders: [name === 'background' && backgroundRuntimeBuild === 'wasm' ? 'wasm' : webgpuProvider], graphOptimizationLevel: 'all',
    externalData: downloaded.externalData,
  });
  downloaded = null;
  current.timings.load += performance.now() - started;
  $('load-time').textContent = seconds(current.timings.load);
  if (current.controller.signal.aborted) { await session.release(); checkStop(); }
  sessions.set(name, session);
  return session;
}
async function releaseSessions() {
  const loaded = [...sessions.values()];
  sessions.clear();
  await Promise.allSettled(loaded.map(session => session.release()));
}
async function timed(name, operation) {
  const start = performance.now();
  const result = await operation();
  current.timings[name] = performance.now() - start;
  $(`${name}-time`).textContent = seconds(current.timings[name]);
  return result;
}
async function conditioning(word) {
  const session = await sessionFor('text');
  const ids = tokenizer.encode(makePrompt(word)).ids;
  if (ids.length > 512) throw new Error('The prompt exceeds the model token limit.');
  log(`Prompt length: ${ids.length} tokens.`);
  status('Encoding the word and image prompt…', .06, 'text');
  return await timed('text', async () => {
    const input = new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, ids.length]);
    try {
      const outputs = await session.run({ input_ids: input });
      const result = { conditional: outputs.conditional.data.slice(), unconditional: outputs.unconditional.data.slice(), shape: outputs.conditional.dims };
      dispose(outputs);
      return result;
    } finally { input.dispose(); }
  });
}
async function denoise(encoded, seed) {
  const session = await sessionFor('dit');
  const conditioning = new ort.Tensor('float32', packConditioning(encoded.unconditional, encoded.conditional), [2, ...encoded.shape.slice(1)]);
  try {
    return await timed('dit', async () => {
      const initial = normalNoise(32 * 16 * 16, seed);
      const history = [];
      let latent = initial;
      for (let i = 0; i < manifest.scheduler.timesteps.length; i++) {
        checkStop();
        status(`Drawing the object · step ${i + 1} / ${manifest.scheduler.timesteps.length}`, .1 + .65 * i / manifest.scheduler.timesteps.length, 'dit');
        const input = new ort.Tensor('float32', latent, manifest.latent_shape);
        const time = new ort.Tensor('float32', new Float32Array([manifest.scheduler.timesteps[i]]), [1]);
        try {
          const outputs = await session.run({ latent: input, timestep: time, conditioning });
          history.push(outputs.prediction.data.slice());
          dispose(outputs);
          if (!history[i].every(Number.isFinite)) throw new Error(`Non-finite diffusion prediction at step ${i + 1}.`);
          latent = schedulerStep(manifest.scheduler.coefficients[i], initial, history);
        } finally { input.dispose(); time.dispose(); }
      }
      return latent;
    });
  } finally { conditioning.dispose(); }
}
async function decode(latent) {
  const session = await sessionFor('vae');
  checkStop();
  status('Decoding the image pixels…', .8, 'vae');
  await timed('vae', async () => {
    const input = new ort.Tensor('float32', latent, manifest.latent_shape);
    try {
      const outputs = await session.run({ latent: input });
      const pixels = imagePixels(outputs.image.data, 512, 512);
      $('original').getContext('2d').putImageData(new ImageData(pixels, 512, 512), 0, 0);
      $('original-placeholder').hidden = true;
      dispose(outputs);
    } finally { input.dispose(); }
  });
}
async function removeBackground() {
  const session = await sessionFor('background');
  checkStop();
  status('Removing the background…', .92, 'background');
  await timed('background', async () => {
    const result = $('original').getContext('2d').getImageData(0, 0, 512, 512);
    const input = new backgroundOrt.Tensor('float32', backgroundInput(result.data), [1, 3, 512, 512]);
    try {
      const outputs = await session.run({ input_image: input });
      try { applyMask(result.data, sigmoidMask(Object.values(outputs)[0].data)); }
      finally { dispose(outputs); }
      $('cutout').getContext('2d').putImageData(result, 0, 0);
      $('cutout-placeholder').hidden = true;
    } finally { input.dispose(); }
  });
}

async function warmModels() {
  const ids = manifest.tokenizer_check.ids;
  const text = await sessionFor('text');
  status('Warming text conditioning…');
  const tokens = new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, ids.length]);
  let encoded;
  try {
    const outputs = await text.run({ input_ids: tokens });
    encoded = new ort.Tensor('float32', packConditioning(outputs.unconditional.data, outputs.conditional.data), [2, ...outputs.conditional.dims.slice(1)]);
    dispose(outputs);
  } finally { tokens.dispose(); }
  const latent = new ort.Tensor('float32', new Float32Array(8192), manifest.latent_shape);
  const timestep = new ort.Tensor('float32', new Float32Array([manifest.scheduler.timesteps[0]]), [1]);
  try {
    const dit = await sessionFor('dit');
    status('Warming diffusion kernels…');
    dispose(await dit.run({ latent, timestep, conditioning: encoded }));
    const vae = await sessionFor('vae');
    status('Warming the VAE decoder…');
    dispose(await vae.run({ latent }));
    const background = await sessionFor('background');
    status('Warming background removal…');
    const input = new backgroundOrt.Tensor('float32', backgroundInput(new Uint8ClampedArray(512 * 512 * 4)), [1, 3, 512, 512]);
    try { dispose(await background.run({ input_image: input })); }
    finally { input.dispose(); }
  } finally { latent.dispose(); timestep.dispose(); encoded.dispose(); }
}

$('prepare').addEventListener('click', async () => {
  if (current || !modelFiles) return;
  const started = performance.now();
  current = { controller: new AbortController(), timings: { load: 0 }, calibrationMs: 0 };
  inferenceState(true);
  $('prepare').disabled = true;
  $('stop').hidden = false;
  $('status').classList.remove('error');
  const timer = setInterval(() => { $('elapsed').textContent = seconds(performance.now() - started); }, 250);
  try {
    if (!runtimeCalibration && !forcedRuntime && !forcedBackgroundRuntime) {
      status('Checking inference speed for this browser. This is part of setup…');
      runtimeCalibration = await calibrateRuntime(manifest.models.background, runtimeBuild, status, checkStop, current.controller.signal, modelFiles);
      current.calibrationMs = runtimeCalibration.durationMs;
      // Trial workers are terminated; staying here preserves access to the selected folder.
      try {
        sessionStorage.setItem(runtimeChoiceKey, JSON.stringify(runtimeCalibration));
      } catch {
        log('Runtime preference could not be saved. Continuing setup in this page.');
      }
      backgroundRuntimeBuild = runtimeCalibration.runtime;
      backgroundOrt = backgroundRuntimeBuild === runtimeBuild ? ort : await loadRuntime(backgroundRuntimeBuild);
      manifest.models.background = runtimeCalibration.backgroundModel;
      showModelPayload();
    }
    await warmModels();
    checkStop();
    setupTimings = { load: current.timings.load, total: performance.now() - started, calibration: current.calibrationMs };
    $('setup-time').textContent = seconds(setupTimings.total);
    $('load-time').textContent = seconds(0);
    $('prepare').hidden = true;
    status('Models are loaded and warm. Enter a word to measure generation.', 0);
  } catch (error) {
    await releaseSessions();
    status(error.name === 'AbortError' ? 'Setup stopped. You can load the models again.' : `Setup failed: ${error.message.split('\n')[0].slice(0, 250)}`);
    log(error.message);
    console.error(error);
  } finally {
    clearInterval(timer);
    $('elapsed').textContent = seconds(performance.now() - started);
    $('prepare').disabled = false;
    $('generate').disabled = sessions.size !== 4;
    $('stop').hidden = true;
    current = null;
    inferenceState(false);
  }
});

$('form').addEventListener('submit', async event => {
  event.preventDefault();
  if (current || sessions.size !== 4) return;
  let word, seed;
  try {
    word = validateWord($('word').value, wordOptions());
    seed = Number($('seed').value);
    if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295) throw new Error('Seed must be an integer from 0 to 4294967295.');
  } catch (error) { status(error.message); return; }
  const started = performance.now();
  current = { controller: new AbortController(), timings: { load: 0 }, heapPeak: null,
    visibilityAtStart: document.visibilityState, visibilityChanges: [] };
  inferenceState(true);
  for (const id of ['text-time', 'dit-time', 'vae-time', 'background-time', 'polygon-time', 'load-time', 'total-time']) $(id).textContent = '—';
  clearCollisionPreview();
  $('load-time').textContent = seconds(0);
  $('status').classList.remove('error');
  $('generate').disabled = true;
  $('stop').hidden = false;
  $('save').hidden = true;
  $('report').hidden = true;
  for (const id of ['original', 'cutout']) {
    $(id).getContext('2d').clearRect(0, 0, 512, 512);
    $(`${id}-placeholder`).hidden = false;
  }
  const timer = setInterval(() => {
    $('elapsed').textContent = seconds(performance.now() - started);
    if (performance.memory) {
      current.heapPeak = Math.max(current.heapPeak || 0, performance.memory.usedJSHeapSize);
      $('heap').textContent = mb(current.heapPeak);
    }
  }, 250);
  let outcome = 'failed', errorMessage = null;
  log(`Run: ${word}, seed ${seed}. Inference provider: WebGPU.`);
  try {
    const encoded = await conditioning(word);
    const latent = await denoise(encoded, seed);
    await decode(latent);
    await removeBackground();
    checkStop();
    status('Tracing the collision shape…', .98, 'polygon');
    try {
      current.geometry = await timed('polygon', () => showCollision($('cutout'), { label: word, word, seed }));
    } catch (error) {
      current.geometryError = error.message;
      collisionFailed(error);
      log(`Collision tracing failed: ${error.message}`);
    }
    $('save').href = $('cutout').toDataURL('image/png');
    $('save').download = `${word}-${seed}.png`;
    $('save').hidden = false;
    if (current.geometry) {
      document.dispatchEvent(new CustomEvent('generated-object', {
        detail: { image: $('cutout'), geometry: current.geometry, word, seed },
      }));
      usedWords.add(wordKey(word));
      updateWordValidity();
      stages.complete();
    }
    status(current.geometry ? 'Done. Image, cutout and collision polygons are ready.' : 'Image ready, but collision tracing failed. See the shape preview below.', 1);
    outcome = 'success';
  } catch (error) {
    errorMessage = error.message || String(error);
    status(error.name === 'AbortError' ? 'Stopped. You can start a new run.' : `Run failed: ${errorMessage.split('\n')[0].slice(0, 350)}. See the run log for details.`);
    if (error.name !== 'AbortError') log(errorMessage);
    $('status').classList.toggle('error', error.name !== 'AbortError');
    outcome = error.name === 'AbortError' ? 'stopped' : 'failed';
    if (outcome === 'failed') {
      await releaseSessions();
      setupTimings = null;
      $('prepare').hidden = false;
    }
    console.error(error);
  } finally {
    clearInterval(timer);
    const total = performance.now() - started;
    $('total-time').textContent = seconds(total);
    $('elapsed').textContent = seconds(total);
    const report = { date: new Date().toISOString(), outcome, error: errorMessage, word, seed,
      model: manifest.model, sourceRevision: manifest.source_revision, precision: manifest.precision,
      runtimeVersion: '1.30.0', runtimeBuild, backgroundRuntimeBuild, runtimeCalibration, diffusionModel: manifest.models.dit, backgroundModel: manifest.models.background,
      modelSource: 'selected-folder',
      distribution: globalThis.trickyWordsAssets ? 'standalone-html' : 'website',
      textModel: manifest.models.text, vaeModel: manifest.models.vae,
      webgpuStorageBufferCacheMode: webgpuProvider.storageBufferCacheMode,
      browser: navigator.userAgent, device: manifest.device, provider: 'webgpu',
      backgroundProvider: backgroundRuntimeBuild === 'wasm' ? 'wasm' : 'webgpu',
      backgroundProxyWorker: !!backgroundOrt.env.wasm.proxy, backgroundThreads: backgroundOrt.env.wasm.numThreads,
      visibilityAtStart: current.visibilityAtStart, visibilityAtEnd: document.visibilityState, visibilityChanges: current.visibilityChanges,
      modelPayloadBytes: Object.values(manifest.models).reduce((sum, model) => sum + modelBytes(model), 0),
      timingsMs: { ...current.timings, total }, sampledJsHeapPeakBytes: current.heapPeak,
      gpuPeakMemoryBytes: null, memoryNote: 'JS heap excludes GPU allocations and may exclude WASM memory. GPU peak is unavailable.',
      setupTimingsMs: setupTimings, modelsKeptLoaded: sessions.size === 4,
      collision: current.geometry || null, collisionError: current.geometryError || null,
      steps: manifest.scheduler.timesteps.length, guidance: manifest.guidance,
      sourceResolution: manifest.resolution, outputResolution: 512, log: $('log').textContent };
    $('report').href = `data:application/json;charset=utf-8,${encodeURIComponent(JSON.stringify(report, null, 2))}`;
    $('report').download = `mobile-o-${word}-${seed}.json`;
    $('report').hidden = false;
    $('generate').disabled = sessions.size !== 4;
    $('stop').hidden = true;
    current = null;
    inferenceState(false);
  }
});
$('stop').addEventListener('click', () => {
  current?.controller.abort();
  $('status').textContent = 'Stopping after the current GPU operation…';
});
document.addEventListener('visibilitychange', () => {
  if (current?.visibilityChanges) current.visibilityChanges.push({ time: performance.now(), state: document.visibilityState });
});

async function initialize() {
  try {
    startupStage('Checking WebGPU support…', 1);
    if (!navigator.gpu) throw new Error('This browser does not expose WebGPU. Try current Chrome or Edge with hardware acceleration enabled.');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('WebGPU could not find a GPU adapter.');
    if (!adapter.features.has('shader-f16')) throw new Error('This GPU/browser does not support shader-f16, which these model exports need.');
    const info = adapter.info || {};
    const device = { vendor: info.vendor, architecture: info.architecture, description: info.description,
      maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      shaderF16: true, fallback: info.isFallbackAdapter || false };
    $('device').textContent = 'WebGPU + FP16 available';
    log(JSON.stringify(device, null, 2));
    startupStage('Reading model information…', 2);
    const response = await fetchAsset('./models/manifest.json', { cache: 'no-store' });
    if (!response.ok) throw new Error('Model exports are not ready. Follow README.md to prepare them.');
    manifest = await response.json();
    const variant = await fetchAsset('./models/background-fp32-test.json', { cache: 'no-store' });
    if (!variant.ok) throw new Error('The FP32 CPU fallback information is missing.');
    const cpuBackgroundModel = await variant.json();
    requiredModels = [...Object.values(manifest.models), cpuBackgroundModel];
    try {
      const saved = JSON.parse(sessionStorage.getItem(runtimeChoiceKey));
      if (saved?.model === manifest.models.background.file && saved.browser === navigator.userAgent && Object.hasOwn(runtimeFiles, saved.runtime)) runtimeCalibration = saved;
    } catch { /* Storage can be unavailable; report a setup error if saving calibration also fails. */ }
    const jspiSupported = typeof WebAssembly.Suspending === 'function' && typeof WebAssembly.promising === 'function';
    if (forcedRuntime && !['asyncify', 'jspi'].includes(forcedRuntime)) throw new Error('Mobile-O requires the native WebGPU runtime.');
    runtimeBuild = forcedRuntime || (jspiSupported ? 'jspi' : 'asyncify');
    startupStage('Loading the AI engine…', 3);
    ort = await loadRuntime(runtimeBuild);
    ort.env.webgpu.adapter = adapter;
    backgroundRuntimeBuild = forcedBackgroundRuntime || runtimeCalibration?.runtime || runtimeBuild;
    if (backgroundRuntimeBuild === 'wasm') {
      manifest.models.background = cpuBackgroundModel;
    }
    backgroundOrt = backgroundRuntimeBuild === runtimeBuild ? ort : await loadRuntime(backgroundRuntimeBuild);
    if (backgroundRuntimeBuild !== 'wasm') backgroundOrt.env.webgpu.adapter = adapter;
    log(`Mobile-O runtime: ${runtimeBuild}. Background runtime: ${backgroundRuntimeBuild}.`);
    if (runtimeCalibration) log(`Setup calibration: ${JSON.stringify(runtimeCalibration.measurements)}`);
    manifest.device = device;
    $('dit-label').textContent = `Denoising · ${manifest.scheduler.timesteps.length} steps`;
    for (const name of ['text', 'dit', 'vae', 'background']) if (!manifest.models[name]) throw new Error(`Missing ${name} export. Model preparation is still incomplete.`);
    log(`Browser: ${navigator.userAgent}\nDiffusion export: ${manifest.models.dit.file}\nBackground export: ${manifest.models.background.file}`);
    showModelPayload();
    startupStage('Preparing word input…', 4);
    const configs = await Promise.all(['tokenizer.json', 'tokenizer_config.json'].map(async file => {
      const response = await fetchAsset(`./models/tokenizer/${file}`);
      if (!response.ok) throw new Error(`Missing tokenizer file: ${file}`);
      return response.json();
    }));
    tokenizer = new Tokenizer(...configs);
    const ids = tokenizer.encode(manifest.tokenizer_check.text).ids;
    if (JSON.stringify(ids) !== JSON.stringify(manifest.tokenizer_check.ids)) throw new Error('Browser tokenizer does not match the Python reference.');
    log('Tokenizer parity check passed.');
    status('Choose your downloaded model folder, then load the AI models.', 0);
    document.dispatchEvent(new CustomEvent('startup-stage', { detail: { complete: true } }));
    $('prepare').disabled = true;
    inferenceState(false);
  } catch (error) {
    $('device').textContent = 'Test unavailable';
    $('status').classList.add('error');
    status(error.message || String(error));
    document.dispatchEvent(new CustomEvent('startup-stage', { detail: { error: true, message: error.message || String(error) } }));
    console.error(error);
  }
}
initialize();
