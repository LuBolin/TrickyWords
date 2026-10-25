import { sigmoidMask } from './core.mjs';
import { loadRuntime, webgpuProvider } from './runtime-choice.mjs';
import { downloadModel } from './model-loading.mjs';

self.onmessage = async ({ data: { model, runtime, pixels, files, assets } }) => {
  let session, input, result;
  try {
    if (assets) globalThis.trickyWordsAssets = assets;
    const ort = await loadRuntime(runtime);
    // This trial already owns a worker. Its WASM threads remain inside this trial.
    ort.env.wasm.proxy = false;
    let downloaded = await downloadModel(model, undefined, undefined, files);
    session = await ort.InferenceSession.create(downloaded.bytes, {
      executionProviders: [runtime === 'wasm' ? 'wasm' : webgpuProvider],
      graphOptimizationLevel: 'all', externalData: downloaded.externalData,
    });
    downloaded = null;
    const runsMs = [];
    let alpha;
    for (let i = 0; i < 3; i++) {
      input = new ort.Tensor('float32', pixels.slice(), [1, 3, 512, 512]);
      const start = performance.now();
      let outputs;
      try {
        outputs = await session.run({ input_image: input });
        const logits = Object.values(outputs)[0].data;
        const elapsed = performance.now() - start;
        if (logits.length !== 512 * 512) throw new Error('Unexpected mask dimensions.');
        alpha = sigmoidMask(logits);
        const solid = alpha.reduce((sum, value) => sum + (value >= .5), 0);
        if (solid < 1000 || solid > 250000) throw new Error('The fixture mask is empty or covers the whole image.');
        if (i) runsMs.push(elapsed);
      } finally {
        for (const tensor of Object.values(outputs || {})) tensor.dispose();
        input.dispose(); input = null;
      }
    }
    result = { runsMs, alpha };
  } catch (error) { result = { error: error.message || String(error) }; }
  finally {
    input?.dispose();
    try { await session?.release(); } catch (error) { result = { error: error.message }; }
  }
  self.postMessage(result, result.alpha ? [result.alpha.buffer] : []);
};
