import { mkdir, copyFile, readdir } from 'node:fs/promises';
import { resolve } from 'node:path';

const out = resolve('public/runtime/ort-1.30.0');
await mkdir(out, { recursive: true });
for (const name of await readdir('node_modules/onnxruntime-web/dist')) {
  if (['ort.webgpu.min.mjs', 'ort.jspi.min.mjs', 'ort.min.mjs', 'ort.wasm.min.mjs'].includes(name) || name.startsWith('ort-wasm-simd-threaded')) {
    await copyFile(resolve('node_modules/onnxruntime-web/dist', name), resolve(out, name));
  }
}
await copyFile('node_modules/@huggingface/tokenizers/dist/tokenizers.mjs', resolve('public/runtime/tokenizers.mjs'));
await copyFile('node_modules/@huggingface/tokenizers/LICENSE', resolve('public/tokenizers-LICENSE.txt'));
await copyFile('node_modules/matter-js/build/matter.min.js', resolve('public/runtime/matter.min.js'));
await copyFile('node_modules/matter-js/LICENSE', resolve('public/matter-LICENSE.txt'));
console.log('Browser runtime copied to public/runtime.');
