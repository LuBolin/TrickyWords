import { build } from 'esbuild';
import { readFile, writeFile, mkdir, copyFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import assert from 'node:assert/strict';
import { runtimeAssetPaths, runtimeFiles } from '../public/runtime-choice.mjs';
import { startupFeedback } from '../public/startup.mjs';

const root = resolve('public'), out = resolve('dist/TrickyWords');
const read = (path, encoding) => readFile(join(root, path), encoding);
const bundle = async entry => (await build({ entryPoints: [join(root, entry)], bundle: true, write: false,
  platform: 'browser', format: 'iife', target: 'es2022', minify: true, legalComments: 'inline',
  define: { 'import.meta.url': 'globalThis.location.href' }, supported: { 'inline-script': true },
})).outputFiles[0].text;

const assets = {}, engineScripts = [];
await mkdir(out, { recursive: true });
for (const path of new Set(Object.keys(runtimeFiles).flatMap(runtimeAssetPaths))) {
  const bytes = await read(path);
  if (path.endsWith('.wasm')) {
    assert.equal(bytes.subarray(0, 4).toString('hex'), '0061736d', `Invalid WASM: ${path}`);
    assets[path] = { type: 'application/wasm', base64: bytes.toString('base64') };
  } else assets[path] = { type: 'text/javascript', text: bytes.toString('utf8') };
}
for (const runtime of Object.keys(runtimeFiles)) {
  const name = `engine-${runtime}.js`;
  const entries = Object.fromEntries(runtimeAssetPaths(runtime).map(path => [path, assets[path]]));
  await writeFile(join(out, name), `Object.assign(globalThis.trickyWordsAssets,${JSON.stringify(entries)});\n`);
  engineScripts.push(name);
  for (const path of runtimeAssetPaths(runtime)) delete assets[path];
}
for (const path of ['models/manifest.json', 'models/background-fp32-test.json', 'models/tokenizer/tokenizer.json', 'models/tokenizer/tokenizer_config.json']) {
  const text = await read(path, 'utf8');
  JSON.parse(text);
  assets[`./${path}`] = { type: 'application/json', text };
}
assets['./samples/slipper.png'] = { type: 'image/png', base64: (await read('samples/slipper.png')).toString('base64') };
assets['./calibration-worker.mjs'] = { type: 'text/javascript', text: await bundle('calibration-worker.mjs') };

const app = await bundle('startup.mjs');
const matter = await read('runtime/matter.min.js', 'utf8');
let html = await read('index.html', 'utf8');
// Classic scripts and embedded runtime assets avoid file:// module and WASM fetches.
for (const file of ['style.css', 'game.css']) {
  const css = await read(file, 'utf8');
  html = html.replace(`<link rel="stylesheet" href="${file}">`, () => `<style>${css}</style>`);
}
html = html.replace('href="./"', 'href="index.html"');
html = html.replace(/<p><a href="background-test.html">[\s\S]*?<\/p>/, '');
html = html.replace('  <script src="runtime/matter.min.js"></script>', () => `  <script>globalThis.trickyWordsAssets={};globalThis.trickyWordsStartup=(${startupFeedback.toString()})(document);addEventListener('error',event=>{globalThis.trickyWordsLoadError=true;globalThis.trickyWordsStartup.fail(event.error||new Error('Could not load a game file. Unzip the complete game folder before opening index.html.'));},true);</script>`);
html = html.replace('  <script type="module" src="startup.mjs"></script>', () =>
  [...engineScripts, 'game.js'].map(name => `  <script defer src="${name}"></script>`).join('\n'));
html = html.replace('<meta charset="utf-8">', `<meta charset="utf-8">\n  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'self' file: 'unsafe-inline' 'wasm-unsafe-eval' blob:; style-src 'unsafe-inline'; img-src data: blob:; connect-src blob:; worker-src blob:; form-action 'none'">`);
assert.ok(!/<link[^>]+rel="stylesheet"|<script[^>]+type="module"/i.test(html), 'Download must have no external styles or module entry points.');
await writeFile(join(out, 'game.js'), `if(!globalThis.trickyWordsLoadError){\nObject.assign(globalThis.trickyWordsAssets,${JSON.stringify(assets)});\n${matter}\n${app}\n}`);
await writeFile(join(out, 'index.html'), html);
for (const name of ['mobile-o-LICENSE.txt', 'birefnet-LICENSE.txt', 'matter-LICENSE.txt', 'onnxruntime-LICENSE.txt', 'tokenizers-LICENSE.txt']) await copyFile(join(root, name), join(out, name));
await copyFile('THIRD_PARTY.md', join(out, 'THIRD_PARTY.md'));
await copyFile('docs/PLAYER.txt', join(out, 'READ-ME.txt'));
console.log(`Built ${join(out, 'index.html')} with four local JavaScript files. Model weights stay in their separate folder.`);
