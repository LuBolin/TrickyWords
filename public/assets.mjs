const urls = new Map();

function packedAsset(path) {
  const assets = globalThis.trickyWordsAssets;
  if (!assets) return;
  if (!Object.hasOwn(assets, path)) throw new Error(`Missing packaged asset: ${path}`);
  return assets[path];
}

export async function fetchAsset(path, options = {}) {
  options.signal?.throwIfAborted();
  const asset = packedAsset(path);
  if (!asset) return fetch(new URL(path, import.meta.url), options);
  return new Response(asset.text, { headers: { 'Content-Type': asset.type } });
}

export function assetURL(path) {
  const asset = packedAsset(path);
  if (!asset) return new URL(path, import.meta.url).href;
  if (asset.url) return asset.url;
  if (!urls.has(path)) {
    const bytes = asset.base64 === undefined ? asset.text : Uint8Array.from(atob(asset.base64), char => char.charCodeAt(0));
    urls.set(path, URL.createObjectURL(new Blob([bytes], { type: asset.type })));
  }
  return urls.get(path);
}

export function workerAssetURLs(paths) {
  if (!globalThis.trickyWordsAssets) return;
  return Object.fromEntries(paths.map(path => [path, { url: assetURL(path) }]));
}
