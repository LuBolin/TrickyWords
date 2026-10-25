export const modelBytes = model => model.bytes + (model.externalData || []).reduce((sum, part) => sum + part.bytes, 0);

export function selectModelFiles(files, models) {
  const expected = new Map(models.flatMap(model => [model, ...(model.externalData || [])]).map(file => [file.file, file.bytes]));
  const selected = new Map();
  for (const file of files) {
    if (!expected.has(file.name)) continue;
    if (selected.has(file.name)) throw new Error(`More than one ${file.name} found. Choose a single model pack folder.`);
    selected.set(file.name, file);
  }
  for (const [name, bytes] of expected) {
    const file = selected.get(name);
    if (!file) throw new Error(`Missing ${name}. Choose the unzipped Tricky Words model pack folder.`);
    if (file.size !== bytes) throw new Error(`${name}: wrong file size. Download the model pack that matches this game version.`);
  }
  return selected;
}

export async function downloadModel(model, signal, progress = () => {}, files) {
  let downloaded = 0;
  const total = modelBytes(model);
  async function read(file, asBlob = false) {
    signal?.throwIfAborted();
    if (!Number.isSafeInteger(file.bytes) || file.bytes <= 0) throw new Error('Invalid model file size.');
    let local, response;
    if (files) {
      local = files.get(file.file);
      if (!local) throw new Error(`Missing ${file.file} in the selected model folder.`);
      if (local.size !== file.bytes) throw new Error(`${file.file}: model file size does not match its manifest.`);
    } else {
      response = await fetch(new URL(`./models/${file.file}`, import.meta.url), { signal, cache: 'no-cache' });
      if (!response.ok) throw new Error(`Could not load ${file.file}: HTTP ${response.status}`);
    }
    // Blobs avoid cloning large weight arrays into CPU workers; JSPI also reads their ranges on demand.
    if (asBlob) {
      const blob = local || await response.blob();
      signal?.throwIfAborted();
      if (blob.size !== file.bytes) throw new Error(`${file.file}: model file size does not match its manifest.`);
      downloaded += blob.size;
      progress(downloaded, total);
      return blob;
    }
    const bytes = new Uint8Array(file.bytes);
    const reader = (local ? local.stream() : response.body).getReader();
    let offset = 0;
    try {
      while (true) {
        const { value, done } = await reader.read();
        signal?.throwIfAborted();
        if (done) break;
        if (offset + value.length > bytes.length) throw new Error(`${file.file}: model file is larger than its manifest.`);
        bytes.set(value, offset);
        offset += value.length;
        downloaded += value.length;
        progress(downloaded, total);
      }
      if (offset !== bytes.length) throw new Error(`${file.file}: incomplete model file.`);
      return bytes;
    } finally { await reader.cancel(); }
  }
  const bytes = await read(model);
  const externalData = [];
  for (const part of model.externalData || []) externalData.push({ path: part.file, data: await read(part, true) });
  return { bytes, externalData };
}
