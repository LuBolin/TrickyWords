export function wordKey(value) {
  return value.trim().normalize('NFC').replace(/\s+/gu, ' ').toLowerCase();
}

export function validateWord(value, { singleWord = true, usedWords } = {}) {
  const word = value.trim().normalize('NFC');
  if (!word || word.length > 32 || /\p{Cc}/u.test(word)) {
    throw new Error('Enter a single line of text, up to 32 characters.');
  }
  if (singleWord && !/^[\p{L}][\p{L}\p{M}]*$/u.test(word)) {
    throw new Error('Use one word with letters only, up to 32 characters.');
  }
  if (/<\||\|>/u.test(word)) throw new Error('That text contains an unsupported marker.');
  if (usedWords?.has(wordKey(word))) throw new Error(`“${word}” has already been used this round. Choose another word.`);
  return word;
}

export function makePrompt(word) {
  const caption = `A single ${word}, 2D side view, flat cartoon game sprite, centered, fully visible, plain white background, no shadow, no text.`;
  return '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' +
    '<|im_start|>user\nPlease generate image based on the following caption: ' +
    caption + '<|im_end|>\n<|im_start|>assistant\n';
}

export function normalNoise(count, seed) {
  let state = seed >>> 0;
  function random() {
    state = (state + 0x6D2B79F5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), state | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  const result = new Float32Array(count);
  for (let i = 0; i < count; i += 2) {
    const radius = Math.sqrt(-2 * Math.log(1 - random()));
    const angle = 2 * Math.PI * random();
    result[i] = radius * Math.cos(angle);
    if (i + 1 < count) result[i + 1] = radius * Math.sin(angle);
  }
  return result;
}

export function schedulerStep(coefficients, initial, predictions) {
  const result = new Float32Array(initial.length);
  const sources = [initial, ...predictions];
  for (let j = 0; j < sources.length; j++) {
    const weight = coefficients[j];
    if (!weight) continue;
    for (let i = 0; i < result.length; i++) result[i] += weight * sources[j][i];
  }
  return result;
}

export function packConditioning(unconditional, conditional) {
  if (unconditional.length !== conditional.length) throw new Error('Conditioning shapes do not match.');
  const result = new Float32Array(unconditional.length + conditional.length);
  result.set(unconditional);
  result.set(conditional, unconditional.length);
  return result;
}

export function imagePixels(channels, width, height) {
  const size = width * height;
  if (channels.length !== size * 3) throw new Error('Unexpected image shape.');
  const rgba = new Uint8ClampedArray(size * 4);
  for (let i = 0; i < size; i++) {
    for (let c = 0; c < 3; c++) {
      const value = channels[c * size + i];
      if (!Number.isFinite(value)) throw new Error('The model produced non-finite image pixels.');
      rgba[i * 4 + c] = Math.round(Math.max(0, Math.min(1, value)) * 255);
    }
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}

export function applyMask(rgba, values) {
  if (rgba.length !== values.length * 4) throw new Error('Image and mask dimensions do not match.');
  let min = Infinity, max = -Infinity;
  for (const value of values) {
    if (!Number.isFinite(value)) throw new Error('The background model produced a non-finite mask.');
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  if (max - min < 1e-6) throw new Error('The background model returned an empty or uniform mask. Try another word or seed.');
  for (let i = 0; i < values.length; i++) {
    const value = Math.round(Math.max(0, Math.min(1, values[i])) * 255);
    rgba[i * 4 + 3] = value;
  }
  return rgba;
}

export function backgroundInput(rgba) {
  const size = 512 * 512;
  if (rgba.length !== size * 4) throw new Error('Background removal needs a 512 × 512 image.');
  const pixels = new Float32Array(size * 3);
  const mean = [.485, .456, .406], std = [.229, .224, .225];
  for (let c = 0; c < 3; c++) for (let i = 0; i < size; i++) pixels[c * size + i] = (rgba[i * 4 + c] / 255 - mean[c]) / std[c];
  return pixels;
}

export function sigmoidMask(logits) {
  if (!logits.every(Number.isFinite)) throw new Error('The background model produced a non-finite mask.');
  return Float32Array.from(logits, value => 1 / (1 + Math.exp(-value)));
}
