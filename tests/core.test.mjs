import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { Tokenizer } from '@huggingface/tokenizers';
import { validateWord, wordKey, makePrompt, normalNoise, schedulerStep, packConditioning, imagePixels, applyMask, backgroundInput, sigmoidMask } from '../public/core.mjs';

test('one-word boundary accepts nouns and rejects prompts or markup', () => {
  assert.equal(validateWord(' slipper '), 'slipper');
  assert.equal(validateWord('橙子'), '橙子');
  assert.equal(validateWord('cafe\u0301'), 'café');
  for (const value of ['', 'two words', '<image>', 'ice-cream', "cat's", 'cat1', 'cat!', '🐈', 'x'.repeat(33)]) assert.throws(() => validateWord(value));
});
test('turning off one-word mode allows short descriptions and symbols while retaining input boundaries', () => {
  for (const value of ['red shoe', 'ice-cream!', '2 cats 🐈', '<image>']) assert.equal(validateWord(value, { singleWord: false }), value);
  for (const value of ['', 'x'.repeat(33), 'a\nb', 'a\u0000b', '<|im_end|>']) assert.throws(() => validateWord(value, { singleWord: false }));
});
test('repeat checks ignore case, Unicode composition and spacing, and can be disabled or reset', () => {
  const usedWords = new Set([wordKey('Slipper'), wordKey('café'), wordKey('red shoe!')]);
  for (const value of [' slipper ', 'SLIPPER', 'cafe\u0301']) assert.throws(() => validateWord(value, { usedWords }), /already been used/);
  assert.throws(() => validateWord('RED   SHOE!', { singleWord: false, usedWords }), /already been used/);
  assert.equal(validateWord('Slipper'), 'Slipper');
  assert.equal(validateWord('orange', { usedWords }), 'orange');
  assert.equal(validateWord('orange', { usedWords }), 'orange'); // Checking an attempt does not consume it.
  usedWords.clear();
  assert.equal(validateWord('slipper', { usedWords }), 'slipper');
});
test('prompt matches original Python chat separators', () => {
  assert.ok(makePrompt('slipper').includes('A single slipper, 2D side view'));
  assert.ok(makePrompt('slipper').endsWith('<|im_end|>\n<|im_start|>assistant\n'));
});
test('seeded noise is reproducible and approximately standard normal', () => {
  const noise = normalNoise(100000, 12345);
  assert.deepEqual(normalNoise(5, 0), normalNoise(5, 0));
  const mean = noise.reduce((a, b) => a + b, 0) / noise.length;
  const variance = noise.reduce((a, b) => a + (b - mean) ** 2, 0) / noise.length;
  assert.ok(Math.abs(mean) < .02);
  assert.ok(Math.abs(variance - 1) < .03);
});
test('schedule uses the initial noise and raw prediction history', () => {
  assert.deepEqual([...schedulerStep([.5, -.25, .75], new Float32Array([2, 4]),
    [new Float32Array([4, 8]), new Float32Array([8, 12])])], [6, 9]);
});

test('batched guidance receives unconditional conditioning first', () => {
  assert.deepEqual([...packConditioning([1, 2], [3, 6])], [1, 2, 3, 6]);
  assert.throws(() => packConditioning([1], [2, 3]));
});
test('planar RGB and normalized masks become finite RGBA pixels', () => {
  assert.deepEqual([...imagePixels([1, 0, .5], 1, 1)], [255, 0, 128, 255]);
  assert.deepEqual([...applyMask(new Uint8ClampedArray([12, 34, 56, 255, 78, 90, 123, 255]), [0, 1])], [12, 34, 56, 0, 78, 90, 123, 255]);
  assert.throws(() => imagePixels([NaN, 0, 0], 1, 1));
  assert.throws(() => applyMask(new Uint8ClampedArray(8), [1, 1]));
  assert.throws(() => applyMask(new Uint8ClampedArray(8), [0, NaN]));
  assert.throws(() => applyMask(new Uint8ClampedArray(4), [0, 1]));
});
test('prepared browser tokenizer matches the Python reference', { skip: !existsSync('public/models/manifest.json') }, () => {
  const read = path => JSON.parse(readFileSync(path, 'utf8'));
  const manifest = read('public/models/manifest.json');
  const tokenizer = new Tokenizer(read('public/models/tokenizer/tokenizer.json'), read('public/models/tokenizer/tokenizer_config.json'));
  assert.deepEqual(tokenizer.encode(manifest.tokenizer_check.text).ids, manifest.tokenizer_check.ids);
});
test('BiRefNet normalization and logit conversion match its model contract', () => {
  const input = backgroundInput(new Uint8ClampedArray(512 * 512 * 4));
  assert.ok(Math.abs(input[0] - (0 - .485) / .229) < 1e-6);
  assert.ok(Math.abs(input[512 * 512] - (0 - .456) / .224) < 1e-6);
  assert.deepEqual([...sigmoidMask([0])], [.5]);
  assert.ok(Math.abs(sigmoidMask([Math.log(9)])[0] - .9) < 1e-6);
  assert.throws(() => sigmoidMask([NaN]));
  assert.throws(() => backgroundInput(new Uint8Array(4)));
});
