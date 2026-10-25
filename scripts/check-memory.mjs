import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const report = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const physical = report.sampledPrivatePhysicalIncreaseBytes ?? report.readyPrivatePhysicalIncreaseBytes;
const commit = report.sampledPeakIncrease.privateCommitBytes;
const gpu = report.sampledPeakIncrease.gpuCombinedCounterBytes;
assert.ok([physical, commit, gpu].every(Number.isFinite), 'Measure private RAM, commit and GPU allocations against the unloaded browser.');
console.log(`Extra private RAM: ${(physical / 1e9).toFixed(2)} GB; peak extra commit: ${(commit / 1e9).toFixed(2)} GB; peak GPU allocations: ${(gpu / 1e9).toFixed(2)} GB`);
assert.ok(physical < 3e9, 'The AI models must use under 3 GB extra private physical RAM.');
// Commit includes pagefile/GPU backing, so it is not a RAM limit. This conservative planning sum
// can double-count shared resources and leaves about 3 GB for Windows/browser on an 8 GB laptop.
const planning = report.sampledPhysicalPlusGpuIncreaseBytes ?? physical + gpu;
assert.ok(planning < 5e9, 'Physical RAM plus GPU allocations must fit the conservative 5 GB planning budget.');
