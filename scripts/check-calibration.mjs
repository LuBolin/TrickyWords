import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

assert.ok(process.argv[2], 'Usage: node scripts/check-calibration.mjs <browser-report.json>');
const report = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const measurements = report.runtimeCalibration?.measurements;
assert.equal(report.outcome, 'success');
assert.ok(measurements?.some(item => item.runtime === 'wasm'), 'The prepared CPU fallback must be tested.');
for (const item of measurements) {
  assert.equal(item.error, undefined, `${item.runtime} must complete repeated inference.`);
  assert.equal(item.runsMs.length, 2);
  assert.ok(item.runsMs.every(ms => Number.isFinite(ms) && ms > 0));
  assert.ok(item.agreement.iou >= .995 && item.agreement.meanAbsoluteError <= .005, 'Masks must agree.');
}
console.log(`Calibration passed for ${measurements.map(item => item.runtime).join(' and ')}.`);
