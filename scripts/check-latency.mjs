import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const path = process.argv[2];
assert.ok(path, 'Usage: node scripts/check-latency.mjs <browser-report.json>');
const report = JSON.parse(readFileSync(path, 'utf8'));
assert.equal(report.outcome, 'success', 'Generation must finish successfully.');
if (Object.hasOwn(report, 'collision')) {
  assert.equal(report.collisionError, null, 'Collision tracing must finish successfully.');
  assert.ok(report.collision?.outline?.length >= 3 && report.collision?.convexParts?.length,
    'The object must include a usable collision shape.');
}
assert.ok(Number.isFinite(report.timingsMs.total), 'The report must contain a measured total.');
console.log(`${report.word}: ${(report.timingsMs.total / 1000).toFixed(2)} s, load ${(report.timingsMs.load / 1000).toFixed(2)} s`);
assert.ok(report.timingsMs.total < 6000, 'The complete object must be ready in under six seconds.');
