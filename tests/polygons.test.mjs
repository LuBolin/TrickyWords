import { test } from 'node:test';
import assert from 'node:assert/strict';
import { collisionPolygons, polygonArea } from '../public/polygons.mjs';

function trace(rows) {
  const pixels = new Uint8ClampedArray(rows.length * rows[0].length * 4);
  rows.forEach((row, y) => [...row].forEach((c, x) => { pixels[(y * row.length + x) * 4 + 3] = c === '#' ? 255 : 0; }));
  return collisionPolygons(pixels, rows[0].length, rows.length);
}
const cross = (a, b, c) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
function assertFixtures(shape) {
  assert.ok(shape.convexParts.length > 0);
  for (const part of shape.convexParts) {
    assert.ok(part.length >= 3 && part.length <= 8);
    assert.ok(part.every((p, i) => cross(p, part[(i + 1) % part.length], part[(i + 2) % part.length]) > 0));
  }
  const area = shape.convexParts.reduce((sum, part) => sum + polygonArea(part), 0);
  assert.ok(Math.abs(area - shape.areaPx) < 1e-6);
}

test('a solid rectangle on the image border keeps its area and center', () => {
  const shape = trace(['####', '####', '####']);
  assert.equal(shape.areaPx, 12);
  assert.equal(shape.convexParts.length, 1);
  assert.deepEqual(shape.centroid, [2, 1.5]);
  assertFixtures(shape);
});
test('a concave U keeps its opening and splits into convex fixtures', () => {
  const rows = ['##....##', '##....##', '##....##', '##....##', '########', '########'];
  const shape = trace(rows);
  assert.equal(shape.areaPx, 32);
  assert.ok(shape.convexParts.length > 1);
  assert.ok(shape.outline.some(p => p[0] === 2 && p[1] === 4));
  assert.equal(shape.filledHoles, 0);
  assertFixtures(shape);
  rows.forEach((row, y) => [...row].forEach((pixel, x) => {
    const point = [x + .37, y + .61];
    const coverage = shape.convexParts.filter(part => part.every((a, i) => cross(a, part[(i + 1) % part.length], point) >= 0)).length;
    assert.equal(coverage, pixel === '#' ? 1 : 0, 'Fixtures must cover the body once and leave the opening empty.');
  }));
});
test('detached pixels and diagonal-only contacts do not join the main body', () => {
  const shape = trace(['###.....', '###.....', '...#....', '......##']);
  assert.equal(shape.solidPixels, 6);
  assert.equal(shape.discardedPixels, 3);
  assert.equal(shape.areaPx, 6);
  assertFixtures(shape);
});
test('enclosed holes are counted and deliberately filled for collisions', () => {
  const shape = trace(['########', '########', '##....##', '##....##', '########', '########']);
  assert.equal(shape.filledHoles, 1);
  assert.equal(shape.solidPixels, 40);
  assert.equal(shape.areaPx, 48);
  assertFixtures(shape);
});
test('alpha threshold, empty images and bad dimensions have explicit outcomes', () => {
  assert.equal(collisionPolygons([0, 0, 0, 128], 1, 1).areaPx, 1);
  assert.throws(() => collisionPolygons([0, 0, 0, 127], 1, 1), /No solid pixels/);
  assert.throws(() => collisionPolygons([], 4, 4), /RGBA/);
  assert.throws(() => collisionPolygons([], -1, 1), /RGBA/);
});
test('thin protrusions remain present and all pieces cover the outline', () => {
  const shape = trace(['...#...', '...#...', '#######', '#######', '...#...']);
  assert.equal(shape.areaPx, 17);
  assertFixtures(shape);
});
