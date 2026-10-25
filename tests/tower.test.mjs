import { test } from 'node:test';
import assert from 'node:assert/strict';
import Matter from 'matter-js';
import { readFileSync } from 'node:fs';
import { objectBody, Tower, field } from '../public/tower-physics.mjs';
import { collisionPolygons } from '../public/polygons.mjs';

const rectangle = { convexParts: [[[0, 0], [130, 0], [130, 70], [0, 70]]] };
const asset = { geometry: rectangle };
function settle(tower, limit = 1800) {
  for (let i = 0; i < limit && tower.phase !== 'idle' && tower.phase !== 'over'; i++) tower.step();
  return tower.phase;
}

test('stacking counts only settled objects and supports five real rigid bodies', () => {
  const tower = new Tower(Matter);
  for (let i = 1; i <= 5; i++) {
    assert.equal(tower.spawn(asset), true);
    assert.equal(tower.stacked, i - 1);
    assert.equal(tower.spawn(asset), false);
    assert.equal(tower.drop(), true);
    assert.equal(tower.drop(), false);
    assert.equal(settle(tower), 'idle');
    assert.equal(tower.stacked, i);
    assert.ok(Math.abs(tower.height - i * .7) < .08);
  }
  for (let i = 0; i < 2400; i++) tower.step();
  assert.equal(tower.phase, 'idle');
  assert.ok(Math.abs(tower.bodies[4].position.y - (field.floor - 315)) < 4);
});

test('a missed platform ends the round and reset removes all previous bodies', () => {
  const tower = new Tower(Matter);
  tower.spawn(asset);
  tower.move(-1, 300);
  tower.drop();
  assert.equal(settle(tower), 'over');
  assert.equal(tower.stacked, 0);
  assert.match(tower.reason, /fell/);
  tower.reset();
  assert.equal(tower.bodies.length, 0);
  assert.equal(Matter.Composite.allBodies(tower.engine.world).length, 1);
  assert.equal(tower.phase, 'idle');
});

test('rotation and horizontal control work while aiming and falling, within the field', () => {
  const tower = new Tower(Matter);
  tower.spawn(asset);
  const body = tower.active.body;
  tower.rotate(1);
  assert.ok(Math.abs(body.angle - Math.PI / 12) < 1e-9);
  tower.move(-1, 1000);
  assert.ok(body.bounds.min.x >= 28 - 1e-6);
  tower.rotate(1);
  assert.ok(body.bounds.min.x >= 28 - 1e-6);
  tower.drop();
  tower.move(1);
  assert.ok(body.velocity.x > 0);
  tower.rotate(-1);
  assert.ok(body.angularVelocity < 0);
  assert.throws(() => tower.step(100), /fixed step/);
});

test('mouse targets clamp an aimed body and steer a falling body without teleporting it', () => {
  const tower = new Tower(Matter);
  tower.spawn(asset);
  const body = tower.active.body;
  assert.ok(body.position.y < field.height * .2);
  assert.ok(field.floor - body.position.y > field.height * .6);
  tower.moveTo(160);
  assert.equal(body.position.x, 160);
  tower.moveTo(-1000);
  assert.ok(body.bounds.min.x >= 28 - 1e-6);
  tower.moveTo(1000);
  assert.ok(body.bounds.max.x <= field.width - 28 + 1e-6);
  tower.moveTo(field.width / 2);
  tower.drop();
  const before = { ...body.position };
  Matter.Body.setVelocity(body, { x: 0, y: 4 });
  tower.moveTo(400);
  assert.deepEqual(body.position, before);
  assert.equal(body.velocity.y, 4);
  assert.ok(body.velocity.x > 0 && body.velocity.x <= 3.5);
  tower.moveTo(body.position.x);
  assert.equal(body.velocity.x, 0);
  tower.moveTo(NaN);
  assert.equal(body.velocity.x, 0);
  tower.phase = 'settling';
  tower.moveTo(100);
  assert.deepEqual(body.position, before);
  assert.equal(body.velocity.x, 0);
});

test('compound fixtures preserve a concave opening and the sprite origin', () => {
  const rows = ['##....##', '##....##', '##....##', '########', '########'];
  const pixels = new Uint8ClampedArray(8 * 5 * 4);
  rows.forEach((row, y) => [...row].forEach((c, x) => { pixels[(y * 8 + x) * 4 + 3] = c === '#' ? 255 : 0; }));
  const geometry = collisionPolygons(pixels, 8, 5);
  const body = objectBody(Matter, geometry, 320, 300), { scale, offset } = body.plugin.sprite;
  const point = (x, y) => ({ x: body.position.x + offset.x + x * scale, y: body.position.y + offset.y + y * scale });
  assert.equal(body.parts.length, geometry.convexParts.length + 1);
  assert.equal(Matter.Query.point([body], point(4, 1)).length, 0);
  assert.equal(Matter.Query.point([body], point(1, 1)).length, 1);
  assert.ok(Math.abs(body.area - geometry.areaPx * scale * scale) < 1e-6);
  assert.throws(() => objectBody(Matter, { convexParts: [[[0, 0], [1, NaN], [0, 1]]] }, 0, 0), /invalid/);
  assert.throws(() => objectBody(Matter, { convexParts: [[[0, 0], [4, 0], [1, 1], [4, 4], [0, 4]]] }, 0, 0), /convex/);
});

test('the real saved slipper and rotated teapot form a stable stack', () => {
  const shapes = JSON.parse(readFileSync(new URL('./fixtures/saved-shapes.json', import.meta.url), 'utf8'));
  const tower = new Tower(Matter);
  tower.spawn({ geometry: shapes.slipper }); tower.drop();
  assert.equal(settle(tower), 'idle');
  tower.spawn({ geometry: shapes.teapot }); tower.move(1, 24); tower.rotate(-1); tower.drop();
  assert.equal(settle(tower), 'idle');
  assert.equal(tower.stacked, 2);
  for (let i = 0; i < 2400; i++) tower.step();
  assert.equal(tower.phase, 'idle');
  assert.ok(tower.height > 1.8 && tower.height < 2);
  // A previously settled object falling away must end the round too.
  Matter.Body.translate(tower.bodies[0], { x: -400, y: 0 });
  Matter.Sleeping.set(tower.bodies[0], false);
  for (let i = 0; i < 1200 && tower.phase !== 'over'; i++) tower.step();
  assert.equal(tower.phase, 'over');
});
