import { polygonArea } from './polygons.mjs';

export const field = { width: 480, height: 880, floor: 800, platformWidth: 250 };

export function objectBody(Matter, geometry, x, y) {
  const { Body, Vertices } = Matter;
  const parts = geometry?.convexParts;
  if (!Array.isArray(parts) || !parts.length || parts.length > 94 ||
      parts.some(part => !Array.isArray(part) || part.length < 3 || part.length > 8 ||
        part.some(p => !Array.isArray(p) || p.length !== 2 || !p.every(Number.isFinite)) || !(polygonArea(part) > 0))) {
    throw new Error('This object has invalid collision polygons. Try another word or seed.');
  }
  const points = parts.flat();
  const width = Math.max(...points.map(p => p[0])) - Math.min(...points.map(p => p[0]));
  const height = Math.max(...points.map(p => p[1])) - Math.min(...points.map(p => p[1]));
  if (!(width > 0 && height > 0 && Number.isFinite(width) && Number.isFinite(height))) throw new Error('This object has no finite collision area.');
  const scale = Math.min(130 / width, 105 / height);
  const fixtures = parts.map(part => {
    const vertices = part.map(([px, py]) => ({ x: px * scale, y: py * scale }));
    if (!Vertices.isConvex(vertices)) throw new Error('The object needs convex collision pieces.');
    // Each fixture keeps its own centroid, so concave openings survive assembly.
    return Body.create({ position: Vertices.centre(vertices), vertices, density: .002 });
  });
  const body = fixtures.length === 1 ? fixtures[0] : Body.create({ parts: fixtures });
  const offset = { x: -body.position.x, y: -body.position.y };
  Object.assign(body, { friction: .85, frictionStatic: 1.5, frictionAir: .012, restitution: .02, slop: .03, sleepThreshold: 75 });
  Body.setPosition(body, { x, y });
  body.plugin.sprite = { scale, offset };
  return body;
}

export class Tower {
  constructor(Matter) {
    this.Matter = Matter;
    this.engine = Matter.Engine.create({ enableSleeping: true, positionIterations: 10, velocityIterations: 8 });
    this.reset();
  }
  reset() {
    const { Composite, Engine, Bodies } = this.Matter;
    Composite.clear(this.engine.world, false);
    Engine.clear(this.engine);
    this.platform = Bodies.rectangle(field.width / 2, field.floor + 12, field.platformWidth, 24,
      { isStatic: true, friction: .9, label: 'platform' });
    Composite.add(this.engine.world, this.platform);
    this.bodies = [];
    this.active = null;
    this.phase = 'idle';
    this.stacked = 0;
    this.height = 0;
    this.peak = 0;
    this.stableMs = 0;
    this.reason = '';
  }
  top() {
    return Math.min(field.floor, ...this.bodies.filter(body => body !== this.active?.body).map(body => body.bounds.min.y));
  }
  spawn(asset) {
    if (this.phase !== 'idle') return false;
    const body = objectBody(this.Matter, asset.geometry, field.width / 2, Math.min(120, this.top() - 180));
    body.plugin.asset = asset;
    this.active = { body, asset };
    this.phase = 'aiming';
    return true;
  }
  move(direction, amount = 12) {
    if (!this.active || !['aiming', 'falling'].includes(this.phase)) return;
    const { Body } = this.Matter, body = this.active.body;
    if (this.phase === 'falling') Body.setVelocity(body, { x: direction * 3.5, y: body.velocity.y });
    else this.moveTo(body.position.x + direction * amount);
  }
  moveTo(x) {
    if (!Number.isFinite(x) || !this.active || !['aiming', 'falling'].includes(this.phase)) return;
    const { Body } = this.Matter, body = this.active.body;
    const shift = Math.max(28 - body.bounds.min.x, Math.min(x - body.position.x, field.width - 28 - body.bounds.max.x));
    if (this.phase === 'falling') Body.setVelocity(body, { x: Math.max(-3.5, Math.min(3.5, shift * .12)), y: body.velocity.y });
    else Body.translate(body, { x: shift, y: 0 });
  }
  rotate(direction) {
    if (!this.active || !['aiming', 'falling'].includes(this.phase)) return;
    const { Body } = this.Matter, body = this.active.body;
    if (this.phase === 'falling') Body.setAngularVelocity(body, direction * .045);
    else { Body.rotate(body, direction * Math.PI / 12); this.move(0, 0); }
  }
  drop() {
    if (this.phase !== 'aiming') return false;
    this.Matter.Composite.add(this.engine.world, this.active.body);
    this.bodies.push(this.active.body);
    this.phase = 'falling';
    this.stableMs = 0;
    return true;
  }
  step(ms = 1000 / 120) {
    if (this.phase === 'over') return;
    if (!Number.isFinite(ms) || ms <= 0 || ms > 1000 / 60 + .001) throw new Error('Physics needs a fixed step of at most 1/60 second.');
    this.Matter.Engine.update(this.engine, ms);
    if (this.bodies.some(body => body.bounds.min.y > field.floor + 125 || body.bounds.max.x < -100 || body.bounds.min.x > field.width + 100)) {
      this.phase = 'over'; this.reason = 'An object fell off the tower.'; return;
    }
    if (!this.active || this.phase === 'aiming') return;
    const body = this.active.body;
    const touching = this.engine.pairs.list.some(pair => pair.isActive &&
      (pair.bodyA.parent === body || pair.bodyB.parent === body));
    if (touching) this.phase = 'settling';
    if (this.phase !== 'settling') return;
    const quiet = touching && this.bodies.every(piece => piece.isSleeping || (piece.speed < .3 && piece.angularSpeed < .015));
    this.stableMs = quiet ? this.stableMs + ms : 0;
    if (this.stableMs < 650) return;
    this.stacked++;
    this.active = null;
    this.phase = 'idle';
    this.height = Math.max(0, (field.floor - this.top()) / 100);
    this.peak = Math.max(this.peak, this.height);
  }
}
