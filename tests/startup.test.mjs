import test from 'node:test';
import assert from 'node:assert/strict';
import { startupFeedback } from '../public/startup.mjs';

function fixture(t) {
  let now = 0, reloads = 0;
  t.mock.method(performance, 'now', () => now);
  t.mock.timers.enable({ apis: ['setInterval'] });
  const nodes = new Map();
  const page = new EventTarget();
  page.defaultView = { location: { reload: () => { reloads++; } } };
  page.getElementById = id => {
    if (!nodes.has(id)) {
      const node = new EventTarget(), attributes = new Map(), classes = new Set();
      Object.assign(node, { textContent: '', hidden: false, disabled: false,
        setAttribute: (name, value) => attributes.set(name, value),
        removeAttribute: name => attributes.delete(name),
        getAttribute: name => attributes.get(name) ?? null,
        classList: { add: name => classes.add(name), contains: name => classes.has(name) },
      });
      nodes.set(id, node);
    }
    return nodes.get(id);
  };
  const feedback = startupFeedback(page);
  return { $, feedback, reloads: () => reloads,
    advance(ms) { now += ms; t.mock.timers.tick(ms); },
    stage(detail) { page.dispatchEvent(new CustomEvent('startup-stage', { detail })); },
  };
  function $(id) { return page.getElementById(id); }
}

test('startup keeps elapsed time and real stage updates during slow GPU and model-information checks', t => {
  const ui = fixture(t);
  ui.advance(500);
  assert.equal(ui.$('startup-time').textContent, '0.5 s elapsed');
  for (const [step, message] of [[1, 'Checking WebGPU support…'], [2, 'Reading model information…']]) {
    ui.stage({ step, message });
    const before = ui.$('startup-time').textContent;
    ui.advance(1000);
    assert.notEqual(ui.$('startup-time').textContent, before);
    assert.equal(ui.$('status').textContent, message);
    assert.equal(ui.$('startup-stage').textContent, message);
    assert.equal(ui.$('startup-step').textContent, `Check ${step} of 4`);
    assert.equal(ui.$('progress').getAttribute('value'), null);
    assert.equal(ui.$('progress').getAttribute('aria-label'), 'Startup progress');
  }
  ui.advance(10000);
  assert.equal(ui.$('startup-retry').hidden, false);
  assert.match(ui.$('startup-note').textContent, /wait or retry/);
  ui.stage({ step: 3, message: 'Loading the AI engine…' });
  assert.equal(ui.$('startup-retry').hidden, true);
});

test('finishing startup stops its timer and leaves generation progress ready', t => {
  const ui = fixture(t);
  ui.stage({ complete: true });
  assert.equal(ui.$('startup').hidden, true);
  assert.equal(ui.$('startup').getAttribute('aria-busy'), 'false');
  assert.equal(ui.$('progress').value, 0);
  assert.equal(ui.$('progress').getAttribute('aria-label'), 'Generation progress');
  ui.advance(20000);
  assert.equal(ui.$('elapsed').textContent, '');
});

test('startup failures stop the animation and timer, show the reason and offer reload', t => {
  const ui = fixture(t);
  ui.advance(1200);
  ui.stage({ error: true, message: 'GPU unavailable' });
  assert.equal(ui.$('startup-stage').textContent, 'Startup failed');
  assert.equal(ui.$('startup-note').textContent, 'GPU unavailable');
  assert.equal(ui.$('startup-spinner').hidden, true);
  assert.equal(ui.$('startup-progress').hidden, true);
  assert.equal(ui.$('startup-retry').hidden, false);
  assert.equal(ui.$('prepare').disabled, true);
  assert.equal(ui.$('status').classList.contains('error'), true);
  const elapsed = ui.$('startup-time').textContent;
  ui.advance(20000);
  assert.equal(ui.$('startup-time').textContent, elapsed);
  ui.$('startup-retry').dispatchEvent(new Event('click'));
  assert.equal(ui.reloads(), 1);
  ui.stage({ complete: true });
  assert.equal(ui.$('startup').hidden, false);
});

test('a game-module loading failure uses the same visible failure state', t => {
  const ui = fixture(t);
  ui.feedback.fail(new Error('Game code could not load'));
  assert.equal(ui.$('startup-stage').textContent, 'Startup failed');
  assert.equal(ui.$('status').textContent, 'Game code could not load');
  assert.equal(ui.$('startup-retry').hidden, false);
});
