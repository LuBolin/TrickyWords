import test from 'node:test';
import assert from 'node:assert/strict';
import { generationProgress } from '../public/generation-progress.mjs';

function fixture() {
  const names = [['text', 'Read word'], ['dit', 'Draw'], ['vae', 'Make image'], ['background', 'Cut out'], ['polygon', 'Add physics']];
  const children = names.map(([key, label]) => {
    const attributes = new Map(), ball = { textContent: '' };
    return { dataset: { stage: key }, ball,
      querySelector: selector => selector === '.stage-ball' ? ball : { textContent: label },
      setAttribute: (name, value) => attributes.set(name, value),
      removeAttribute: name => attributes.delete(name),
      getAttribute: name => attributes.get(name),
    };
  });
  const list = { children, hidden: true }, count = { hidden: true, textContent: '' };
  const progress = generationProgress({ getElementById: id => id === 'loading-stages' ? list : count });
  return { progress, list, count, children };
}

test('generation shows every stage, with only the first one active initially', () => {
  const ui = fixture();
  ui.progress.reset(true);
  assert.equal(ui.list.hidden, false);
  assert.equal(ui.count.hidden, false);
  assert.deepEqual(ui.children.map(item => item.dataset.state), ['active', 'pending', 'pending', 'pending', 'pending']);
  assert.equal(ui.children[0].getAttribute('aria-current'), 'step');
  assert.equal(ui.count.textContent, 'Stage 1 of 5 · 0 done · 4 after this');
});

test('later stages mark only earlier stages complete and report what remains', () => {
  const ui = fixture();
  ui.progress.reset(true);
  for (const [index, item] of ui.children.entries()) {
    ui.progress.stage(item.dataset.stage);
    assert.deepEqual(ui.children.map(child => child.dataset.state), ui.children.map((_, i) => i < index ? 'done' : i === index ? 'active' : 'pending'));
    assert.equal(ui.children.filter(child => child.getAttribute('aria-current') === 'step').length, 1);
  }
  ui.progress.stage('background');
  assert.equal(ui.children[2].ball.textContent, '✓');
  assert.equal(ui.children[3].ball.textContent, '4');
  assert.equal(ui.children[3].getAttribute('aria-label'), 'Cut out: current stage');
  assert.equal(ui.children[4].getAttribute('aria-label'), 'Add physics: not started');
  assert.equal(ui.count.textContent, 'Stage 4 of 5 · 3 done · 1 after this');
});

test('completion and a new run cannot leave stale completed stages', () => {
  const ui = fixture();
  ui.progress.reset(true);
  ui.progress.complete();
  assert.ok(ui.children.every(item => item.dataset.state === 'done' && item.ball.textContent === '✓' && !item.getAttribute('aria-current')));
  assert.equal(ui.count.textContent, 'All 5 stages complete');
  ui.progress.reset(false);
  assert.equal(ui.list.hidden, true);
  assert.equal(ui.count.hidden, true);
  ui.progress.reset(true);
  assert.equal(ui.children[0].dataset.state, 'active');
  assert.equal(ui.children[4].ball.textContent, '5');
  assert.equal(ui.children[4].dataset.state, 'pending');
});

test('unknown stages fail without changing progress', () => {
  const ui = fixture();
  ui.progress.reset(true);
  const before = ui.count.textContent;
  assert.throws(() => ui.progress.stage('missing'), /Unknown generation stage/);
  assert.equal(ui.count.textContent, before);
});
