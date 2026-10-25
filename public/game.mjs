import { Tower, field } from './tower-physics.mjs';

const $ = id => document.getElementById(id);
const tower = new Tower(window.Matter);
const canvas = $('tower'), context = canvas.getContext('2d');
const collisionToggle = $('show-collisions');
const controls = ['move-left', 'move-right', 'rotate-left', 'rotate-right', 'drop'];
const keys = new Set();
let mouseX = null, wheelDelta = 0;
let running = false, manualPause = false, inferenceBusy = false, modelsReady = false;
let queued = null;
let camera = 0, previous = performance.now(), accumulator = 0, lastState = '';
let best = 0;
try { best = Math.max(0, Number(localStorage.getItem('tricky-words-best-v1')) || 0); }
catch { /* A private tab can still play without a saved record. */ }
if (!Number.isFinite(best)) best = 0;
$('tower-best').textContent = `${best.toFixed(2)} m`;

const paused = () => manualPause || inferenceBusy;
const playing = () => running && tower.phase !== 'over';
const resetWordHistory = () => document.dispatchEvent(new Event('word-history-reset'));
function clearInput() { keys.clear(); mouseX = null; wheelDelta = 0; }

function preview(asset) {
  $('next-name').textContent = asset.word;
  const ctx = $('next-preview').getContext('2d');
  ctx.clearRect(0, 0, 256, 256);
  ctx.drawImage(asset.image, 0, 0, 256, 256);
  $('next-preview').hidden = false;
}
function nextObject() {
  const asset = queued;
  queued = null;
  if (asset) {
    tower.spawn(asset); preview(asset);
    canvas.focus({ preventScroll: true });
    canvas.closest('.stage').scrollIntoView({ block: 'start' });
  }
}
function start() {
  if (inferenceBusy) return;
  tower.reset();
  queued = null;
  resetWordHistory();
  running = true; manualPause = false; clearInput();
  camera = 0; accumulator = 0;
  $('next-name').textContent = 'Your first word';
  $('next-preview').getContext('2d').clearRect(0, 0, 256, 256);
  $('next-preview').hidden = true;
  refresh(true);
  $('word').focus();
}
function updateBest() {
  if (tower.peak <= best) return;
  best = tower.peak;
  try { localStorage.setItem('tricky-words-best-v1', String(best)); }
  catch { /* The score remains visible when browser storage is unavailable. */ }
}
function refresh(force = false) {
  const state = [running, tower.phase, tower.stacked, paused(), modelsReady, !!queued].join(':');
  if (!force && state === lastState) return;
  lastState = state;
  updateBest();
  $('intro-overlay').hidden = running;
  const needsFolder = $('prepare').disabled && !modelsReady;
  $('start-game').disabled = inferenceBusy || (needsFolder && $('model-folder').disabled);
  $('start-game').textContent = inferenceBusy ? 'Working…' : modelsReady ? 'Choose your first word' : needsFolder ? 'Choose model folder' : 'Load AI models';
  $('intro-copy').textContent = inferenceBusy ? 'The AI is working.' :
    modelsReady ? 'Enter a word and make your first object. Lose one object and the round ends.' :
    needsFolder ? 'Choose your downloaded model folder, then load the AI to begin.' : 'Your model folder is ready. Load the AI, then generate your first word.';
  $('over-overlay').hidden = tower.phase !== 'over';
  $('pause-overlay').hidden = !playing() || !paused();
  $('pause-title').textContent = inferenceBusy ? 'Making your object.' : 'Paused.';
  $('pause-copy').textContent = inferenceBusy ? 'The tower is paused while the AI works.' : 'Your tower can wait.';
  $('resume-game').hidden = inferenceBusy;
  $('pause-game').textContent = manualPause ? 'Resume' : 'Pause';
  $('pause-game').disabled = !playing() || inferenceBusy;
  $('restart-game').disabled = !running || inferenceBusy;
  const controllable = playing() && !paused() && ['aiming', 'falling'].includes(tower.phase);
  for (const id of controls) $(id).disabled = !controllable || (id === 'drop' && tower.phase !== 'aiming');
  $('tower-height').textContent = `${tower.height.toFixed(2)} m`;
  $('tower-count').textContent = String(tower.stacked);
  $('tower-best').textContent = `${best.toFixed(2)} m`;
  $('turn-state').textContent = !running ? 'Ready' : paused() ? 'Paused' :
    ({ aiming: 'Aim', falling: 'Falling', settling: 'Balancing', idle: 'Your word?', over: 'Round over' })[tower.phase];
  $('mode-note').textContent = queued ? `${queued.word} is ready for the next turn.` :
    running ? 'Make a new object after each drop.' : 'Every piece starts with your word.';
  const messages = {
    aiming: 'Move and rotate your object. Drop it when the balance looks right.',
    falling: 'You can still steer and rotate until the object touches the tower.',
    settling: 'Let it find its balance…', idle: 'Name your next object, then click Make object.',
    over: tower.reason,
  };
  $('game-status').textContent = inferenceBusy ? (modelsReady ? 'Generating an object. The tower is paused.' : 'Loading the AI models…') :
    !running ? (modelsReady ? 'Enter a word and make your first object.' : needsFolder ? 'Choose your model folder to begin.' : 'Load the AI models to begin.') :
    manualPause ? 'Paused. Press P or Resume to continue.' : messages[tower.phase];
  canvas.setAttribute('aria-label', `Tower game. ${tower.stacked} objects stacked, ${tower.height.toFixed(2)} meters high. ${$('turn-state').textContent}.`);
  if (tower.phase === 'over') {
    resetWordHistory();
    $('round-result').textContent = `${tower.reason} You stacked ${tower.stacked} ${tower.stacked === 1 ? 'object' : 'objects'} and reached ${tower.peak.toFixed(2)} m.`;
    clearInput();
    if (document.activeElement === canvas) $('play-again').focus({ preventScroll: true });
  }
}
function drop() {
  if (!playing() || paused()) return;
  tower.drop(); refresh(); canvas.focus({ preventScroll: true });
}
function togglePause() {
  if (!playing() || inferenceBusy) return;
  manualPause = !manualPause; clearInput(); refresh();
  if (!manualPause) canvas.focus({ preventScroll: true });
}
$('start-game').addEventListener('click', () => {
  if (modelsReady) $('word').focus();
  else if ($('prepare').disabled) $('model-folder').click();
  else $('prepare').click();
});
$('play-again').addEventListener('click', start);
$('restart-game').addEventListener('click', () => {
  if (!inferenceBusy) start();
});
$('pause-game').addEventListener('click', togglePause);
$('resume-game').addEventListener('click', togglePause);
$('drop').addEventListener('click', drop);
canvas.addEventListener('pointermove', event => {
  if (event.pointerType !== 'mouse' || !playing() || paused() || !['aiming', 'falling'].includes(tower.phase)) return;
  const bounds = canvas.getBoundingClientRect();
  keys.clear();
  mouseX = (event.clientX - bounds.left) * field.width / bounds.width;
});
canvas.addEventListener('pointerleave', () => { mouseX = null; wheelDelta = 0; });
canvas.addEventListener('click', event => { if (event.button === 0) drop(); });
canvas.addEventListener('wheel', event => {
  if (!playing() || paused() || !['aiming', 'falling'].includes(tower.phase) || event.ctrlKey) return;
  event.preventDefault();
  wheelDelta += event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? canvas.clientHeight : 1);
  const steps = Math.trunc(wheelDelta / 80);
  wheelDelta -= steps * 80;
  for (let i = 0; i < Math.min(4, Math.abs(steps)); i++) tower.rotate(Math.sign(steps));
}, { passive: false });
for (const [id, direction] of [['move-left', -1], ['move-right', 1]]) {
  $(id).addEventListener('pointerdown', event => {
    if ($(id).disabled) return;
    mouseX = null;
    keys.add(id); $(id).setPointerCapture(event.pointerId);
  });
  for (const event of ['pointerup', 'pointercancel', 'lostpointercapture']) $(id).addEventListener(event, () => keys.delete(id));
  $(id).addEventListener('click', () => {
    mouseX = null;
    if (playing() && !paused()) tower.move(direction, 18);
    canvas.focus({ preventScroll: true });
  });
}
for (const [id, direction] of [['rotate-left', -1], ['rotate-right', 1]]) $(id).addEventListener('click', () => {
  if (playing() && !paused()) tower.rotate(direction);
  canvas.focus({ preventScroll: true });
});
window.addEventListener('keydown', event => {
  if (!playing() || event.target.closest('input, textarea, select, button, summary, a') || event.ctrlKey || event.metaKey || event.altKey) return;
  const key = event.key.toLowerCase();
  if (!['a', 'd', 's', 'arrowleft', 'arrowright', 'arrowdown', 'q', 'e', ' ', 'p', 'escape'].includes(key)) return;
  event.preventDefault();
  if (key === 'p' || key === 'escape') { if (!event.repeat) togglePause(); return; }
  if (paused()) return;
  if (['a', 'd', 'arrowleft', 'arrowright'].includes(key)) { mouseX = null; keys.add(key); }
  if (!event.repeat) {
    if (key === 'q' || key === 'e') tower.rotate(key === 'q' ? -1 : 1);
    if (key === 's' || key === ' ' || key === 'arrowdown') drop();
  }
});
window.addEventListener('keyup', event => keys.delete(event.key.toLowerCase()));
function backgroundPause() {
  clearInput();
  if (playing()) { manualPause = true; refresh(); }
}
window.addEventListener('blur', backgroundPause);
document.addEventListener('visibilitychange', () => { if (document.hidden) backgroundPause(); });

document.addEventListener('inference-state', event => {
  inferenceBusy = event.detail.busy;
  modelsReady = event.detail.ready;
  clearInput();
  if (!inferenceBusy && playing() && tower.phase === 'idle') nextObject();
  refresh(true);
});
document.addEventListener('generated-object', event => {
  const { image, geometry, word, seed } = event.detail;
  const copy = document.createElement('canvas'); copy.width = image.width; copy.height = image.height;
  copy.getContext('2d').drawImage(image, 0, 0);
  const asset = { image: copy, geometry, word, seed };
  queued = asset;
  if (!playing()) {
    // The inference-state event starts the turn after generation finishes.
    tower.reset(); running = true; manualPause = false; camera = 0;
    resetWordHistory();
  } else if (tower.phase === 'aiming') {
    tower.active = null; tower.phase = 'idle';
  }
  preview(asset); refresh(true);
});

function drawAsset(body, asset) {
  const { scale, offset } = body.plugin.sprite;
  context.save(); context.translate(body.position.x, body.position.y); context.rotate(body.angle);
  context.shadowColor = '#071e2380'; context.shadowBlur = 5; context.shadowOffsetY = 3;
  context.drawImage(asset.image, offset.x, offset.y, asset.image.width * scale, asset.image.height * scale);
  context.shadowBlur = 0; context.shadowOffsetY = 0;
  context.restore();
  if (!collisionToggle.checked) return;
  context.save();
  context.lineWidth = 1.5; context.strokeStyle = '#d2e5aa'; context.fillStyle = '#d2e5aa20';
  for (const part of body.parts.length > 1 ? body.parts.slice(1) : body.parts) {
    context.beginPath();
    part.vertices.forEach(({ x, y }, i) => {
      i ? context.lineTo(x, y) : context.moveTo(x, y);
    });
    context.closePath(); context.fill(); context.stroke();
  }
  context.restore();
}
function draw() {
  context.clearRect(0, 0, field.width, field.height);
  context.fillStyle = '#183b40'; context.fillRect(0, 0, field.width, field.height);
  const target = Math.min(0, tower.top() - 260);
  camera += (target - camera) * .08;
  context.save(); context.translate(0, -camera);
  const first = Math.floor((field.floor - camera) / 100);
  for (let level = Math.max(0, first - Math.ceil(field.height / 100)); level <= first + 1; level++) {
    const y = field.floor - level * 100;
    context.beginPath(); context.moveTo(42, y); context.lineTo(field.width - 24, y);
    context.strokeStyle = '#34555a'; context.lineWidth = 1; context.setLineDash([2, 9]); context.stroke(); context.setLineDash([]);
    context.fillStyle = '#91aaa5'; context.font = '11px ui-monospace, monospace'; context.fillText(`${level}m`, 12, y + 4);
  }
  const left = (field.width - field.platformWidth) / 2;
  context.beginPath(); context.moveTo(left + 12, field.floor + 24); context.lineTo(left - 28, field.floor + 110);
  context.lineTo(field.width / 2 - 12, field.floor + 155); context.lineTo(left + field.platformWidth + 22, field.floor + 105);
  context.lineTo(left + field.platformWidth - 12, field.floor + 24); context.closePath(); context.fillStyle = '#506a60'; context.fill();
  context.beginPath(); context.moveTo(left + 12, field.floor + 24); context.lineTo(field.width / 2 - 12, field.floor + 155);
  context.lineTo(left + field.platformWidth + 22, field.floor + 105); context.lineTo(left + field.platformWidth - 12, field.floor + 24); context.closePath(); context.fillStyle = '#3f5c57'; context.fill();
  context.fillStyle = '#b5c797'; context.fillRect(left, field.floor, field.platformWidth, 24);
  context.fillStyle = '#d2e5aa'; context.fillRect(left, field.floor, field.platformWidth, 4);
  for (const body of tower.bodies) drawAsset(body, body.plugin.asset);
  if (tower.phase === 'aiming' && tower.active) {
    const { body, asset } = tower.active;
    context.beginPath(); context.moveTo(body.position.x, body.bounds.max.y + 12); context.lineTo(body.position.x, tower.top());
    context.strokeStyle = '#aac59070'; context.setLineDash([4, 8]); context.stroke(); context.setLineDash([]);
    drawAsset(body, asset);
  }
  context.restore();
}
function frame(now) {
  const elapsed = Math.min(50, Math.max(0, now - previous)); previous = now;
  if (playing() && !paused()) {
    accumulator += elapsed;
    const step = 1000 / 120;
    while (accumulator >= step) {
      if (mouseX !== null) tower.moveTo(mouseX);
      if (keys.has('a') || keys.has('arrowleft') || keys.has('move-left')) tower.move(-1, 1.8);
      if (keys.has('d') || keys.has('arrowright') || keys.has('move-right')) tower.move(1, 1.8);
      tower.step(step); accumulator -= step;
      if (tower.phase === 'idle') nextObject();
    }
  } else accumulator = 0;
  refresh(); draw(); requestAnimationFrame(frame);
}
if (location.hash === '#collision-test') $('lab').open = true;
refresh(); requestAnimationFrame(frame);
