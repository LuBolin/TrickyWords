export function startupFeedback(page) {
  const $ = id => page.getElementById(id);
  const started = performance.now();
  let stepStarted = started, active = true;
  function tick() {
    const elapsed = `${((performance.now() - started) / 1000).toFixed(1)} s elapsed`;
    $('startup-time').textContent = elapsed;
    $('elapsed').textContent = elapsed;
    if (performance.now() - stepStarted >= 10000) {
      $('startup-note').textContent = 'This step is taking longer than usual. You can wait or retry.';
      $('startup-retry').hidden = false;
    }
  }
  const timer = setInterval(tick, 250);
  function stop() {
    tick();
    active = false;
    clearInterval(timer);
    page.removeEventListener('startup-stage', onStage);
    $('startup').setAttribute('aria-busy', 'false');
    $('progress').value = 0;
    $('progress').setAttribute('aria-label', 'Generation progress');
  }
  function fail(error) {
    if (!active) return;
    stop();
    const message = error.message || String(error);
    $('startup-stage').textContent = 'Startup failed';
    $('startup-note').textContent = message;
    $('startup-spinner').hidden = true;
    $('startup-progress').hidden = true;
    $('startup-retry').hidden = false;
    $('status').textContent = message;
    $('status').classList.add('error');
    $('prepare').disabled = true;
    $('start-game').disabled = true;
  }
  function onStage({ detail }) {
    if (!active) return;
    if (detail.error) { fail(new Error(detail.message)); return; }
    if (detail.complete) {
      stop();
      $('startup').hidden = true;
      $('elapsed').textContent = '';
      return;
    }
    stepStarted = performance.now();
    $('startup-stage').textContent = detail.message;
    $('status').textContent = detail.message;
    $('startup-step').textContent = `Check ${detail.step} of 4`;
    $('startup-note').textContent = 'Choose a model folder after these checks.';
    $('startup-retry').hidden = true;
    tick();
  }
  page.addEventListener('startup-stage', onStage);
  $('startup-retry').addEventListener('click', () => page.defaultView.location.reload());
  $('progress').removeAttribute('value');
  $('progress').setAttribute('aria-label', 'Startup progress');
  tick();
  return { fail };
}

// Start feedback before fetching or evaluating the game and AI dependencies.
if (typeof document !== 'undefined') {
  const feedback = globalThis.trickyWordsStartup || startupFeedback(document);
  import('./game.mjs').then(() => import('./app.mjs')).catch(feedback.fail);
}
