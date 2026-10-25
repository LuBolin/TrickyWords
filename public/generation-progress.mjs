export function generationProgress(page) {
  const list = page.getElementById('loading-stages');
  const count = page.getElementById('loading-stage-count');
  const stages = [...list.children].map(item => ({
    item, ball: item.querySelector('.stage-ball'), label: item.querySelector('.stage-label').textContent,
  }));
  function show(index) {
    stages.forEach(({ item, ball, label }, i) => {
      const state = i < index ? 'done' : i === index ? 'active' : 'pending';
      item.dataset.state = state;
      item.setAttribute('aria-label', `${label}: ${state === 'done' ? 'completed' : state === 'active' ? 'current stage' : 'not started'}`);
      if (state === 'active') item.setAttribute('aria-current', 'step');
      else item.removeAttribute('aria-current');
      ball.textContent = state === 'done' ? '✓' : String(i + 1);
    });
    count.textContent = index === stages.length ? `All ${stages.length} stages complete` :
      `Stage ${index + 1} of ${stages.length} · ${index} done · ${stages.length - index - 1} after this`;
  }
  return {
    reset(visible) {
      list.hidden = count.hidden = !visible;
      if (visible) show(0);
    },
    stage(key) {
      const index = stages.findIndex(({ item }) => item.dataset.stage === key);
      if (index < 0) throw new Error(`Unknown generation stage: ${key}`);
      show(index);
    },
    complete() { show(stages.length); },
  };
}
