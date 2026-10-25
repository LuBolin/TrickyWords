import { collisionPolygons } from './polygons.mjs';

const $ = id => document.getElementById(id);
const sprite = document.createElement('canvas');
sprite.width = sprite.height = 512;
let geometry, loading = 0;

function draw() {
  const context = $('collision').getContext('2d');
  context.clearRect(0, 0, 512, 512);
  context.drawImage(sprite, 0, 0);
  if (!geometry) return;
  const path = points => {
    context.beginPath();
    points.forEach(([x, y], i) => i ? context.lineTo(x, y) : context.moveTo(x, y));
    context.closePath();
  };
  context.lineJoin = 'round';
  if ($('show-parts').checked) geometry.convexParts.forEach((part, i) => {
    path(part);
    context.fillStyle = `hsla(${(i * 67 + 30) % 360}, 70%, 50%, .22)`;
    context.strokeStyle = '#284f47'; context.lineWidth = 1;
    context.fill(); context.stroke();
  });
  path(geometry.outline);
  context.strokeStyle = '#176f53'; context.lineWidth = 2.5; context.stroke();
  for (const [x, y] of geometry.outline) {
    context.beginPath(); context.arc(x, y, 2.5, 0, 2 * Math.PI);
    context.fillStyle = '#f7f8f1'; context.fill(); context.strokeStyle = '#176f53'; context.lineWidth = 1; context.stroke();
  }
}

export function clearCollisionPreview() {
  loading++;
  geometry = null;
  sprite.getContext('2d').clearRect(0, 0, 512, 512);
  draw();
  $('shape-save').hidden = true;
  $('shape-stats').textContent = 'Waiting for an object.';
  $('shape-note').textContent = '';
}

export function showCollision(image, source) {
  clearCollisionPreview();
  const context = sprite.getContext('2d');
  const scale = Math.min(512 / image.width, 512 / image.height);
  context.drawImage(image, (512 - image.width * scale) / 2, (512 - image.height * scale) / 2, image.width * scale, image.height * scale);
  const start = performance.now();
  geometry = { ...collisionPolygons(context.getImageData(0, 0, 512, 512).data, 512, 512), source };
  geometry.traceTimeMs = performance.now() - start;
  draw();
  $('shape-stats').textContent = `${source.label} · ${geometry.outline.length} vertices · ${geometry.convexParts.length} convex pieces · ${geometry.traceTimeMs.toFixed(1)} ms`;
  $('shape-note').textContent = `Largest connected shape. ${geometry.discardedPixels} detached pixels ignored. ${geometry.filledHoles} enclosed holes filled. Outline area changed ${geometry.outlineAreaChangePercent.toFixed(2)}% by simplification.`;
  $('shape-save').href = `data:application/json;charset=utf-8,${encodeURIComponent(JSON.stringify(geometry, null, 2))}`;
  $('shape-save').download = 'collision-shape.json';
  $('shape-save').hidden = false;
  return geometry;
}

export function collisionFailed(error) {
  clearCollisionPreview();
  $('shape-stats').textContent = `Shape unavailable: ${error.message}`;
}
$('show-parts').addEventListener('change', draw);
$('shape-file').addEventListener('change', async event => {
  const file = event.target.files[0];
  if (!file) return;
  const request = ++loading;
  let url;
  try {
    if (file.type !== 'image/png' || file.size > 10 * 1024 * 1024) throw new Error('Choose a PNG smaller than 10 MB.');
    const image = new Image();
    url = URL.createObjectURL(file); image.src = url;
    await image.decode();
    if (image.width > 4096 || image.height > 4096) throw new Error('Choose a PNG no larger than 4096 × 4096.');
    if (request === loading) {
      try { showCollision(image, { label: file.name }); }
      catch (error) { collisionFailed(error); }
    }
  } catch (error) { if (request === loading) collisionFailed(error); }
  finally { if (url) URL.revokeObjectURL(url); }
});
