const cross = (a, b, c) => (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
const same = (a, b) => a[0] === b[0] && a[1] === b[1];
export const polygonArea = points => points.reduce((sum, p, i) => {
  const q = points[(i + 1) % points.length];
  return sum + p[0] * q[1] - q[0] * p[1];
}, 0) / 2;

function removeCollinear(points) {
  return points.filter((p, i) => cross(points[(i + points.length - 1) % points.length], p, points[(i + 1) % points.length]) !== 0);
}

function simplifyLine(points, tolerance) {
  const keep = new Set([0, points.length - 1]), pending = [[0, points.length - 1]];
  while (pending.length) {
    const [start, end] = pending.pop(), a = points[start], b = points[end];
    const dx = b[0] - a[0], dy = b[1] - a[1], length = dx * dx + dy * dy;
    let farthest = -1, distance = tolerance * tolerance;
    for (let i = start + 1; i < end; i++) {
      const p = points[i];
      const t = length ? Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length)) : 0;
      const d = (p[0] - a[0] - t * dx) ** 2 + (p[1] - a[1] - t * dy) ** 2;
      if (d > distance) { farthest = i; distance = d; }
    }
    if (farthest !== -1) {
      keep.add(farthest);
      pending.push([start, farthest], [farthest, end]);
    }
  }
  return [...keep].sort((a, b) => a - b).map(i => points[i]);
}

function isSimple(points) {
  const onSegment = (a, b, p) => cross(a, b, p) === 0 && p[0] >= Math.min(a[0], b[0]) && p[0] <= Math.max(a[0], b[0]) && p[1] >= Math.min(a[1], b[1]) && p[1] <= Math.max(a[1], b[1]);
  for (let i = 0; i < points.length; i++) for (let j = i + 1; j < points.length; j++) {
    if (j === i + 1 || (i === 0 && j === points.length - 1)) continue;
    const a = points[i], b = points[(i + 1) % points.length], c = points[j], d = points[(j + 1) % points.length];
    if ((cross(a, b, c) * cross(a, b, d) < 0 && cross(c, d, a) * cross(c, d, b) < 0) ||
        onSegment(a, b, c) || onSegment(a, b, d) || onSegment(c, d, a) || onSegment(c, d, b)) return false;
  }
  return true;
}

function simplifyOutline(points) {
  const exact = removeCollinear(points), area = polygonArea(exact);
  let split = 1, longest = 0;
  for (let i = 1; i < exact.length; i++) {
    const distance = (exact[i][0] - exact[0][0]) ** 2 + (exact[i][1] - exact[0][1]) ** 2;
    if (distance > longest) { split = i; longest = distance; }
  }
  for (const tolerance of [1.5, 2, 3, 4, 6]) {
    const candidate = removeCollinear([
      ...simplifyLine(exact.slice(0, split + 1), tolerance).slice(0, -1),
      ...simplifyLine([...exact.slice(split), exact[0]], tolerance).slice(0, -1),
    ]);
    if (candidate.length >= 3 && candidate.length <= 96 && Math.abs(polygonArea(candidate) / area - 1) <= .03 && isSimple(candidate)) {
      return { outline: candidate, tolerance };
    }
  }
  if (exact.length <= 96 && isSimple(exact)) return { outline: exact, tolerance: 0 };
  throw new Error('This silhouette is too complex or touches itself. Try another word or seed.');
}

function convexParts(outline) {
  const remaining = outline.slice(), parts = [];
  while (remaining.length > 3) {
    const ear = remaining.findIndex((b, i) => {
      const a = remaining[(i + remaining.length - 1) % remaining.length], c = remaining[(i + 1) % remaining.length];
      return cross(a, b, c) > 0 && !remaining.some(p => p !== a && p !== b && p !== c && cross(a, b, p) >= 0 && cross(b, c, p) >= 0 && cross(c, a, p) >= 0);
    });
    if (ear < 0) throw new Error('Could not split the collision outline safely. Try another seed.');
    parts.push([remaining[(ear + remaining.length - 1) % remaining.length], remaining[ear], remaining[(ear + 1) % remaining.length]]);
    remaining.splice(ear, 1);
  }
  parts.push(remaining);
  // Merge neighboring triangles, with at most eight vertices per physics fixture.
  let merged = true;
  while (merged) {
    merged = false;
    outer: for (let i = 0; i < parts.length; i++) for (let j = i + 1; j < parts.length; j++) {
      const a = parts[i], b = parts[j];
      for (let ai = 0; ai < a.length; ai++) for (let bi = 0; bi < b.length; bi++) {
        if (!same(a[ai], b[(bi + 1) % b.length]) || !same(a[(ai + 1) % a.length], b[bi])) continue;
        const joined = removeCollinear([
          ...a.map((_, k) => a[(ai + 1 + k) % a.length]),
          ...b.slice(2).map((_, k) => b[(bi + 2 + k) % b.length]),
        ]);
        if (joined.length <= 8 && joined.every((p, k) => cross(p, joined[(k + 1) % joined.length], joined[(k + 2) % joined.length]) > 0)) {
          parts[i] = joined;
          parts.splice(j, 1);
          merged = true;
          break outer;
        }
      }
    }
  }
  return parts;
}

export function collisionPolygons(rgba, width, height) {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || width * height > 1024 * 1024 || rgba.length !== width * height * 4) {
    throw new Error('Collision tracing needs RGBA pixels, up to 1024 × 1024.');
  }
  const size = width * height, mask = new Uint8Array(size), visited = new Uint8Array(size), queue = new Int32Array(size);
  let foregroundPixels = 0, largest = [];
  for (let i = 0; i < size; i++) if (rgba[i * 4 + 3] >= 128) { mask[i] = 1; foregroundPixels++; }
  for (let start = 0; start < size; start++) {
    if (!mask[start] || visited[start]) continue;
    let head = 0, tail = 1;
    queue[0] = start; visited[start] = 1;
    while (head < tail) {
      const p = queue[head++], x = p % width;
      for (const next of [x > 0 ? p - 1 : -1, x < width - 1 ? p + 1 : -1, p - width, p + width]) {
        if (next >= 0 && next < size && mask[next] && !visited[next]) { visited[next] = 1; queue[tail++] = next; }
      }
    }
    if (tail > largest.length) largest = queue.slice(0, tail);
  }
  if (!largest.length) throw new Error('No solid pixels found. Try another word, seed, or transparent PNG.');
  mask.fill(0);
  for (const p of largest) mask[p] = 1;
  const stride = width + 1, edges = new Map();
  const edge = (x, y, nx, ny, direction) => {
    const key = y * stride + x;
    if (!edges.has(key)) edges.set(key, []);
    edges.get(key).push({ end: ny * stride + nx, direction });
  };
  for (const p of largest) {
    const x = p % width, y = Math.floor(p / width);
    if (y === 0 || !mask[p - width]) edge(x, y, x + 1, y, 0);
    if (x === width - 1 || !mask[p + 1]) edge(x + 1, y, x + 1, y + 1, 1);
    if (y === height - 1 || !mask[p + width]) edge(x + 1, y + 1, x, y + 1, 2);
    if (x === 0 || !mask[p - 1]) edge(x, y + 1, x, y, 3);
  }
  const loops = [];
  while (edges.size) {
    const start = edges.keys().next().value, points = [];
    let key = start, direction = edges.get(start)[0].direction;
    do {
      points.push([key % stride, Math.floor(key / stride)]);
      const choices = edges.get(key);
      if (!choices?.length) throw new Error('The silhouette boundary is not closed.');
      const priority = next => [1, 0, 3, 2].indexOf((next.direction - direction + 4) % 4);
      choices.sort((a, b) => priority(a) - priority(b));
      const next = choices.shift();
      if (!choices.length) edges.delete(key);
      key = next.end; direction = next.direction;
    } while (key !== start);
    loops.push(points);
  }
  loops.sort((a, b) => polygonArea(b) - polygonArea(a));
  const { outline, tolerance } = simplifyOutline(loops[0]), parts = convexParts(outline);
  if (parts.some(part => part.length < 3 || part.length > 8 || !part.every((p, i) => cross(p, part[(i + 1) % part.length], part[(i + 2) % part.length]) > 0))) {
    throw new Error('The silhouette produced an invalid convex collision piece. Try another seed.');
  }
  const area = polygonArea(outline), partsArea = parts.reduce((sum, part) => sum + polygonArea(part), 0);
  if (!(area > 0) || Math.abs(partsArea - area) > 1e-6) throw new Error('Collision pieces do not cover the outline safely.');
  let cx = 0, cy = 0;
  for (let i = 0; i < outline.length; i++) {
    const a = outline[i], b = outline[(i + 1) % outline.length], weight = a[0] * b[1] - b[0] * a[1];
    cx += (a[0] + b[0]) * weight; cy += (a[1] + b[1]) * weight;
  }
  return {
    version: 1, width, height, coordinates: 'Pixels; origin top-left; x right; y down; clockwise winding.',
    alphaThreshold: 128, simplificationTolerancePx: tolerance,
    outline, convexParts: parts, centroid: [cx / (6 * area), cy / (6 * area)],
    areaPx: area, solidPixels: largest.length, discardedPixels: foregroundPixels - largest.length,
    filledHoles: loops.filter(loop => polygonArea(loop) < 0).length,
    outlineAreaChangePercent: (area / polygonArea(loops[0]) - 1) * 100,
  };
}
