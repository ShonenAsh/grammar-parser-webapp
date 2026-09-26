// Chu-Liu-Edmonds (ported from GraphDecoder / AllenNLP).
// Score matrix convention: mat[head * n + dep]  (head-major).
// The model outputs attended_arcs as dep-major [d][h], so we transpose when
// calling decodeMST — same fix as eval.py.

function _findCycle(parents, n, active) {
  const added = new Uint8Array(n);
  added[0] = 1;
  for (let i = 1; i < n; i++) {
    if (added[i] || !active[i]) continue;
    const path = [];
    let node = i;
    while (node !== 0 && !added[node]) {
      added[node] = 1;
      path.push(node);
      node = parents[node];
    }
    if (node !== 0 && path.includes(node)) {
      const k = path.indexOf(node);
      return path.slice(k);
    }
    for (const p of path) added[p] = 2;
  }
  return null;
}

function _cle(n, mat, active, finalEdges, oldIn, oldOut, reps) {
  const parents = new Int32Array(n);
  for (let d = 1; d < n; d++) {
    if (!active[d]) continue;
    let best = 0, bs = mat[0 * n + d];
    for (let h = 1; h < n; h++) {
      if (h === d || !active[h]) continue;
      if (mat[h * n + d] > bs) { bs = mat[h * n + d]; best = h; }
    }
    parents[d] = best;
  }

  const cycle = _findCycle(parents, n, active);
  if (!cycle) {
    finalEdges[0] = -1;
    for (let d = 1; d < n; d++) {
      if (!active[d]) continue;
      finalEdges[oldOut[parents[d] * n + d]] = oldIn[parents[d] * n + d];
    }
    return;
  }

  const inCycle = new Set(cycle);
  const cr = cycle[0];
  const cw = cycle.reduce((s, c) => s + mat[parents[c] * n + c], 0);

  const savedIn = new Map(), savedOut = new Map();
  for (let node = 0; node < n; node++) {
    if (!active[node] || inCycle.has(node)) continue;
    let bi = -1, bis = -Infinity, bo = -1, bos = -Infinity;
    for (const c of cycle) {
      if (mat[c * n + node] > bis) { bis = mat[c * n + node]; bi = c; }
      const sc = cw + mat[node * n + c] - mat[parents[c] * n + c];
      if (sc > bos) { bos = sc; bo = c; }
    }
    savedIn.set(node, mat[cr * n + node]);
    savedOut.set(node, mat[node * n + cr]);
    mat[cr * n + node] = bis;
    oldIn[cr * n + node] = oldIn[bi * n + node];
    oldOut[cr * n + node] = oldOut[bi * n + node];
    mat[node * n + cr] = bos;
    oldOut[node * n + cr] = oldOut[node * n + bo];
    oldIn[node * n + cr] = oldIn[node * n + bo];
  }

  const considered = cycle.map((c, i) => {
    const s = new Set(reps[c]);
    if (i > 0) { active[c] = 0; for (const r of s) reps[cr].add(r); }
    return s;
  });

  _cle(n, mat, active, finalEdges, oldIn, oldOut, reps);

  let keyNode = -1;
  outer: for (let i = 0; i < cycle.length; i++) {
    for (const r of considered[i]) {
      if (r in finalEdges) { keyNode = cycle[i]; break outer; }
    }
  }
  let prev = parents[keyNode];
  while (prev !== keyNode) {
    finalEdges[oldOut[parents[prev] * n + prev]] = oldIn[parents[prev] * n + prev];
    prev = parents[prev];
  }
  for (const c of cycle) {
    if (c !== keyNode) finalEdges[oldOut[parents[c] * n + c]] = oldIn[parents[c] * n + c];
  }
}

export function decodeMST(arcScoresDepMajor, n) {
  // arcScoresDepMajor[d * n + h] — model convention.
  // Build head-major mat for CLE: mat[h * n + d].
  const mat = new Float64Array(n * n);
  for (let d = 0; d < n; d++)
    for (let h = 0; h < n; h++)
      mat[h * n + d] = arcScoresDepMajor[d * n + h];

  for (let i = 0; i < n; i++) mat[i * n + i] = 0;

  const oldIn = new Int32Array(n * n);
  const oldOut = new Int32Array(n * n);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      oldIn[i * n + j] = i;
      oldOut[i * n + j] = j;
    }

  const active = new Uint8Array(n).fill(1);
  const reps = Array.from({ length: n }, (_, i) => new Set([i]));
  const finalEdges = {};

  _cle(n, mat, active, finalEdges, oldIn, oldOut, reps);

  const heads = new Int32Array(n);
  for (const [d, h] of Object.entries(finalEdges)) heads[+d] = h;
  heads[0] = 0;
  return heads;
}
