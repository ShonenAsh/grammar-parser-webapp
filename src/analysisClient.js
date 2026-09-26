// Main-thread side of the analysis: a pool of Web Workers (src/worker.js),
// each with its own copy of the model on its own CPU core. Exposes the same
// ready() and analyze() as src/pipeline.js. analyze() splits the text into
// sentences, hands each sentence to whichever worker is free, and computes the
// features once all sentences are back, so the page stays responsive and long
// texts are parsed in parallel.

// Each worker holds its own model, so memory grows with the pool size and
// parsing speeds up nearly linearly with it.
export const MAX_POOL_SIZE = 4;
const STORAGE_KEY = "biber.poolSize";

// The user's last choice, else as many workers as cores allow (one core is
// left for the page), up to MAX_POOL_SIZE.
function initialPoolSize() {
  const saved = Number(localStorage.getItem(STORAGE_KEY));
  if (saved >= 1 && saved <= MAX_POOL_SIZE) return Math.floor(saved);
  return Math.max(1, Math.min(MAX_POOL_SIZE, (navigator.hardwareConcurrency || 2) - 1));
}

// Where public/ files are served from, as an absolute URL. Workers have no
// document to resolve relative URLs against, so the page resolves it.
const siteBaseUrl = new URL(import.meta.env.BASE_URL, document.baseURI).href;

// A worker plus request/reply bookkeeping: each request gets an id so that
// replies, which arrive as separate messages, can be matched to their promise.
class PoolWorker {
  constructor() {
    // new URL(..., import.meta.url) is the pattern Vite recognizes: it bundles
    // worker.js and its imports as a separate file and fixes up this URL.
    this.worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    this.nextId = 0;
    this.pending = new Map();  // id -> { resolve, reject }
    this.loading = null;
    this.worker.onmessage = (event) => {
      const { id, result, error } = event.data;
      const request = this.pending.get(id);
      this.pending.delete(id);
      if (error !== undefined) request?.reject(new Error(error));
      else request?.resolve(result);
    };
    // Fires if the worker script itself fails, e.g. a module that doesn't load.
    this.worker.onerror = (event) => this.failAll(new Error(event.message || "analysis worker failed to start"));
  }

  call(op, args) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, op, args });
    });
  }

  // Loads the model in this worker; after a failure, the next call retries.
  ready() {
    this.loading ??= this.call("init", { siteBaseUrl }).catch((e) => { this.loading = null; throw e; });
    return this.loading;
  }

  terminate() {
    this.worker.terminate();
    this.failAll(new Error("analysis worker stopped"));
  }

  failAll(error) {
    for (const request of this.pending.values()) request.reject(error);
    this.pending.clear();
  }
}

const pool = [];

export function poolSize() {
  return pool.length;
}

// Grows or shrinks the pool to n workers (1..MAX_POOL_SIZE) and remembers the
// choice. New workers load the model; the returned promise resolves when all
// workers are ready. Call it only while no analysis is running.
export function setPoolSize(n) {
  const size = Math.max(1, Math.min(MAX_POOL_SIZE, Math.floor(n)));
  localStorage.setItem(STORAGE_KEY, String(size));
  while (pool.length > size) pool.pop().terminate();
  while (pool.length < size) pool.push(new PoolWorker());
  return ready();
}

// Resolves once every worker has loaded the model. Loading starts as soon as
// this module is imported.
export function ready() {
  return Promise.all(pool.map((w) => w.ready()));
}

setPoolSize(initialPoolSize()).catch(() => {});

// Same result as pipeline.js's analyze(): { sentences, features, skipped }.
// onProgress(done, total) is called as sentences come back.
export async function analyze(text, onProgress) {
  await ready();
  const workers = [...pool];
  const [first] = workers;
  const allWords = await first.call("split", { text });

  // Each worker takes the next unparsed sentence as soon as it is free.
  // Results are stored by index, so their order doesn't depend on timing.
  const results = new Array(allWords.length);
  let next = 0, done = 0;
  await Promise.all(workers.map(async (w) => {
    while (next < allWords.length) {
      const k = next++;
      results[k] = await w.call("sentence", { words: allWords[k] });
      onProgress?.(++done, allWords.length);
    }
  }));

  const sentences = results.filter((r) => r !== null);
  const features = await first.call("features", { sentences });
  return { sentences, features, skipped: results.length - sentences.length };
}
