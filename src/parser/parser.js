import * as ort from "onnxruntime-web";
import { AutoTokenizer, env } from "@huggingface/transformers";
import { decodeMST } from "./decode.js";
import { shapeOf } from "./shapes.js";

const DEBUG = false;
const log = (...a) => DEBUG && console.log("[parser]", ...a);

async function fetchOk(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res;
}

// Different tokenizer versions expose special-token ids slightly differently;
// try the vocab map first, then fall back to encoding an empty string.
function resolveSpecials(tok) {
  const m = tok.model?.tokens_to_ids;
  const get = (name) => m?.get?.(name) ?? m?.[name];
  let clsId = get("[CLS]");
  let sepId = get("[SEP]");
  if (clsId == null || sepId == null) {
    const enc = tok("");
    const arr = Array.from(enc.input_ids.data, Number);
    if (arr.length >= 2) { clsId = arr[0]; sepId = arr[arr.length - 1]; }
  }
  const unk = tok.unk_token ?? "[UNK]";
  const unkId = tok.unk_token_id ?? get(unk) ?? tok.convert_tokens_to_ids?.([unk])[0];
  if (clsId == null || sepId == null || unkId == null) {
    throw new Error("Could not resolve [CLS]/[SEP]/[UNK] ids");
  }
  return { clsId: Number(clsId), sepId: Number(sepId), unkId: Number(unkId) };
}

// Encodes each word alone (no specials), prepends [CLS] and appends [SEP], and
// builds wordIds in lockstep. Deterministic and doesn't depend on any tokenizer
// API method that may or may not exist on the JS side. Mirrors training
// (util/prepare_data.py): a word that yields no pieces (e.g. a lone zero-width
// space) becomes [UNK] so later words stay aligned, and truncation keeps [SEP]
// last, as HF's truncation=True does.
async function encodePreTokenized(words, tokenizer, { clsId, sepId, unkId }, maxLen = 256) {
  const inputIds = [clsId];
  const wordIds  = [null];
  for (let wi = 0; wi < words.length; wi++) {
    const enc = await tokenizer(words[wi], { add_special_tokens: false });
    // enc.input_ids is a Tensor; .data is a BigInt64Array
    const ids = Array.from(enc.input_ids.data, (x) => Number(x));
    if (ids.length === 0) ids.push(unkId);
    for (const id of ids) {
      inputIds.push(id);
      wordIds.push(wi);
    }
  }

  if (inputIds.length > maxLen - 1) {
    inputIds.length = maxLen - 1;
    wordIds.length  = maxLen - 1;
  }
  inputIds.push(sepId);
  wordIds.push(null);
  return { inputIds, wordIds };
}

export class Parser {
  // Both arguments are absolute URLs: modelBaseUrl is the directory holding
  // manifest.json, the parser model it names (parser.uint8.onnx),
  // bilinear.onnx and a tokenizer/ subdirectory; wasmBaseUrl holds ORT's .wasm runtime (public/ort/, filled
  // by scripts/copy-ort-wasm.js). They can't be resolved here because this
  // runs in a Web Worker, which has no document to resolve against.
  async init(modelBaseUrl, wasmBaseUrl) {
    const base = new URL(modelBaseUrl).href.replace(/\/?$/, "/");

    // Set here rather than at module load: transformers.js points the shared
    // ort.env at its CDN when it is imported.
    ort.env.wasm.wasmPaths = wasmBaseUrl;
    // Multithreading needs cross-origin isolation, which GitHub Pages can't provide.
    ort.env.wasm.numThreads = 1;
    ort.env.logLevel = "error";

    // Load the tokenizer as a "remote" model rooted at base, which works both
    // cross-origin (the Hub) and same-origin: "tokenizer" -> base/tokenizer/.
    env.allowLocalModels = false;
    env.allowRemoteModels = true;
    env.remoteHost = base;
    env.remotePathTemplate = "{model}/";

    const loadModel = fetchOk(base + "manifest.json")
      .then((r) => r.json())
      .then(async (m) => [m, await fetchOk(base + m.model).then((r) => r.arrayBuffer())]);
    const [[manifest, modelBuf], labelerBuf, tokenizer] = await Promise.all([
      loadModel,
      fetchOk(base + "bilinear.onnx").then((r) => r.arrayBuffer()),
      AutoTokenizer.from_pretrained("tokenizer"),
    ]);

    const opts = { executionProviders: ["wasm"] };
    this.manifest = manifest;
    this.session = await ort.InferenceSession.create(new Uint8Array(modelBuf), opts);
    // Bilinear label scorer, see util/export_bilinear_onnx.py.
    this.labeler = await ort.InferenceSession.create(new Uint8Array(labelerBuf), opts);
    this.tokenizer = tokenizer;
    this.specials = resolveSpecials(tokenizer);
    log("special ids", this.specials);
  }

  async parse(words) {
    const { session, labeler, tokenizer, specials, manifest } = this;
    const { tags, deps, tag_dim: tagDim, n_deps: nDeps } = manifest;
    const { inputIds, wordIds } = await encodePreTokenized(words, tokenizer, specials);
    const T = inputIds.length;

    const firstSub = new Map();
    for (let pos = 0; pos < wordIds.length; pos++) {
      const wid = wordIds[pos];
      if (wid != null && !firstSub.has(wid)) firstSub.set(wid, pos);
    }
    const W = firstSub.size;
    if (W === 0) return null;

    const wordIndex = [0, ...Array.from({ length: W }, (_, i) => firstSub.get(i) ?? 0)];
    const shapes = words.slice(0, W).map(shapeOf);

    const feeds = {
      input_ids:      new ort.Tensor("int64", new BigInt64Array(inputIds.map(BigInt)), [1, T]),
      attention_mask: new ort.Tensor("int64", new BigInt64Array(T).fill(1n),           [1, T]),
      word_index:     new ort.Tensor("int64", new BigInt64Array(wordIndex.map(BigInt)), [1, W + 1]),
      shapes:         new ort.Tensor("int64", new BigInt64Array(shapes.map(BigInt)),   [1, W]),
    };

    const out = await session.run(feeds);
    const posLogits = out.pos_logits.data;
    const arcScores = out.attended_arcs.data;
    const headTagD  = out.head_tag.data;
    const deptTagD  = out.dept_tag.data;

    const nPos = tags.length;
    const Wr = W + 1;

    const posIds = Array.from({ length: W }, (_, i) => {
      let best = 0, bs = -Infinity;
      const off = i * nPos;
      for (let k = 0; k < nPos; k++) if (posLogits[off + k] > bs) { bs = posLogits[off + k]; best = k; }
      return best;
    });

    const predHeads = decodeMST(arcScores, Wr);

    // Label every word in one call: pair word d's dept_tag vector with the
    // head_tag vector of its predicted head.
    const headVecs = new Float32Array(W * tagDim);
    const deptVecs = new Float32Array(W * tagDim);
    for (let i = 0; i < W; i++) {
      const d = i + 1;
      const h = predHeads[d];
      headVecs.set(headTagD.subarray(h * tagDim, (h + 1) * tagDim), i * tagDim);
      deptVecs.set(deptTagD.subarray(d * tagDim, (d + 1) * tagDim), i * tagDim);
    }
    const { scores } = await labeler.run({
      head: new ort.Tensor("float32", headVecs, [W, tagDim]),
      dep:  new ort.Tensor("float32", deptVecs, [W, tagDim]),
    });
    const labelScores = scores.data;

    const predDeps = Array.from({ length: W }, (_, i) => {
      let best = 0, bs = -Infinity;
      const off = i * nDeps;
      for (let k = 0; k < nDeps; k++) if (labelScores[off + k] > bs) { bs = labelScores[off + k]; best = k; }
      return best;
    });

    return {
      words: words.slice(0, W),
      pos: posIds.map((i) => tags[i]),
      heads: Array.from(predHeads.slice(1)),
      deps: predDeps.map((i) => deps[i]),
    };
  }
}
