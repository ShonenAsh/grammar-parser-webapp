import { splitSentences } from "./tokenizer/sentences.js";
import { Parser } from "./parser/parser.js";
import { applyAttributeRules } from "./parser/attributeRuler.js";
import { Lemmatizer } from "./lemmatizer/lemmatizer.js";
import { TokenTable, biber } from "./features/biber.js";
import { dimensionProfile } from "./features/dimensions.js";

// Sentences outside this range are not parsed (and so not counted in the
// features); the parser's input is capped at 256 subwords.
const MIN_WORDS = 2;
const MAX_WORDS = 100;

// The model files (see Parser.init) are served from the Hugging Face Hub,
// pinned to a commit so a new upload can't change the deployed site. To test
// a fresh export, put it in public/model/ and run with VITE_MODEL_URL=model/
// (resolved against the site base URL).
const MODEL_URL = import.meta.env.VITE_MODEL_URL ||
  "https://huggingface.co/shonenash/electra-small-tagger-parser/resolve/0e02dd467eb493f1cb6dde2ff7307898dd951275/";

const parser = new Parser();
const lemmatizer = new Lemmatizer();
let loading = null;

// Loads the model, tokenizer and lemma data. siteBaseUrl is the absolute URL
// the site is served from (ORT's WASM and the lemma data live under it). This
// module runs in a Web Worker (src/worker.js), so it can't work that out from
// the page itself. Resolves once everything is loaded; after a failure, the
// next call retries.
export function ready(siteBaseUrl) {
  loading ??= Promise.all([
    parser.init(new URL(MODEL_URL, siteBaseUrl).href, new URL("ort/", siteBaseUrl).href),
    lemmatizer.init(new URL("data/en_lemma_data.json", siteBaseUrl).href),
  ]).catch((e) => { loading = null; throw e; });
  return loading;
}

// The steps of analyze(), exported separately so a pool of workers can split
// the work: splitSentences() once, analyzeSentence() for each sentence on any
// worker, then extractFeatures() on all of them. See src/analysisClient.js.
export { splitSentences };

// One sentence (an array of words) -> { words, pos, upos, heads, deps, lemmas },
// or null if it is too short or too long to parse. pos holds Penn tags and
// upos spaCy's coarse POS. heads are 1-based word indices, with 0 meaning ROOT.
export async function analyzeSentence(words) {
  if (!loading) throw new Error("analyzeSentence() called before ready()");
  await loading;
  if (words.length < MIN_WORDS || words.length > MAX_WORDS) return null;
  const r = await parser.parse(words);
  if (!r) return null;
  const attrs = applyAttributeRules(r.words, r.pos, r.deps);
  r.upos = attrs.pos;
  // As in spaCy, a lemma set by the attribute ruler wins over the lemmatizer.
  r.lemmas = r.words.map((w, i) => attrs.lemmas[i] ?? lemmatizer.lemmatize(w, attrs.pos[i], attrs.morphs[i]));
  return r;
}

// Returns {
//   sentences: [analyzeSentence() results],
//   features: { f_01_past_tense: { count, per1000 }, ... },
//   dimensions: Biber dimension profile, see features/dimensions.js (null if no words),
//   skipped: number of sentences not analyzed (too short or too long),
// }
// The text is treated as one document. onProgress(done, total), if given, is
// called after each sentence.
export async function analyze(text, onProgress) {
  const all = splitSentences(text);
  const sentences = [];
  let skipped = 0;
  for (const [k, words] of all.entries()) {
    const r = await analyzeSentence(words);
    onProgress?.(k + 1, all.length);
    if (r) sentences.push(r);
    else skipped++;
  }
  return { sentences, ...extractFeatures(sentences), skipped };
}

// analyzeSentence() results -> {
//   features: { f_01_past_tense: { count, per1000 }, ... },
//   dimensions: dimensionProfile() result or null,
// }
export function extractFeatures(sentences) {
  if (sentences.length === 0) return { features: {}, dimensions: null };
  const table = TokenTable.buildTable([{ sentences }]);
  const counts = biber(table, { normalize: false });
  const rates = biber(table);
  const features = Object.fromEntries(
    Object.keys(counts).map((name) => [name, { count: counts[name][0], per1000: rates[name][0] }]),
  );
  return { features, dimensions: dimensionProfile(sentences, features) };
}
