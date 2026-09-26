// One analysis worker. src/analysisClient.js runs a pool of these, off the
// page's main thread; each has its own copy of the model and handles one
// request at a time.
//
// Request:  { id, op, args }   op is one of the OPS below
// Reply:    { id, result } or { id, error }
// Errors are sent as strings: Error objects lose their message in some
// browsers when copied between threads.
import { analyzeSentence, extractFeatures, ready, splitSentences } from "./pipeline.js";

const OPS = {
  init: ({ siteBaseUrl }) => ready(siteBaseUrl),
  split: ({ text }) => splitSentences(text),
  sentence: ({ words }) => analyzeSentence(words),
  features: ({ sentences }) => extractFeatures(sentences),
};

self.onmessage = async (event) => {
  const { id, op, args } = event.data;
  try {
    const handler = OPS[op];
    if (!handler) throw new Error(`unknown op ${op}`);
    self.postMessage({ id, result: await handler(args) });
  } catch (e) {
    self.postMessage({ id, error: String(e) });
  }
};
