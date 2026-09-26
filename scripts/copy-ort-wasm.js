// Copies onnxruntime-web's WASM runtime into public/ort/ so it is served as a
// static file (see ort.env.wasm.wasmPaths in src/parser/parser.js). Runs after
// every install, keeping it in sync with the installed onnxruntime-web version.
import { copyFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const from = `${root}node_modules/onnxruntime-web/dist/`;
const to = `${root}public/ort/`;
// The plain SIMD build only. The JSEP (WebGPU) build is 28 MB and unused.
const FILES = ["ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"];

mkdirSync(to, { recursive: true });
for (const f of FILES) copyFileSync(from + f, to + f);
