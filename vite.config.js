import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

// transformers.js imports "onnxruntime-web/webgpu" and "onnxruntime-common".
// Pointing both at the WASM-only build keeps a single ORT instance in the
// bundle and avoids loading the 28 MB WebGPU (JSEP) runtime. The tokenizer is
// the only part of transformers.js used, so it never needs WebGPU.
const ORT_WASM = fileURLToPath(
  new URL("./node_modules/onnxruntime-web/dist/ort.wasm.min.mjs", import.meta.url),
);

export default defineConfig({
  // Relative asset URLs, so the build works under any GitHub Pages subpath.
  base: "./",
  resolve: {
    alias: [
      { find: /^onnxruntime-web\/webgpu$/, replacement: ORT_WASM },
      { find: /^onnxruntime-common$/, replacement: ORT_WASM },
      { find: /^onnxruntime-web$/, replacement: ORT_WASM },
    ],
  },
  // Pre-bundling rewrites ORT's runtime loader, which then can't find its
  // .mjs/.wasm files in public/ort/.
  optimizeDeps: {
    exclude: ["onnxruntime-web", "@huggingface/transformers"],
  },
  // src/worker.js runs as an ES module worker: ORT loads its WASM glue with a
  // dynamic import(), which the default (iife) worker format can't contain.
  worker: {
    format: "es",
  },
  build: {
    target: "es2022",
    // transformers.js alone is ~440 kB minified; the default 500 kB warning
    // would fire on every build.
    chunkSizeWarningLimit: 700,
  },
});
