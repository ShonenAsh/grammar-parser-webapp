importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js");
// Tell ONNX where to find the WASM binaries
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

let session;
let vocab;
let tagMap;

onmessage = async (e) => {
    const { type, text } = e.data;

    try {
        if (type == "LOAD") {
            // Fetch metadata
            vocab = await fetch("./vocab.json").then(r => r.json());
            const tagToId = await fetch("./tag_to_id.json").then(r => r.json());
            tagMap = Object.fromEntries(Object.entries(tagToId).map(([k, v]) => [v, k]));

            // Initialize ONNX Session
            session = await ort.InferenceSession.create("./pos_tagger.onnx");
            postMessage({ type: "READY" });
        }

        if (type == "INFER") {
            const words = text.toLowerCase().match(/\w+|[^\w\s]/g) || [];
            const inputIds = words.map(w => BigInt(vocab[w] ?? vocab["<UNK>"]));

            const tensor = new ort.Tensor("int64", BigInt64Array.from(inputIds), [1, inputIds.length]);
            const results = await session.run({ input: tensor });

            const logits = results.output.data;
            const [batch, seqLen, numTags] = results.output.dims;
            
            // logits to predictions
            const predictions = words.map((word, i) => {
                let maxVal = -Infinity;
                let maxId = 0;
                for (let j = 0; j < numTags; j++) {
                    const val = logits[i * numTags + j];
                    if (val > maxVal) {
                        maxVal = val;
                        maxId = j;
                    }
                }
                return { word, tag: tagMap[maxId] };
            });

            postMessage({ type: "RESULT", predictions });
        }
    } catch (err) {
        postMessage({ type: "ERROR", error: err.message });
    }
};
