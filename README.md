# grammar-parser

Biber's multidimensional-analysis features, computed entirely in the browser. Paste a
text and get all 67 features (as in [PyBiber](https://pypi.org/project/pybiber/) with
spaCy's `en_core_web_sm`), plus a POS tag and dependency parse for every sentence.

The tagger/parser is ELECTRA-small fine-tuned for POS tagging and biaffine dependency
parsing, running in [onnxruntime-web](https://onnxruntime.ai/) inside Web Workers.

## Develop

```sh
bun install        # also copies the ORT WASM runtime into public/ort/
bun run dev        # http://localhost:5173
bun run typecheck
bun run build      # -> dist/
bun run preview    # serve dist/, http://localhost:4173
```

The site must be served through Vite (or from `dist/`); opening the source files with
a plain static server won't work.

## Deploy

`.github/workflows/deploy.yml` builds on every push to `main` and publishes `dist/` to
GitHub Pages. In the repo settings set Pages -> Source to "GitHub Actions".

## Layout

```
index.html, src/main.js    UI
src/analysisClient.js      worker pool: splits sentences across Web Workers
src/worker.js              one worker, with its own model copy
src/pipeline.js            sentences -> parse -> coarse POS -> lemmas -> features
src/tokenizer/             spaCy-port word tokenizer and sentence splitter
src/parser/                ONNX inference, tree decoding, spaCy attribute ruler
src/lemmatizer/            spaCy rule lemmatizer
src/features/              Biber features, ported from PyBiber
public/model/              local model export (gitignored; the site loads from the HF Hub)
public/data/               lemmatizer tables
model/, util/              training, evaluation and export scripts (Python)
runs/, web/                trained checkpoint and raw ONNX export
```

## Rebuilding the model files

After training (`model/train.py`, checkpoint in `runs/electra/`):

```sh
uv run python model/onnx_export.py --run runs/electra --out web/ --no-quantize
uv run python util/quantize_parser.py web/parser.onnx    # -> public/model/parser.uint8.onnx
uv run python util/export_bilinear_onnx.py               # -> public/model/bilinear.onnx
cp runs/electra/tokenizer.json runs/electra/tokenizer_config.json public/model/tokenizer/
```

Copy `web/manifest.json` to `public/model/` and set its `"model"` field to
`"parser.uint8.onnx"`.

The site loads these files from
[shonenash/electra-small-tagger-parser](https://huggingface.co/shonenash/electra-small-tagger-parser),
pinned to a commit in `src/pipeline.js`. To try a local export first, run
`VITE_MODEL_URL=model/ bun run dev`. To publish it (needs an HF token with write
access in `.env`):

```sh
uv run --env-file .env hf upload shonenash/electra-small-tagger-parser public/model . \
  --commit-message "Update model"
```

Then set the commit hash the upload prints in `MODEL_URL` in `src/pipeline.js`.

Generated from spaCy, rerun after upgrading it:

```sh
uv run python util/export_tokenizer_exceptions.py   # -> src/tokenizer/exceptions.js
uv run python util/export_attribute_rules.py        # -> src/parser/attributeRules.js
```

## Performance

Sentences are parsed in parallel; the Workers dropdown picks 1-4 workers (default
`min(4, cores - 1)`, remembered in the browser). Each worker holds its own model copy.
On a 127-sentence text (3,231 words):

| Workers | Time | Memory |
| --- | --- | --- |
| 1 | 3.9 s | ~220 MB |
| 2 | 2.2 s | ~320 MB |
| 4 | 1.2 s | ~530 MB |

The shipped parser is quantized to uint8: 1.5x faster than fp16 for about -0.1 UAS/LAS
points. First load downloads about 46 MB.
