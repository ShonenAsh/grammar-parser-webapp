// The pipeline runs in a Web Worker; see analysisClient.js.
import { MAX_POOL_SIZE, analyze, poolSize, ready, setPoolSize } from "./analysisClient.js";
import { renderProfile } from "./radar.js";

// Rates rather than counts: shown as-is, without a count column.
const RATE_FEATURES = new Set(["f_43_type_token", "f_44_mean_word_length"]);

function esc(s) {
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// "f_17_agentless_passives" -> ["f_17", "agentless passives"]
function splitName(name) {
  const m = name.match(/^(f_\d+)_(.*)$/);
  return m ? [m[1], m[2].replaceAll("_", " ")] : [name, ""];
}

function renderFeatures(features) {
  const rows = Object.entries(features).map(([name, { count, per1000 }]) => {
    const [code, label] = splitName(name);
    const rate = RATE_FEATURES.has(name);
    const value = rate ? per1000.toFixed(name === "f_43_type_token" ? 3 : 2) : per1000.toFixed(2);
    return `
      <tr class="${count === 0 ? "zero" : ""}">
        <td class="code">${esc(code)}</td>
        <td>${esc(label)}</td>
        <td class="num">${rate ? "" : count}</td>
        <td class="num">${value}</td>
      </tr>`;
  }).join("");
  return `
    <h2>Features</h2>
    <table>
      <thead><tr><th></th><th>Feature</th><th class="num">Count</th><th class="num">Per 1,000 tokens</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p class="note">Type-token ratio and mean word length are shown as values, not rates.</p>`;
}

function renderSentence(sent) {
  const { words, lemmas, pos, upos, heads, deps } = sent;
  const rows = words.map((word, i) => {
    const h = heads[i];
    return `
      <tr>
        <td class="num">${i + 1}</td>
        <td>${esc(word)}</td>
        <td>${esc(lemmas[i])}</td>
        <td>${esc(pos[i])}</td>
        <td>${esc(upos[i])}</td>
        <td class="num">${h}</td>
        <td>${h === 0 ? "ROOT" : esc(words[h - 1] ?? "?")}</td>
        <td>${esc(deps[i])}</td>
      </tr>`;
  }).join("");
  return `
    <details>
      <summary>${esc(words.join(" "))}</summary>
      <table class="parse">
        <thead><tr>
          <th class="num">ID</th><th>Word</th><th>Lemma</th><th>Tag</th><th>POS</th>
          <th class="num">Head</th><th>Head word</th><th>Dep</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </details>`;
}

const input   = document.getElementById("input");
const btn     = document.getElementById("parseBtn");
const status  = document.getElementById("status");
const workers = document.getElementById("workers");
const results = document.getElementById("results");
const profile = document.getElementById("profile");

async function run() {
  const text = input.value.trim();
  if (!text) return;
  setBusy(true);
  status.textContent = "analyzing…";
  results.innerHTML = "";
  // The previous profile stays visible, dimmed, until the new one is ready.
  profile.style.opacity = "0.5";

  try {
    const t0 = performance.now();
    const data = await analyze(text, (done, total) => {
      status.textContent = `analyzing sentence ${done} of ${total}…`;
    });
    const ms = (performance.now() - t0).toFixed(0);

    renderProfile(profile, data.dimensions);
    if (!data.sentences.length) {
      results.innerHTML = `<p class="error">No analyzable sentences found.</p>`;
      status.textContent = "";
      return;
    }
    const n = data.sentences.length;
    const tokens = data.sentences.reduce((s, r) => s + r.words.length, 0);
    const skipped = data.skipped ? ` · ${data.skipped} skipped` : "";
    status.textContent = `${n} sentence${n > 1 ? "s" : ""} · ${tokens} tokens${skipped} · ${ms} ms`;
    results.innerHTML = renderFeatures(data.features)
      + `<h2>Sentences</h2>`
      + (data.skipped ? `<p class="note">${data.skipped} sentence${data.skipped > 1 ? "s were" : " was"} skipped (fewer than 2 or more than 100 words) and not counted.</p>` : "")
      + data.sentences.map(renderSentence).join("");
  } catch (e) {
    results.innerHTML = `<p class="error">${esc(String(e))}</p>`;
    status.textContent = "error";
    renderProfile(profile, null);
  } finally {
    profile.style.opacity = "";
    setBusy(false);
  }
}

// The pool can only be resized while no analysis is running.
function setBusy(busy) {
  btn.disabled = busy;
  workers.disabled = busy;
}

// Shows "loading model..." until every worker has the model. On failure the
// controls come back, so Analyze or another pool size retries the load.
function waitForModel(loaded) {
  setBusy(true);
  status.textContent = "loading model...";
  loaded.then(
    () => { status.textContent = "ready"; },
    (e) => {
      results.innerHTML = `<p class="error">Failed to load model: ${esc(String(e))}</p>`;
      status.textContent = "error";
    },
  ).finally(() => setBusy(false));
}

for (let n = 1; n <= MAX_POOL_SIZE; n++) {
  const option = new Option(String(n), String(n));
  workers.add(option);
}
workers.value = String(poolSize());
workers.addEventListener("change", () => waitForModel(setPoolSize(Number(workers.value))));

renderProfile(profile, null);
waitForModel(ready());

btn.addEventListener("click", run);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && !btn.disabled) run();
});
