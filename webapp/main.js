const worker = new Worker("worker.js");
const btn = document.getElementById("tag-btn");
const status = document.getElementById("status");
const resultsDiv = document.getElementById("results");

// Worker loading first
worker.postMessage({type: "LOAD"});

// Listen to messages from worker
worker.onmessage = (e) => {
    const { type, predictions, error} = e.data;

    if (type === "READY") {
        status.innerText = "Model Ready!";
        status.style.color = "green";
        btn.disabled = false;
    }

    if (type === "RESULT") {
        renderResults(predictions);
    }

    if (type === "ERROR") {
        alert("Error: " + error);
    }
};


btn.onclick = () => {
    const text = document.getElementById("input-text").value;
    if (!text) return;
    resultsDiv.innerHTML = "Processing ...";
    worker.postMessage({type: "INFER", text });
};

function renderResults(predictions) {
    resultsDiv.innerHTML = "";
    predictions.forEach(p => {
        const div = document.createElement("div");
        div.className = "token";
        div.innerHTML = `${p.word} <span class="tag">${p.tag}</span>`;
        resultsDiv.appendChild(div);
    });
}
