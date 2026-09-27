// The dimension profile as a Chart.js radar chart: one spoke per Biber
// dimension. The center of a spoke is the far end of the dimension's negative
// pole, the tip the far end of its positive pole, and the gray ring halfway
// out is the average text in Biber's corpus. Main thread only.
import {
  Chart, Filler, LineElement, PointElement, RadarController, RadialLinearScale, Tooltip,
} from "chart.js";
import { DIMENSIONS, MIN_WORDS, RELIABLE_WORDS } from "./features/dimensions.js";

Chart.register(RadarController, RadialLinearScale, PointElement, LineElement, Filler, Tooltip);

// Score at the edge of each spoke (and, negated, at its center), about as far
// as Biber's text types spread on each dimension. Dimension 1 spreads much
// wider than the others, so a shared scale would flatten dimensions 2-6.
// Chart.js has one scale for all spokes, so scores are mapped to 0..1 (0.5 is
// the average) and scores beyond the edge are drawn at the edge.
const RANGE = [45, 8, 13, 6, 10, 6];
// Scores within this fraction of the range count as "near average".
const NEAR_AVERAGE = 0.1;

/** @type {Chart | null} */
let chart = null;

/** @param {number} score @param {number} k */
function toScale(score, k) {
  return 0.5 + 0.5 * Math.max(-1, Math.min(1, score / RANGE[k]));
}

/** "f_17_agentless_passives" -> "agentless passives" */
function featureLabel(name) {
  return name.replace(/^f_\d+_/, "").replaceAll("_", " ");
}

/** @param {number} x */
function signed(x) {
  return `${x < 0 ? "−" : "+"}${Math.abs(x).toFixed(1)}`;
}

/** Which way a score leans, in words. */
function leaning(dim, k, score) {
  if (Math.abs(score) < NEAR_AVERAGE * RANGE[k]) return "Near the average";
  if (score > 0) return `Leans ${dim.positive.toLowerCase()}`;
  return dim.negative ? `Leans ${dim.negative.toLowerCase()}` : `Less ${dim.positive.toLowerCase()}`;
}

/** The page's colors and font, from the CSS custom properties in index.html. */
function theme() {
  const css = getComputedStyle(document.documentElement);
  const v = (name) => css.getPropertyValue(name).trim();
  return {
    bg: v("--bg"), text: v("--text"), muted: v("--muted"), border: v("--border"),
    accent: v("--accent"), font: v("--font-ui"),
  };
}

/**
 * @param {HTMLCanvasElement} canvas
 * @param {import("./features/dimensions.js").DimensionProfile | null} profile
 *   null draws the spokes and labels only
 * @param {boolean} faint
 */
function drawChart(canvas, profile, faint) {
  const t = theme();
  const scores = profile?.dimensions ?? [];
  const alpha = faint ? 0.55 : 1;
  // --accent is a #rrggbb hex color.
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(t.accent.slice(i, i + 2), 16));
  const accent = (a) => `rgba(${r}, ${g}, ${b}, ${a})`;

  return new Chart(canvas, {
    type: "radar",
    data: {
      labels: DIMENSIONS.map((d) => (d.negative ? [d.positive, `vs. ${d.negative}`] : d.positive)),
      datasets: profile ? [{
        data: scores.map((d, k) => toScale(d.score, k)),
        fill: true,
        backgroundColor: accent(0.1 * alpha),
        borderColor: accent(alpha),
        borderWidth: 2,
        pointBackgroundColor: accent(alpha),
        pointBorderColor: t.bg,
        pointBorderWidth: 2,
        pointRadius: 4.5,
        pointHoverRadius: 6,
        pointHitRadius: 13,
      }] : [],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      aspectRatio: 1.25,
      animation: { duration: 250 },
      font: { family: t.font },
      scales: {
        r: {
          min: 0,
          max: 1,
          ticks: {
            stepSize: 0.5,
            // Only the middle ring is labeled.
            callback: (value) => (value === 0.5 ? "average" : ""),
            color: t.muted,
            // Clears the spoke line behind the label.
            backdropColor: t.bg,
            backdropPadding: 2,
            font: { family: t.font, size: 10 },
          },
          grid: {
            color: (ctx) => (ctx.tick.value === 0.5 ? t.muted : t.border),
            lineWidth: (ctx) => (ctx.tick.value === 0.5 ? 1.5 : 1),
          },
          angleLines: { color: t.border },
          pointLabels: { color: t.text, font: { family: t.font, size: 12 } },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: t.bg,
          borderColor: t.border,
          borderWidth: 1,
          titleColor: t.text,
          bodyColor: t.muted,
          titleFont: { family: t.font, size: 13 },
          bodyFont: { family: t.font, size: 12 },
          padding: 8,
          displayColors: false,
          callbacks: {
            title: ([item]) => {
              const k = item.dataIndex;
              const { dimension, score } = scores[k];
              return `${signed(score)}  ${leaning(dimension, k, score)}`;
            },
            label: (item) => {
              const k = item.dataIndex;
              const { dimension, score, contributions } = scores[k];
              const top = [...contributions]
                .sort((a, b) => Math.abs(b.sign * b.z) - Math.abs(a.sign * a.z))
                .slice(0, 3)
                .map((c) => `${featureLabel(c.feature)} ${signed(c.sign * c.z)}`);
              return [
                `${dimension.id}. ${dimension.name}`,
                "Driven by:",
                ...top.map((line) => `  ${line}`),
                ...(Math.abs(score) > RANGE[k] ? ["Beyond the chart edge"] : []),
              ];
            },
          },
        },
      },
    },
  });
}

/** @param {import("./features/dimensions.js").DimensionProfile} profile */
function scoreTable(profile) {
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "Dimension scores";
  const table = document.createElement("table");
  const head = table.createTHead().insertRow();
  for (const [label, num] of [["Dimension", false], ["Score", true], ["", false]]) {
    const th = document.createElement("th");
    th.textContent = label;
    if (num) th.className = "num";
    head.append(th);
  }
  const body = table.createTBody();
  profile.dimensions.forEach(({ dimension, score }, k) => {
    const row = body.insertRow();
    row.insertCell().textContent = `${dimension.id}. ${dimension.name}`;
    const cell = row.insertCell();
    cell.className = "num";
    cell.textContent = signed(score);
    row.insertCell().textContent = leaning(dimension, k, score);
  });
  details.append(summary, table);
  return details;
}

/**
 * Renders the profile into root, or an empty chart (spokes and labels only)
 * when profile is null.
 * @param {HTMLElement} root
 * @param {import("./features/dimensions.js").DimensionProfile | null} profile
 */
export function renderProfile(root, profile) {
  chart?.destroy();
  chart = null;
  root.replaceChildren();

  const heading = document.createElement("div");
  heading.className = "profile-heading";
  const title = document.createElement("h2");
  const sub = document.createElement("p");
  sub.className = "note";
  const usable = profile && profile.confidence !== "insufficient" ? profile : null;
  if (usable) {
    title.textContent = `Closest to: ${usable.textTypes[0].name}`;
    sub.textContent = usable.confidence === "low"
      ? `${usable.words.toLocaleString()} words. Under ${RELIABLE_WORDS.toLocaleString()} words, treat this as a rough estimate.`
      : `${usable.words.toLocaleString()} words`;
  } else {
    title.textContent = "Style profile";
    sub.textContent = profile
      ? `${profile.words.toLocaleString()} words. Paste at least ${MIN_WORDS} words to see a profile.`
      : `Analyze a text of at least ${MIN_WORDS} words to see where it sits on Biber's six dimensions.`;
  }
  heading.append(title, sub);

  // Chart.js sizes the canvas from its parent, which must be positioned.
  const box = document.createElement("div");
  box.className = "radar";
  const canvas = document.createElement("canvas");
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", usable
    ? "Radar chart of the six Biber dimensions. The scores are in the table below."
    : "Empty radar chart of the six Biber dimensions");
  box.append(canvas);

  const legend = document.createElement("p");
  legend.className = "note";
  legend.textContent = "Gray ring: the average text in Biber's (1988) corpus. "
    + "Outside the ring leans toward a spoke's label, inside toward its opposite. "
    + "Neither end is better.";

  root.append(heading, box, legend);
  if (usable) root.append(scoreTable(usable));
  chart = drawChart(canvas, usable, usable?.confidence === "low");
}
