// @ts-check
import {
    blockAdjAndPrepositions, blockAuxTense, blockClauseEmbedding, blockCoordination,
    blockDerivedMetrics, blockLexicalMembership, blockParticipialClauses,
    blockPassiveVoice, blockRegexFeatures, blockSentenceLevel, blockSplitConstructions,
} from "./parseFunctions.js";

// TokenTable is a datastructure to hold documents and their tokens for Biber analysis.
export class TokenTable {
    /** @param {number} n number of tokens */
    constructor(n) {
        this.length = n;
        // In the future maybe be able to support multiple docs?
        this.doc = new Uint32Array(n);
        this.sent = new Uint32Array(n); // uniqueid across docs
        /** @type {string[]} */ this.text = new Array(n);
        /** @type {string[]} */ this.lower = new Array(n);
        /** @type {string[]} */ this.lemma = new Array(n);
        /** @type {string[]} */ this.tag = new Array(n);
        /** @type {string[]} */ this.pos = new Array(n);
        /** @type {string[]} */ this.dep = new Array(n);
        // absolute index of head, ROOT = -1
        this.head = new Int32Array(n);

        this.childStart = new Uint32Array(n + 1);
        this.childIdx = new Uint32Array(n);

        /** @type {number[]} */ this.docStart = [0];
    }

    /** @param {{ sentences: { words: string[], lemmas: string[], pos: string[],
     * upos: string[], heads: number[], deps: string[] }[] }[]} docs
     *   analyze() sentences: pos holds Penn tags, upos spaCy's coarse POS
     */
    static buildTable(docs) {
        let n = 0;
        for (const d of docs) for (const s of d.sentences) n += s.words.length;
        const tt = new TokenTable(n);

        let i = 0, sentId = 0;
        docs.forEach((d, docId) => {
            for (const s of d.sentences) {
                const first = i;
                for (let w = 0; w < s.words.length; w++, i++) {
                    tt.doc[i] = docId;
                    tt.sent[i] = sentId;
                    tt.text[i] = s.words[w];
                    tt.lower[i] = s.words[w].toLowerCase();
                    tt.lemma[i] = s.lemmas[w];
                    tt.tag[i] = s.pos[w];
                    tt.dep[i] = s.deps[w];
                    tt.pos[i] = s.upos[w];
                    tt.head[i] = s.heads[w] === 0 ? -1 : first + s.heads[w] - 1;
                }
                sentId++;
            }
            tt.docStart.push(i);
        });

        tt.buildChildren();
        return tt;
    }

    buildChildren() {
        const { length: n, head, childStart, childIdx } = this;
        for (let i = 0; i < n; i++) {
            if (head[i] >= 0) childStart[head[i] + 1]++;
        }
        for (let i = 0; i < n; i++) {
            childStart[i + 1] += childStart[i];
        }
        const fill = childStart.slice(0, n);
        for (let i = 0; i < n; i++) {
            if (head[i] >= 0)
                childIdx[fill[head[i]]++] = i;
        }
    }

    get docCount() { return this.docStart.length - 1; }

    /** @param {number} i: index of current token
     *  @param {number} k: index i + k'th token or -1 if it crosses a sentence
     *      boundary.
     */
    at(i, k) {
        const j = i + k;
        return j >= 0 && j < this.length && this.sent[j] === this.sent[i] ? j : -1;
    }

    /** @param {number} i: token at position i
     * @returns {Uint32Array} indices of i's dependents.
     */
    children(i) {
        return this.childIdx.subarray(this.childStart[i], this.childStart[i + 1]);
    }

    /** @param {number} i: token at position i
     *  @param {string} dep: dependency label to look for
     *  @returns {boolean} true if children with dep label exist
     */
    hasChild(i, dep) {
        for (const c of this.children(i)) if (this.dep[c] === dep) return true;
        return false;
    }
    /** Debugging: util for printing a token's values
     *  @param {number}: token at index i 
     */
    row(i) {
        return {
            i, doc: this.doc[i], sent: this.sent[i], text: this.text[i], lemma: this.lemma[i],
            tag: this.tag[i], pos: this.pos[i], dep: this.dep[i], head: this.head[i],
        };
    }
}

const DEBUG = false;
const log = (/** @type {unknown[]} */ ...a) => DEBUG && console.log("[biber]", ...a);

// Rust regex [[:punct:]]: ASCII punctuation only.
const ALL_PUNCT = /^[!-/:-@[-`{-~]+$/;
const ALPHABETIC = /^[a-z]+$/;
// Rates, not counts, so they are not normalized per 1,000 tokens.
const NOT_NORMALIZED = new Set(["f_43_type_token", "f_44_mean_word_length"]);

/**
 * Extracts the 67 Biber features for every document in the table, like
 * PyBiber's biber().
 *
 * f_43_type_token is plain TTR when any document has 200 or fewer
 * alphabetic tokens (or forceTtr is set), otherwise moving-average TTR; the
 * MATTR window shrinks to the shortest document if that is shorter.
 *
 * @param {TokenTable} table
 * @param {object} [options]
 * @param {boolean} [options.normalize] counts per 1,000 non-punctuation tokens (default true)
 * @param {boolean} [options.forceTtr] always use plain TTR for f_43 (default false)
 * @param {number} [options.mattrWindow] MATTR window size (default 100)
 * @param {boolean} [options.strictBeMainVerb] f_19 counts finite "be" only as sentence root (default true)
 * @returns {Record<string, Float64Array>} feature name -> value per document, names sorted
 */
export function biber(table, options = {}) {
    const {
        normalize = true, forceTtr = false, mattrWindow = 100, strictBeMainVerb = true,
    } = options;
    if (mattrWindow < 1) throw new RangeError("mattrWindow must be >= 1");

    const docTotals = new Uint32Array(table.docCount);
    const alphabetic = new Uint32Array(table.docCount);
    for (let i = 0; i < table.length; i++) {
        const d = table.doc[i] ?? 0;
        if (!ALL_PUNCT.test(table.text[i] ?? "")) docTotals[d] = (docTotals[d] ?? 0) + 1;
        if (ALPHABETIC.test(table.lower[i] ?? "")) alphabetic[d] = (alphabetic[d] ?? 0) + 1;
    }

    const shortestDoc = table.docCount > 0 ? Math.min(...alphabetic) : 0;
    let useTtr = forceTtr || shortestDoc <= 200;
    let window = mattrWindow;
    if (!useTtr && shortestDoc < 1) {
        useTtr = true;
    } else if (!useTtr && shortestDoc < window) {
        log(`MATTR window ${window} exceeds the shortest document (${shortestDoc}); using ${shortestDoc}`);
        window = shortestDoc;
    }
    log(useTtr ? "f_43 uses TTR" : `f_43 uses MATTR, window ${window}`);

    /** @type {Record<string, Uint32Array | Float64Array>} */
    const all = {
        ...blockRegexFeatures(table),
        ...blockAuxTense(table),
        ...blockLexicalMembership(table),
        ...blockSentenceLevel(table),
        ...blockClauseEmbedding(table),
        ...blockAdjAndPrepositions(table),
        ...blockPassiveVoice(table, strictBeMainVerb),
        ...blockParticipialClauses(table),
        ...blockDerivedMetrics(table, useTtr, window),
        ...blockSplitConstructions(table),
        ...blockCoordination(table),
    };

    /** @type {Record<string, Float64Array>} */
    const out = {};
    for (const name of Object.keys(all).sort()) {
        const values = Float64Array.from(all[name] ?? []);
        if (normalize && !NOT_NORMALIZED.has(name)) {
            for (let d = 0; d < values.length; d++) {
                values[d] = ((values[d] ?? 0) / (docTotals[d] ?? 0)) * 1000;
            }
        }
        out[name] = values;
    }
    return out;
}
