// @ts-check
// en_core_web_sm's attribute ruler: assigns spaCy's coarse POS (pos_) and
// morphology from fine tags, words and dependency labels, plus a few lemma
// overrides. Rules come from util/export_attribute_rules.py and are applied in
// order, so a later matching rule overrides an earlier one, as in spaCy.
import { ATTRIBUTE_RULES } from "./attributeRules.js";

/** @typedef {{ tag: string, lower: string, dep: string }} TokenAttrs */
/** @typedef {(t: TokenAttrs) => boolean} TokenTest */

/**
 * One spaCy Matcher value: an exact value, or {IN}, {NOT_IN}, {REGEX}.
 * @param {unknown} value
 * @returns {(actual: string) => boolean}
 */
function compileValue(value) {
    if (typeof value === "string") return (actual) => actual === value;
    const ops = /** @type {{ IN?: string[], NOT_IN?: string[], REGEX?: string }} */ (value);
    /** @type {((actual: string) => boolean)[]} */
    const checks = [];
    if (ops.IN) { const set = new Set(ops.IN); checks.push((a) => set.has(a)); }
    if (ops.NOT_IN) { const set = new Set(ops.NOT_IN); checks.push((a) => !set.has(a)); }
    if (ops.REGEX) { const re = new RegExp(ops.REGEX); checks.push((a) => re.test(a)); }
    return (actual) => checks.every((check) => check(actual));
}

/**
 * @param {Record<string, unknown>} spec token pattern, e.g. { TAG: "VBD", LOWER: "was" }
 * @returns {TokenTest}
 */
function compileSpec(spec) {
    /** @type {TokenTest[]} */
    const tests = [];
    for (const [key, value] of Object.entries(spec)) {
        if (key === "IS_SPACE") {
            // Whitespace tokens never reach the parser.
            tests.push(() => value === false);
            continue;
        }
        const matches = compileValue(value);
        if (key === "TAG") tests.push((t) => matches(t.tag));
        else if (key === "LOWER") tests.push((t) => matches(t.lower));
        else if (key === "DEP") tests.push((t) => matches(t.dep));
        else throw new Error(`unsupported attribute-ruler key ${key}`);
    }
    return (t) => tests.every((test) => test(t));
}

const RULES = ATTRIBUTE_RULES.map((rule) => ({
    patterns: rule.patterns.map((pattern) => pattern.map(compileSpec)),
    pos: /** @type {string | undefined} */ (rule.attrs.POS),
    lemma: /** @type {string | undefined} */ (rule.attrs.LEMMA),
    // "_" is spaCy's notation for an empty morphology in rules.
    morph: rule.attrs.MORPH === "_" ? "" : /** @type {string | undefined} */ (rule.attrs.MORPH),
    index: rule.index,
}));

/**
 * Coarse POS and morphology for every word ("" where no rule sets one), and
 * the lemma a rule assigns (undefined where none does).
 * @param {string[]} words
 * @param {string[]} tags Penn tags
 * @param {string[]} deps dependency labels
 * @returns {{ pos: string[], morphs: string[], lemmas: (string | undefined)[] }}
 */
export function applyAttributeRules(words, tags, deps) {
    const n = words.length;
    /** @type {TokenAttrs[]} */
    const tokens = words.map((w, i) => ({ tag: tags[i] ?? "", lower: w.toLowerCase(), dep: deps[i] ?? "" }));
    /** @type {string[]} */
    const pos = new Array(n).fill("");
    /** @type {string[]} */
    const morphs = new Array(n).fill("");
    /** @type {(string | undefined)[]} */
    const lemmas = new Array(n).fill(undefined);

    for (const rule of RULES) {
        for (const pattern of rule.patterns) {
            const len = pattern.length;
            for (let start = 0; start + len <= n; start++) {
                if (!pattern.every((test, k) => test(/** @type {TokenAttrs} */ (tokens[start + k])))) continue;
                // index says which token of the match gets the attributes;
                // negative values count from the end, as in spaCy.
                const target = start + ((rule.index % len) + len) % len;
                if (rule.pos !== undefined) pos[target] = rule.pos;
                if (rule.lemma !== undefined) lemmas[target] = rule.lemma;
                if (rule.morph !== undefined) morphs[target] = rule.morph;
            }
        }
    }
    return { pos, morphs, lemmas };
}
