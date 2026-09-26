// @ts-check
// Biber feature blocks, ported from PyBiber's parse_functions.py. Each
// blockXxx() mirrors the Python _block_xxx() of the same name and returns
// feature name -> count per document. Conditions are kept exactly as in
// PyBiber, including its quirks, so the counts can be compared one to one.
import { FEATURES, WORDLISTS } from "./patterns.js";

/** @typedef {import("./biber.js").TokenTable} TokenTable */
/** @typedef {Record<string, Uint32Array>} Counts */

/**
 * PyBiber's lag columns: <col>_lag_k is the value k tokens back (negative k
 * means ahead), within the same document. Past a document edge PyBiber fills
 * dep and lemma with "punct" and tag with "PUNCT", and leaves pos and token
 * null. A null never matches in a Polars filter, not even with "!=";
 * undefined plays that role here.
 * @param {TokenTable} table
 */
function lags(table) {
    /**
     * @param {string[]} column
     * @param {string | undefined} fill
     * @returns {(i: number, k: number) => string | undefined}
     */
    const make = (column, fill) => (i, k) => {
        const j = i - k;
        return j >= 0 && j < table.length && table.doc[j] === table.doc[i] ? column[j] : fill;
    };
    return {
        dep: make(table.dep, "punct"),
        lem: make(table.lemma, "punct"),
        pos: make(table.pos, undefined),
        tag: make(table.tag, "PUNCT"),
        tok: make(table.text, undefined),
    };
}

/** @param {TokenTable} table @param {string[]} names @returns {Counts} */
function zeroCounts(table, names) {
    /** @type {Counts} */
    const counts = {};
    for (const name of names) counts[name] = new Uint32Array(table.docCount);
    return counts;
}

/** @param {Uint32Array | undefined} column @param {number} d */
function bump(column, d) {
    if (column) column[d] = (column[d] ?? 0) + 1;
}

// Polars str.contains patterns used by several blocks, unchanged.
const NOUN_NUM_DET_TAG = /^N|^CD|DT/;
const WH_TAG = /^W/;
// Rust regex [[:punct:]]: ASCII punctuation only.
const ONE_PUNCT_CHAR = /^[!-/:-@[-`{-~]$/;

const DEMONSTRATIVES = new Set(WORDLISTS.pronoun_matchlist);
const LINKING_VERBS = new Set(WORDLISTS.linking_matchlist);
const THAT_DELETION_VERBS = new Set(WORDLISTS.verb_matchlist);
const NOMINALIZATION_STOPLIST = new Set(WORDLISTS.nominalization_stoplist);
const GERUND_STOPLIST = new Set(WORDLISTS.gerund_stoplist);

// One regex per feature: its patterns joined with "|", as PyBiber's
// FEATURE_PATTERNS does. Compiled once at module load.
/** @type {[string, RegExp][]} */
const FEATURE_REGEXES = Object.entries(FEATURES).map(
    ([name, patterns]) => [name, new RegExp(patterns.join("|"), "g")],
);

/**
 * PyBiber's per-document string: "token_tag" pairs, lowercased and joined
 * with spaces. Punctuation becomes "_punct_"; "&" used as a conjunction
 * becomes "and".
 * @param {TokenTable} t
 * @param {number} start
 * @param {number} end
 * @returns {string}
 */
function docString(t, start, end) {
    const parts = new Array(end - start);
    for (let i = start; i < end; i++) {
        let token = t.text[i] ?? "";
        let tag = t.tag[i] ?? "";
        if (t.dep[i] === "punct") {
            token = "_punct";
            tag = "";
        }
        if (token === "&" && tag === "CC") token = "and";
        parts[i - start] = `${token.toLowerCase()}_${tag.toLowerCase()}`;
    }
    return parts.join(" ");
}

/**
 * The 31 regex-based features in patterns.js (f_01, f_03, f_04, ...).
 * @param {TokenTable} table
 * @returns {Counts}
 */
export function blockRegexFeatures(table) {
    const counts = zeroCounts(table, FEATURE_REGEXES.map(([name]) => name));
    for (let d = 0; d < table.docCount; d++) {
        const s = docString(table, table.docStart[d] ?? 0, table.docStart[d + 1] ?? 0);
        for (const [name, re] of FEATURE_REGEXES) {
            // match() with the g flag returns every match and ignores
            // lastIndex, so reusing the same regex across documents is safe.
            const column = counts[name];
            if (column) column[d] = s.match(re)?.length ?? 0;
        }
    }
    return counts;
}

/**
 * f_02_perfect_aspect, f_12_proverb_do
 * @param {TokenTable} table
 * @returns {Counts}
 */
export function blockAuxTense(table) {
    const c = zeroCounts(table, ["f_02_perfect_aspect", "f_12_proverb_do"]);
    for (let i = 0; i < table.length; i++) {
        const d = table.doc[i] ?? 0;
        const isAux = (table.dep[i] ?? "").includes("aux");
        if (table.lemma[i] === "have" && isAux) bump(c.f_02_perfect_aspect, d);
        if (table.lemma[i] === "do" && !isAux) bump(c.f_12_proverb_do, d);
    }
    return c;
}

// PyBiber's gerund pattern really is "nsub", not "nsubj".
const CORE_ARGUMENT = /nsubj|dobj|pobj/;
const GERUND_ARGUMENT = /nsub|dobj|pobj/;
const NOMINALIZATION_SUFFIX = /tion$|tions$|ment$|ments$|ness$|nesses$|ity$|ities$/;
const GERUND_SUFFIX = /ing$|ings$/;

/**
 * f_10_demonstrative_pronoun, f_14_nominalizations, f_15_gerunds,
 * f_16_other_nouns (nouns minus nominalizations and noun gerunds),
 * f_51_demonstratives (demonstratives used as determiners)
 * @param {TokenTable} table
 * @returns {Counts}
 */
export function blockLexicalMembership(table) {
    const demPronouns = new Uint32Array(table.docCount);
    const nominalizations = new Uint32Array(table.docCount);
    const gerunds = new Uint32Array(table.docCount);
    const otherNouns = new Uint32Array(table.docCount);
    const demDeterminers = new Uint32Array(table.docCount);

    for (let d = 0; d < table.docCount; d++) {
        const start = table.docStart[d] ?? 0;
        const end = table.docStart[d + 1] ?? start;
        let nDemPronoun = 0, nNominal = 0, nGerund = 0, nGerundNoun = 0;
        let nNoun = 0, nDemDeterminer = 0;

        for (let i = start; i < end; i++) {
            const lower = table.lower[i] ?? "";
            const tag = table.tag[i] ?? "";
            const dep = table.dep[i] ?? "";
            const pos = table.pos[i];
            const isNoun = pos === "NOUN";

            // A demonstrative standing alone as subject or object ("I like
            // that"), unless the previous token in the document is a noun,
            // number or determiner. This block does its own unfilled shift.
            if (tag === "DT" && CORE_ARGUMENT.test(dep) && DEMONSTRATIVES.has(lower)) {
                const prevTag = i > start ? table.tag[i - 1] ?? "" : null;
                if (prevTag === null || !NOUN_NUM_DET_TAG.test(prevTag)) nDemPronoun++;
            }

            if (isNoun && NOMINALIZATION_SUFFIX.test(lower) && !NOMINALIZATION_STOPLIST.has(lower)) {
                nNominal++;
            }

            if (GERUND_SUFFIX.test(lower) && GERUND_ARGUMENT.test(dep) && !GERUND_STOPLIST.has(lower)) {
                nGerund++;
                if (isNoun) nGerundNoun++;
            }

            if ((isNoun || pos === "PROPN") && !(table.text[i] ?? "").includes("-")) nNoun++;

            if (dep === "det" && DEMONSTRATIVES.has(lower)) nDemDeterminer++;
        }

        demPronouns[d] = nDemPronoun;
        nominalizations[d] = nNominal;
        gerunds[d] = nGerund;
        // Nominalizations and noun gerunds are counted by their own features.
        otherNouns[d] = Math.max(nNoun - nGerundNoun - nNominal, 0);
        demDeterminers[d] = nDemDeterminer;
    }

    return {
        f_10_demonstrative_pronoun: demPronouns,
        f_14_nominalizations: nominalizations,
        f_15_gerunds: gerunds,
        f_16_other_nouns: otherNouns,
        f_51_demonstratives: demDeterminers,
    };
}

/**
 * f_39_prepositions, f_40_adj_attr, f_41_adj_pred, f_61_stranded_preposition
 * @param {TokenTable} table
 * @returns {Counts}
 */
export function blockAdjAndPrepositions(table) {
    const c = zeroCounts(table, [
        "f_39_prepositions", "f_40_adj_attr", "f_41_adj_pred", "f_61_stranded_preposition",
    ]);
    const L = lags(table);
    for (let i = 0; i < table.length; i++) {
        const d = table.doc[i] ?? 0;
        const dep = table.dep[i];
        const hyphenated = (table.text[i] ?? "").includes("-");

        if (dep === "prep") bump(c.f_39_prepositions, d);

        if (table.pos[i] === "ADJ" && !hyphenated) {
            const next = L.pos(i, -1);
            // Attributive: followed by a noun or adjective, or by ", ADJ".
            if (next === "NOUN" || next === "ADJ" || (L.tok(i, -1) === "," && L.pos(i, -2) === "ADJ")) {
                bump(c.f_40_adj_attr, d);
            }
            // Predicative: after a linking verb and not followed by a noun,
            // adjective or adverb. A missing next token never matches.
            const prev = L.pos(i, 1);
            if ((prev === "VERB" || prev === "AUX")
                && LINKING_VERBS.has(L.lem(i, 1) ?? "")
                && next !== undefined && next !== "NOUN" && next !== "ADJ" && next !== "ADV") {
                bump(c.f_41_adj_pred, d);
            }
        }

        // A preposition directly followed by a single punctuation tag
        // ("the man I spoke to."). At a document end the tag fill is "PUNCT",
        // which is not a single character.
        if (table.tag[i] === "IN" && dep === "prep" && ONE_PUNCT_CHAR.test(L.tag(i, -1) ?? "")) {
            bump(c.f_61_stranded_preposition, d);
        }
    }
    return c;
}

const OTHER_ADV_SUB_EXCLUDED = new Set(["because", "if", "unless", "though", "although", "tho"]);

/**
 * f_21_that_verb_comp, f_22_that_adj_comp, f_23_wh_clause, f_29_that_subj,
 * f_30_that_obj, f_31_wh_subj, f_32_wh_obj, f_34_sentence_relatives,
 * f_35_because, f_38_other_adv_sub, f_60_that_deletion
 * @param {TokenTable} table
 * @returns {Counts}
 */
export function blockClauseEmbedding(table) {
    const c = zeroCounts(table, [
        "f_21_that_verb_comp", "f_22_that_adj_comp", "f_23_wh_clause", "f_29_that_subj",
        "f_30_that_obj", "f_31_wh_subj", "f_32_wh_obj", "f_34_sentence_relatives",
        "f_35_because", "f_38_other_adv_sub", "f_60_that_deletion",
    ]);
    const L = lags(table);
    for (let i = 0; i < table.length; i++) {
        const d = table.doc[i] ?? 0;
        const token = table.text[i] ?? "";
        const lower = table.lower[i] ?? "";
        const tag = table.tag[i] ?? "";
        const pos = table.pos[i];
        const dep = table.dep[i] ?? "";
        const prevPos = L.pos(i, 1);
        const isWh = WH_TAG.test(tag);

        // f_21 and f_22 compare the token case-sensitively, as PyBiber does.
        if (token === "that" && pos === "SCONJ") {
            if (prevPos === "VERB") bump(c.f_21_that_verb_comp, d);
            if (prevPos === "ADJ") bump(c.f_22_that_adj_comp, d);
        }

        if (isWh && token !== "which" && prevPos === "VERB") bump(c.f_23_wh_clause, d);

        if (lower === "that" && NOUN_NUM_DET_TAG.test(L.tag(i, 1) ?? "")) {
            if (dep.includes("nsubj")) bump(c.f_29_that_subj, d);
            if (dep.includes("dobj")) bump(c.f_30_that_obj, d);
        }

        // WH relative: after a noun, number or determiner, or "who" after
        // punctuation that follows one; not after "ask"/"tell".
        if (isWh && token !== "that" && L.lem(i, 2) !== "ask" && L.lem(i, 2) !== "tell"
            && (NOUN_NUM_DET_TAG.test(L.tag(i, 1) ?? "")
                || (prevPos === "PUNCT" && NOUN_NUM_DET_TAG.test(L.tag(i, 2) ?? "") && token === "who"))) {
            if (dep.includes("nsubj")) bump(c.f_31_wh_subj, d);
            if (dep.includes("obj")) bump(c.f_32_wh_obj, d);
        }

        if (lower === "which" && prevPos === "PUNCT") bump(c.f_34_sentence_relatives, d);

        // A missing next token never matches, so a document-final "because"
        // is not counted.
        const nextTok = L.tok(i, -1);
        if (lower === "because" && nextTok !== undefined && nextTok.toLowerCase() !== "of") {
            bump(c.f_35_because, d);
        }

        // dep_lag_1 is a dependency label, so != "ADV" always holds and every
        // "that" is excluded. Kept as in PyBiber.
        if (pos === "SCONJ" && dep === "mark" && !OTHER_ADV_SUB_EXCLUDED.has(lower)
            && !(lower === "that" && L.dep(i, 1) !== "ADV")) {
            bump(c.f_38_other_adv_sub, d);
        }

        if (pos === "VERB" && THAT_DELETION_VERBS.has(table.lemma[i] ?? "")
            && ((L.dep(i, -1) === "nsubj" && L.pos(i, -2) === "VERB"
                 && L.tag(i, -1) !== "WP" && L.tag(i, -2) !== "VBG")
                || (L.tag(i, -1) === "DT" && L.dep(i, -2) === "nsubj" && L.pos(i, -3) === "VERB")
                || (L.tag(i, -1) === "DT" && L.dep(i, -2) === "amod"
                    && L.dep(i, -3) === "nsubj" && L.pos(i, -4) === "VERB"))) {
            bump(c.f_60_that_deletion, d);
        }
    }

    // WH clauses that are WH relatives are counted by f_31/f_32 instead.
    const whClause = c.f_23_wh_clause, whSubj = c.f_31_wh_subj, whObj = c.f_32_wh_obj;
    if (whClause && whSubj && whObj) {
        for (let d = 0; d < table.docCount; d++) {
            whClause[d] = Math.max((whClause[d] ?? 0) - (whSubj[d] ?? 0) - (whObj[d] ?? 0), 0);
        }
    }
    return c;
}

/**
 * f_13_wh_question: a WH word (not a determiner) followed by an auxiliary, at
 * the start of a sentence or right after punctuation.
 * @param {TokenTable} table
 * @returns {Counts}
 */
export function blockSentenceLevel(table) {
    const c = zeroCounts(table, ["f_13_wh_question"]);
    const L = lags(table);
    let indexInSentence = 0;
    for (let i = 0; i < table.length; i++) {
        indexInSentence = i > 0 && table.sent[i - 1] === table.sent[i] ? indexInSentence + 1 : 0;
        if (WH_TAG.test(table.tag[i] ?? "") && table.pos[i] !== "DET" && L.dep(i, -1) === "aux"
            && (L.pos(i, 1) === "PUNCT" || L.pos(i, 2) === "PUNCT" || indexInSentence <= 1)) {
            bump(c.f_13_wh_question, table.doc[i] ?? 0);
        }
    }
    return c;
}

const ALPHABETIC = /^[a-z]+$/;

/**
 * f_43_type_token (TTR, or moving-average TTR over `window` tokens) and
 * f_44_mean_word_length, both over alphabetic tokens only. Types for f_43 are
 * "token_tag" pairs. A document with no alphabetic tokens (or, for MATTR,
 * fewer than `window`) gets 0.
 * @param {TokenTable} table
 * @param {boolean} useTtr
 * @param {number} window
 * @returns {Record<string, Float64Array>}
 */
export function blockDerivedMetrics(table, useTtr, window) {
    const typeToken = new Float64Array(table.docCount);
    const meanLength = new Float64Array(table.docCount);
    for (let d = 0; d < table.docCount; d++) {
        /** @type {string[]} */
        const types = [];
        let chars = 0;
        for (let i = table.docStart[d] ?? 0; i < (table.docStart[d + 1] ?? 0); i++) {
            const lower = table.lower[i] ?? "";
            if (!ALPHABETIC.test(lower)) continue;
            types.push(`${lower}_${(table.tag[i] ?? "").toLowerCase()}`);
            chars += lower.length;
        }
        if (types.length === 0) continue;
        meanLength[d] = chars / types.length;
        typeToken[d] = useTtr ? new Set(types).size / types.length : mattr(types, window);
    }
    return { f_43_type_token: typeToken, f_44_mean_word_length: meanLength };
}

/**
 * Mean over every full window of `window` consecutive items of
 * (distinct items / window). 0 if there is no full window.
 * @param {string[]} items
 * @param {number} window
 */
function mattr(items, window) {
    if (items.length < window) return 0;
    /** @type {Map<string, number>} */
    const inWindow = new Map();
    let sum = 0;
    for (let i = 0; i < items.length; i++) {
        const add = items[i] ?? "";
        inWindow.set(add, (inWindow.get(add) ?? 0) + 1);
        if (i >= window) {
            const drop = items[i - window] ?? "";
            const left = (inWindow.get(drop) ?? 0) - 1;
            if (left === 0) inWindow.delete(drop);
            else inWindow.set(drop, left);
        }
        if (i >= window - 1) sum += inWindow.size / window;
    }
    return sum / (items.length - window + 1);
}

const FINITE_BE_TAGS = new Set(["VBD", "VBP", "VBZ"]);

/**
 * f_17_agentless_passives, f_18_by_passives, f_19_be_main_verb
 * @param {TokenTable} table
 * @param {boolean} strictBeMainVerb count finite "be" only as sentence root
 * @returns {Counts}
 */
export function blockPassiveVoice(table, strictBeMainVerb) {
    const c = zeroCounts(table, ["f_17_agentless_passives", "f_18_by_passives", "f_19_be_main_verb"]);
    const L = lags(table);
    for (let i = 0; i < table.length; i++) {
        const d = table.doc[i] ?? 0;
        const dep = table.dep[i] ?? "";
        if (dep === "auxpass") {
            // "by" two or three tokens after the passive auxiliary: "was eaten by".
            const byAgent = L.tok(i, -2) === "by" || L.tok(i, -3) === "by";
            bump(byAgent ? c.f_18_by_passives : c.f_17_agentless_passives, d);
        }
        if (table.lemma[i] === "be" && FINITE_BE_TAGS.has(table.tag[i] ?? "") && !dep.includes("aux")
            && (!strictBeMainVerb || dep === "ROOT")) {
            bump(c.f_19_be_main_verb, d);
        }
    }
    return c;
}

/**
 * f_25_present_participle, f_26_past_participle, f_27_past_participle_whiz,
 * f_28_present_participle_whiz
 * @param {TokenTable} table
 * @returns {Counts}
 */
export function blockParticipialClauses(table) {
    const c = zeroCounts(table, [
        "f_25_present_participle", "f_26_past_participle",
        "f_27_past_participle_whiz", "f_28_present_participle_whiz",
    ]);
    const L = lags(table);
    for (let i = 0; i < table.length; i++) {
        const d = table.doc[i] ?? 0;
        const tag = table.tag[i];
        const dep = table.dep[i];
        if (tag !== "VBG" && tag !== "VBN") continue;
        const present = tag === "VBG";

        // Participial clause right after punctuation ("..., walking home").
        // dep_lag_1 is filled with "punct", so a document-initial one counts.
        if ((dep === "advcl" || dep === "ccomp") && L.dep(i, 1) === "punct") {
            bump(present ? c.f_25_present_participle : c.f_26_past_participle, d);
        }
        // Reduced relative right after a noun ("the man sitting there").
        if (dep === "acl" && L.pos(i, 1) === "NOUN") {
            bump(present ? c.f_28_present_participle_whiz : c.f_27_past_participle_whiz, d);
        }
    }
    return c;
}

/**
 * f_62_split_infinitive ("to really go"), f_63_split_auxiliary ("will
 * really go")
 * @param {TokenTable} table
 * @returns {Counts}
 */
export function blockSplitConstructions(table) {
    const c = zeroCounts(table, ["f_62_split_infinitive", "f_63_split_auxiliary"]);
    const L = lags(table);
    for (let i = 0; i < table.length; i++) {
        const d = table.doc[i] ?? 0;
        if (table.tag[i] === "TO" && L.tag(i, -1) === "RB"
            && (L.tag(i, -2) === "VB" || (L.tag(i, -2) === "RB" && L.tag(i, -3) === "VB"))) {
            bump(c.f_62_split_infinitive, d);
        }
        if ((table.dep[i] ?? "").includes("aux") && L.pos(i, -1) === "ADV"
            && (L.pos(i, -2) === "VERB" || (L.pos(i, -2) === "ADV" && L.pos(i, -3) === "VERB"))) {
            bump(c.f_63_split_auxiliary, d);
        }
    }
    return c;
}

const COORDINATED_POS = new Set(["NOUN", "VERB", "ADJ", "ADV"]);

/**
 * f_64_phrasal_coordination, f_65_clausal_coordination
 * @param {TokenTable} table
 * @returns {Counts}
 */
export function blockCoordination(table) {
    const c = zeroCounts(table, ["f_64_phrasal_coordination", "f_65_clausal_coordination"]);
    const L = lags(table);
    for (let i = 0; i < table.length; i++) {
        if (table.tag[i] !== "CC") continue;
        const d = table.doc[i] ?? 0;
        // Same part of speech on both sides: "cats and dogs".
        const next = L.pos(i, -1);
        if (next !== undefined && COORDINATED_POS.has(next) && L.pos(i, 1) === next) {
            bump(c.f_64_phrasal_coordination, d);
        }
        // A subject within the next three tokens: "..., and she left".
        if (table.dep[i] !== "ROOT"
            && (L.dep(i, -1) === "nsubj" || L.dep(i, -2) === "nsubj" || L.dep(i, -3) === "nsubj")) {
            bump(c.f_65_clausal_coordination, d);
        }
    }
    return c;
}
