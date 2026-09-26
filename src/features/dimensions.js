// @ts-check

// Scores a text on Biber's (1988) six dimensions the way MAT does (Nini 2019,
// Multidimensional Analysis Tagger): each feature is z-scored against Biber's
// 1988 corpus, and a dimension score is the sum of its positive features'
// z-scores minus the sum of its negative ones. A score of 0 means "like the
// average text in Biber's corpus"; neither pole is better than the other.

/**
 * Mean and standard deviation of each feature in Biber's corpus (Biber 1988:
 * 77), per 100 words, except f_43 (distinct words among the first 400, as a
 * percentage) and f_44 (mean word length in characters).
 * @type {Record<string, [number, number]>}
 */
export const BIBER_NORMS = {
    f_01_past_tense: [4.01, 3.04],
    f_02_perfect_aspect: [0.86, 0.52],
    f_03_present_tense: [7.77, 3.43],
    f_04_place_adverbials: [0.31, 0.34],
    f_05_time_adverbials: [0.52, 0.35],
    f_06_first_person_pronouns: [2.72, 2.61],
    f_07_second_person_pronouns: [0.99, 1.38],
    f_08_third_person_pronouns: [2.99, 2.25],
    f_09_pronoun_it: [1.03, 0.71],
    f_10_demonstrative_pronoun: [0.46, 0.48],
    f_11_indefinite_pronouns: [0.14, 0.20],
    f_12_proverb_do: [0.30, 0.35],
    f_13_wh_question: [0.02, 0.06],
    f_14_nominalizations: [1.99, 1.44],
    f_15_gerunds: [0.70, 0.38],
    f_16_other_nouns: [18.05, 3.56],
    f_17_agentless_passives: [0.96, 0.66],
    f_18_by_passives: [0.08, 0.13],
    f_19_be_main_verb: [2.83, 0.95],
    f_20_existential_there: [0.22, 0.18],
    f_21_that_verb_comp: [0.33, 0.29],
    f_22_that_adj_comp: [0.03, 0.06],
    f_23_wh_clause: [0.06, 0.10],
    f_24_infinitives: [1.49, 0.56],
    f_25_present_participle: [0.10, 0.17],
    f_26_past_participle: [0.01, 0.04],
    f_27_past_participle_whiz: [0.25, 0.31],
    f_28_present_participle_whiz: [0.16, 0.18],
    f_29_that_subj: [0.04, 0.08],
    f_30_that_obj: [0.08, 0.11],
    f_31_wh_subj: [0.21, 0.20],
    f_32_wh_obj: [0.14, 0.17],
    f_33_pied_piping: [0.07, 0.11],
    f_34_sentence_relatives: [0.01, 0.04],
    f_35_because: [0.11, 0.17],
    f_36_though: [0.05, 0.08],
    f_37_if: [0.25, 0.22],
    f_38_other_adv_sub: [0.10, 0.11],
    f_39_prepositions: [11.05, 2.54],
    f_40_adj_attr: [6.07, 1.88],
    f_41_adj_pred: [0.47, 0.26],
    f_42_adverbs: [6.56, 1.76],
    f_43_type_token: [51.1, 5.2],
    f_44_mean_word_length: [4.5, 0.4],
    f_45_conjuncts: [0.12, 0.16],
    f_46_downtoners: [0.20, 0.16],
    f_47_hedges: [0.06, 0.13],
    f_48_amplifiers: [0.27, 0.26],
    f_49_emphatics: [0.63, 0.42],
    f_50_discourse_particles: [0.12, 0.23],
    f_51_demonstratives: [0.99, 0.42],
    f_52_modal_possibility: [0.58, 0.35],
    f_53_modal_necessity: [0.21, 0.21],
    f_54_modal_predictive: [0.56, 0.42],
    f_55_verb_public: [0.77, 0.54],
    f_56_verb_private: [1.80, 1.04],
    f_57_verb_suasive: [0.29, 0.31],
    f_58_verb_seem: [0.08, 0.10],
    f_59_contractions: [1.35, 1.86],
    f_60_that_deletion: [0.31, 0.41],
    f_61_stranded_preposition: [0.20, 0.27],
    // Biber found no split infinitives, so this feature has no z-score.
    f_62_split_infinitive: [0, 0],
    f_63_split_auxiliary: [0.55, 0.25],
    f_64_phrasal_coordination: [0.34, 0.27],
    f_65_clausal_coordination: [0.45, 0.48],
    f_66_neg_synthetic: [0.17, 0.16],
    f_67_neg_analytic: [0.85, 0.61],
};

/**
 * @typedef {object} Dimension
 * @property {number} id
 * @property {string} name
 * @property {string} positive label of the positive pole
 * @property {string | null} negative label of the negative pole; null when
 *   Biber named only one pole (the low end is just "less" of it)
 * @property {string[]} plus features added to the score
 * @property {string[]} minus features subtracted from the score
 */

// Features with a salient loading (|loading| >= 0.35) in Biber's factor
// analysis, each counted only on the dimension where it loads most strongly.
// As in MAT, features rarer than 1 per 1,000 words in Biber's corpus (WH
// questions, WH clauses, sentence relatives, hedges, pied-piping, by-passives,
// past participial clauses, that adjective complements, that relatives in
// object position) are left out: a single occurrence would swing a short text.
/** @type {Dimension[]} */
export const DIMENSIONS = [
    {
        id: 1,
        name: "Involved vs. Informational Production",
        positive: "Involved",
        negative: "Informational",
        plus: [
            "f_56_verb_private", "f_60_that_deletion", "f_59_contractions", "f_03_present_tense",
            "f_07_second_person_pronouns", "f_12_proverb_do", "f_67_neg_analytic",
            "f_10_demonstrative_pronoun", "f_49_emphatics", "f_06_first_person_pronouns",
            "f_09_pronoun_it", "f_19_be_main_verb", "f_35_because", "f_50_discourse_particles",
            "f_11_indefinite_pronouns", "f_48_amplifiers", "f_52_modal_possibility",
            "f_65_clausal_coordination", "f_61_stranded_preposition",
        ],
        minus: [
            "f_16_other_nouns", "f_44_mean_word_length", "f_39_prepositions", "f_43_type_token",
            "f_40_adj_attr",
        ],
    },
    {
        id: 2,
        name: "Narrative vs. Non-narrative Concerns",
        positive: "Narrative",
        negative: "Non-narrative",
        plus: [
            "f_01_past_tense", "f_08_third_person_pronouns", "f_02_perfect_aspect",
            "f_55_verb_public", "f_66_neg_synthetic", "f_25_present_participle",
        ],
        minus: [],
    },
    {
        id: 3,
        name: "Explicit vs. Situation-dependent Reference",
        positive: "Explicit",
        negative: "Situation-dependent",
        plus: ["f_32_wh_obj", "f_31_wh_subj", "f_64_phrasal_coordination", "f_14_nominalizations"],
        minus: ["f_05_time_adverbials", "f_04_place_adverbials", "f_42_adverbs"],
    },
    {
        id: 4,
        name: "Overt Expression of Persuasion",
        positive: "Overtly persuasive",
        negative: null,
        plus: [
            "f_24_infinitives", "f_54_modal_predictive", "f_57_verb_suasive", "f_37_if",
            "f_53_modal_necessity", "f_63_split_auxiliary",
        ],
        minus: [],
    },
    {
        id: 5,
        name: "Abstract vs. Non-abstract Information",
        positive: "Abstract",
        negative: "Non-abstract",
        plus: [
            "f_45_conjuncts", "f_17_agentless_passives", "f_27_past_participle_whiz",
            "f_38_other_adv_sub",
        ],
        minus: [],
    },
    {
        id: 6,
        name: "On-line Informational Elaboration",
        positive: "On-line elaboration",
        negative: null,
        plus: ["f_21_that_verb_comp", "f_51_demonstratives"],
        minus: [],
    },
];

/**
 * Biber's (1989) text types as mean scores on dimensions 1-5 (dimension 6
 * is not used), with the values MAT uses.
 * @type {{ name: string, center: number[] }[]}
 */
export const TEXT_TYPES = [
    { name: "Intimate interpersonal interaction", center: [45, -1, -6, 1, -4] },
    { name: "Informational interaction", center: [30, -1, -4, 1, -3] },
    { name: "Scientific exposition", center: [-15, -2.5, 4, -2, 9] },
    { name: "Learned exposition", center: [-20, -2, 5, -3, 2] },
    { name: "Imaginative narrative", center: [5, 7, -4, 1, -2] },
    { name: "General narrative exposition", center: [-10, 2, 0, -1, 0] },
    { name: "Situated reportage", center: [0, -3, -13, -4.5, -3] },
    { name: "Involved persuasion", center: [5, -2, 2, 4, -1] },
];

// Word filters as in biber.js. Defined here so the page can import this module
// (for the dimension labels) without pulling in the feature extractors.
const ALL_PUNCT = /^[!-/:-@[-`{-~]+$/;
const ALPHABETIC = /^[a-z]+$/;

// Biber's type-token ratio counts distinct words among the first TTR_WORDS;
// shorter texts get a neutral z-score of 0, as in MAT.
const TTR_WORDS = 400;
// Below MIN_WORDS the scores are mostly noise; below RELIABLE_WORDS a few
// rare features can still move them noticeably.
export const MIN_WORDS = 400;
export const RELIABLE_WORDS = 1000;

/**
 * @typedef {object} DimensionScore
 * @property {Dimension} dimension
 * @property {number} score
 * @property {{ feature: string, z: number, sign: 1 | -1 }[]} contributions
 *   each feature's signed share of the score is sign * z
 */

/**
 * @typedef {object} DimensionProfile
 * @property {number} words non-punctuation tokens
 * @property {"insufficient" | "low" | "ok"} confidence from the text length
 * @property {Record<string, number>} z z-score of every feature with a nonzero SD
 * @property {DimensionScore[]} dimensions
 * @property {{ name: string, distance: number }[]} textTypes nearest first
 */

/**
 * @param {{ words: string[] }[]} sentences analyzeSentence() results
 * @param {Record<string, { per1000: number }>} features extractFeatures() output
 * @returns {DimensionProfile | null} null for a text with no words
 */
export function dimensionProfile(sentences, features) {
    const words = sentences.flatMap((s) => s.words);
    const wordCount = words.filter((w) => !ALL_PUNCT.test(w)).length;
    if (wordCount === 0) return null;

    /** @type {Record<string, number>} */
    const z = {};
    for (const [name, [mean, sd]] of Object.entries(BIBER_NORMS)) {
        if (sd === 0) continue;
        let value;
        if (name === "f_43_type_token") {
            const lexical = words.map((w) => w.toLowerCase()).filter((w) => ALPHABETIC.test(w));
            if (lexical.length < TTR_WORDS) {
                z[name] = 0;
                continue;
            }
            value = (new Set(lexical.slice(0, TTR_WORDS)).size / TTR_WORDS) * 100;
        } else if (name === "f_44_mean_word_length") {
            value = features[name]?.per1000 ?? mean;
        } else {
            // extractFeatures() rates are per 1,000 tokens, Biber's per 100 words.
            value = (features[name]?.per1000 ?? 0) / 10;
        }
        z[name] = (value - mean) / sd;
    }

    const dimensions = DIMENSIONS.map((dimension) => {
        /** @type {DimensionScore["contributions"]} */
        const contributions = [
            ...dimension.plus.map((feature) => ({ feature, z: z[feature] ?? 0, sign: /** @type {1} */ (1) })),
            ...dimension.minus.map((feature) => ({ feature, z: z[feature] ?? 0, sign: /** @type {-1} */ (-1) })),
        ];
        const score = contributions.reduce((sum, c) => sum + c.sign * c.z, 0);
        return { dimension, score, contributions };
    });

    const point = dimensions.slice(0, 5).map((d) => d.score);
    const textTypes = TEXT_TYPES
        .map(({ name, center }) => ({
            name,
            distance: Math.hypot(...center.map((c, i) => (point[i] ?? 0) - c)),
        }))
        .sort((a, b) => a.distance - b.distance);

    const confidence = wordCount < MIN_WORDS ? "insufficient"
        : wordCount < RELIABLE_WORDS ? "low" : "ok";
    return { words: wordCount, confidence, z, dimensions, textTypes };
}
