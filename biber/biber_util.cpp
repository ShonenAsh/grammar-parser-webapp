#include "biber.hpp"

// Mapping is based on the index of the fine tag labels (output of SpaCy's tagger module)
static const UPos tag_to_upos_map[] = {
    UPos::SYM,   // $
    UPos::PUNCT, // ''
    UPos::PUNCT, // ,
    UPos::PUNCT, // -LRB-
    UPos::PUNCT, // -RRB-
    UPos::PUNCT, // .
    UPos::PUNCT, // :
    UPos::X,     // ADD
    UPos::ADJ,   // AFX
    UPos::CCONJ, // CC
    UPos::NUM,   // CD
    UPos::DET,   // DT (Context-sensitive)
    UPos::PRON,  // EX
    UPos::X,     // FW
    UPos::PUNCT, // HYPH
    UPos::ADP,   // IN
    UPos::ADJ,   // JJ
    UPos::ADJ,   // JJR
    UPos::ADJ,   // JJS
    UPos::X,     // LS
    UPos::AUX,   // MD
    UPos::PUNCT, // NFP
    UPos::NOUN,  // NN
    UPos::PROPN, // NNP
    UPos::PROPN, // NNPS
    UPos::NOUN,  // NNS
    UPos::DET,   // PDT
    UPos::PART,  // POS
    UPos::PRON,  // PRP
    UPos::PRON,  // PRP$
    UPos::ADV,   // RB
    UPos::ADV,   // RBR
    UPos::ADV,   // RBS
    UPos::PART,  // RP
    UPos::SYM,   // SYM
    UPos::PART,  // TO
    UPos::INTJ,  // UH
    UPos::VERB,  // VB (Context-sensitive)
    UPos::VERB,  // VBD (Context-sensitive)
    UPos::VERB,  // VBG (Context-sensitive)
    UPos::VERB,  // VBN (Context-sensitive)
    UPos::VERB,  // VBP (Context-sensitive)
    UPos::VERB,  // VBZ (Context-sensitive)
    UPos::DET,   // WDT (Context-sensitive)
    UPos::PRON,  // WP
    UPos::PRON,  // WP$
    UPos::ADV,   // WRB
    UPos::X,     // XX
    UPos::SPACE, // _SP
    UPos::PUNCT  // ``
};

UPos resolve_upos(int32_t tag_idx, int32_t dep_ids) {
    PosTag fine_tag = static_cast<PosTag>(tag_idx);
    // Get coarse tag
    UPos pos = tag_to_upos_map[tag_idx];
    
    DepRel dep = static_cast<DepRel>(dep_ids);

    // Dependency parser based exception from spacy's Attribute Ruler "patterns"
    // Auxiliaries
    if (PosTagCategory::is_verb(fine_tag)) {
        if (DepRel::AUX == dep || DepRel::AUXPASS == dep) {
            return UPos::AUX;
        }
    }

    // Pronoun vs Determiners
    if (PosTag::DT == fine_tag || PosTag::WDT == fine_tag) {
        if (DepRel::DET != dep && DepRel::QUANTMOD != dep) {
            return UPos::PRON;
        }
    }

    return pos;
}
