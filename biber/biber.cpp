#include "biber.hpp"
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

struct BiberDoc {
  std::vector<char> text_pool;
  std::vector<uint32_t> offsets;
  std::vector<uint16_t> lengths;
  std::vector<uint64_t> hashes;   // spacy MurmurHash3
  std::vector<PosTag> tags;       // fine-grained POS tags
  std::vector<UPos> pos;          // Universal POS (coarse) tags
  std::vector<DepRel> dep_rels; // dependency relation to the head token
  std::vector<uint32_t> heads;    // the token head
  size_t token_count() const { return offsets.size(); }
};

float sentence_level_feats(const BiberDoc &doc) {
  float count{};
  for (size_t i{}; i < doc.token_count(); i++) {

    if (PosTagCategory::is_wh_word(doc.tags.at(i)) 
            && doc.pos.at(i) != UPos::DET) continue;

    if (i < doc.token_count() 
            && doc.dep_rels.at(i + 1) == DepRel::AUX) continue;
    
    bool sent_start = (i <= 1);
    if (!sent_start) { // not at the beginning of doc, check if start of sent.
        sent_start = (doc.pos.at(i-1) == UPos::PUNCT ||
                        doc.pos.at(i-2) == UPos::PUNCT);
    }

    if (sent_start) count++;
  }
  return count;
}

int main() { return 0; }
