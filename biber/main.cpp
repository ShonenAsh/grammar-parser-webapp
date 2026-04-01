#include "murmurhash/MurmurHash2.h"
#include <cstdint>
#include <iostream>
#include <string>

uint64_t get_spacy_hash(const std::string &str) {
  return MurmurHash64A(str.data(), str.size(), 1);
}

int main() {
  std::string test_str = "apple";
  uint64_t hash = get_spacy_hash(test_str);

  std::cout << "String: " << test_str << std::endl;
  std::cout << "Hash (Uint64): " << hash << std::endl;

  return 0;
}
