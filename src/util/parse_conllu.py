import sys
import numpy as np
from conllu import parse
import logging 

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_conllu(path):
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()

    lines = parse(data)

    dataset = []
    for line in lines:
        token_pairs = [
            (token["form"].lower(), token["xpos"])
            for token in line
            if isinstance(token["id"], int)
        ]
        if token_pairs:
            dataset.append(token_pairs)
    
    return dataset

def build_vocab_maps(data):
    words = set()
    tags = set()

    for sent in data:
        for word, tag in sent:
            words.add(word)
            tags.add(tag)

    # Set 0 for <PAD> and 1 for <UNK>
    word_id_map = {w: i+2 for i, w in enumerate(sorted(list(words)))}
    word_id_map["<PAD>"] = 0
    word_id_map["<UNK>"] = 1
  
    tag_id_map = {t: i for i, t in enumerate(sorted(list(tags)))}  

    return word_id_map, tag_id_map


# only when ran as an independent script
def main(argv: list):
    if len(argv) < 2:
        logging.info("please pass a path to a conllu file")
        logging.info("parse_conllu.py <PATH>")
    else:
        dataset  = parse_conllu(argv[1])
        words, tags = build_vocab_maps(dataset)

        logging.info("Stats............")
        logging.info(f"Sentences:\t{len(dataset)}")
        logging.info(f"Tokens:\t{len(words.keys())}")
        logging.info(f"Tags:\t {len(tags.keys())}")

if __name__ == "__main__":
    main(sys.argv)
