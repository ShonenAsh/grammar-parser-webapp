import sys
import numpy as np
from conllu import parse

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

def main(argv: list):
    if len(argv) < 2:
        print("please pass a path to a conllu file")
        print("parse_conllu.py <PATH>")
    else:
        dataset = parse_conllu(argv[1])
        print("Stats............")
        print(len(dataset))


if __name__ == "__main__":
    main(sys.argv)
