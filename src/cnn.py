import logging
import sys
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from util.parse_conllu import build_vocab_maps, parse_conllu

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
ONNX_EXPORT_PATH = "cnn_pos_tagger.onnx"

# Dataset class for English Dependency Treebank data
class POSDataset(Dataset):
    def __init__(self, data, word_id_map, tag_id_map):
        self.samples = []
        for sent in data:
            word_ids = torch.tensor(
                [word_id_map.get(w, word_id_map["<UNK>"]) for w, t in sent],
                dtype=torch.long,
            )
            tag_ids = torch.tensor([tag_id_map[t] for w, t in sent], dtype=torch.long)
            self.samples.append((word_ids, tag_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    words, tags = zip(*batch)
    words_padded = pad_sequence(words, batch_first=True, padding_value=0)
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=-1)
    return words_padded, tags_padded


class CNNPOSTagger(nn.Module):
    def __init__(
        self, vocab_size, num_tags, embed_dim=128, num_filters=64, kernel_size=3
    ):
        super(CNNPOSTagger, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Convolution layer
        self.conv = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=1,
        )

        # FC layer
        self.fc = nn.Linear(num_filters, num_tags)

    def forward(self, x):
        # in: (batch_size, seq_len)
        embedded = self.embedding(x)  # out: (batch, seq_len, embed_dim)

        embedded = embedded.permute(0, 2, 1)  # out: (batch, channels, length)

        convolved = F.relu(self.conv(embedded))  # out: (batch, num_filters, seq_len)
        convolved = convolved.permute(0, 2, 1)  # out: (batch, seq_len, num_filters)

        logits = self.fc(convolved)
        return logits

def cal_accuracy(preds, labels, ignore_index=-1):
    predicted_idx = preds.argmax(dim=-1)
    mask = (labels != ignore_index)

    correct = (predicted_idx == labels) & mask
    return correct.sum().item() / mask.sum().item()

def export_vocab_files(word_id_map, tag_id_map, vocab_path="vocab.json", tag_path="tag_to_id.json"):
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(dict(word_id_map), f, ensure_ascii=False, indent=2)

    with open(tag_path, "w", encoding="utf-8") as f:
        json.dump(dict(tag_id_map), f, ensure_ascii=False, indent=2)

    logging.info(f"Exported vocab files to {vocab_path}, {tag_path}")

def train(train_path, test_path):
    train_set = parse_conllu(train_path)
    test_set = parse_conllu(test_path)
    # vocab map is built from the train set
    tr_token_map, tr_tag_map = build_vocab_maps(train_set)
    
    export_vocab_files(tr_token_map, tr_tag_map)

    VOCAB_SIZE = len(tr_token_map)
    NUM_TAGS = len(tr_tag_map)

    model = CNNPOSTagger(VOCAB_SIZE, NUM_TAGS)
    # Test data needs id mappings defined by train set, <UNK> id for words not in train set
    train_dataset = POSDataset(train_set, tr_token_map, tr_tag_map)
    test_dataset = POSDataset(test_set, tr_token_map, tr_tag_map)
    
    dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)


    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for words, tags in dataloader:
            optimizer.zero_grad()
             
            # forward pass
            outputs = model(words)

            # change shape for CrossEntropyLoss, (batch, num_classes, seq_len)
            loss = loss_fn(outputs.transpose(1, 2), tags)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # validation
        model.eval()
        val_acc = 0
        with torch.no_grad():
            for words, tags in test_dataloader:
                outputs = model(words)
                val_acc += cal_accuracy(outputs, tags)
        avg_train_loss = train_loss / len(dataloader)
        avg_val_acc = val_acc / len(test_dataloader)

        print(f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | Val Acc: {avg_val_acc:.2%}") 
    
    return model

def export_trained_model(model, onnx_export_path=ONNX_EXPORT_PATH):
    model.eval() # ensure the model is in eval mode
    device = torch.device('cpu') # ensure model is on CPU for stable export
    model.to(device)

    # onnx traces the math operations applied to the input which are then exported
    dummy_input = torch.randint(0 , 100, (1, 5), dtype=torch.long).to(device)
    
    # define dynamic axes for variable sentence length
    dynamic_axes = {
        "input" : {0 : "batch_size", 1: "sequence_length"},
        "output": {0 : "batch_size", 1: "sequence_length"}
    }

    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_export_path,
        export_params=True,
        opset_version=15,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )
    logging.info(f"Model saved to {onnx_export_path}")

def main(argv: list):
    if len(argv) < 3:
        print("Usage: cnn.py <TRAIN_CONLLU_PATH> <TEST_CONLLU_PATH> <OPTIONAL: ONNX_EXPORT_PATH>")
        sys.exit(1)
    
    if len(argv) == 4:
        ONNX_EXPORT_PATH = argv[3]

    model = train(argv[1], argv[2])

    export_trained_model(model)
    

if __name__ == "__main__":
    main(sys.argv)
