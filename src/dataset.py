"""VQA dataset: images + questions.json per split."""
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def build_vocab(questions, max_vocab=5000):
    words = {}
    for q in questions:
        for w in q.lower().split():
            w = "".join(c for c in w if c.isalnum())
            if w:
                words[w] = words.get(w, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, _ in sorted(words.items(), key=lambda x: -x[1])[: max_vocab - 2]:
        vocab[w] = len(vocab)
    return vocab


def encode_question(question, vocab, max_len=16):
    ids = []
    for w in question.lower().split():
        w = "".join(c for c in w if c.isalnum())
        ids.append(vocab.get(w, 1))
    ids = ids[:max_len] + [0] * max(0, max_len - len(ids))
    return ids


class VQADataset(Dataset):
    def __init__(self, data_dir, split, vocab, answers):
        self.root = Path(data_dir)
        items = json.loads((self.root / split / "questions.json").read_text())
        self.items = items
        self.vocab = vocab
        self.answers = answers
        self.answer_to_idx = {a: i for i, a in enumerate(answers)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(self.root / item["image"]).convert("RGB")
        image = transform(image)
        q_ids = torch.tensor(encode_question(item["question"], self.vocab), dtype=torch.long)
        label = self.answer_to_idx[item["answer"]]
        return image, q_ids, label, item
