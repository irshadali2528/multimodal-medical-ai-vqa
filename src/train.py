"""Train the vision-language fusion model."""
import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import VQADataset, build_vocab
from model import VQAModel

ANSWERS = ["normal", "pneumonia", "fracture", "yes", "no"]


def load_split(data_dir, split, vocab):
    return VQADataset(data_dir, split, vocab, ANSWERS)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, q_ids, labels, _ in loader:
            images, q_ids, labels = images.to(device), q_ids.to(device), labels.to(device)
            pred = model(images, q_ids).argmax(1)
            correct += (pred == labels).sum().item()
            total += len(labels)
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--output", default="outputs/model.pth")
    args = ap.parse_args()

    device = torch.device("cpu")
    train_items = json.loads((Path(args.data_dir) / "train" / "questions.json").read_text())
    vocab = build_vocab([i["question"] for i in train_items])
    Path("outputs").mkdir(exist_ok=True)
    (Path(args.output).parent / "vocab.json").write_text(json.dumps(vocab))

    train_ds = load_split(args.data_dir, "train", vocab)
    val_ds = load_split(args.data_dir, "val", vocab)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = VQAModel(num_answers=len(ANSWERS), vocab_size=len(vocab)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    best = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = correct = total = 0
        for images, q_ids, labels, _ in train_loader:
            images, q_ids, labels = images.to(device), q_ids.to(device), labels.to(device)
            opt.zero_grad()
            out = model(images, q_ids)
            loss = loss_fn(out, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(labels)
            correct += (out.argmax(1) == labels).sum().item()
            total += len(labels)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs}  train loss {total_loss/total:.4f} "
              f"acc {correct/total:.4f}  val acc {val_acc:.4f}")
        history.append({"epoch": epoch, "train_loss": total_loss / total,
                        "train_acc": correct / total, "val_acc": val_acc})
        if val_acc >= best:
            best = val_acc
            torch.save({"state_dict": model.state_dict(), "vocab": vocab,
                        "answers": ANSWERS}, args.output)
            print(f"  -> saved best checkpoint to {args.output} (val acc {best:.4f})")
    with open("outputs/training_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_acc"])
        w.writeheader()
        w.writerows(history)
    print(f"Done. Best val accuracy: {best:.4f}")


if __name__ == "__main__":
    main()
