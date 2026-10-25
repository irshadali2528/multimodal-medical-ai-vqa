"""Evaluate VQA: accuracy, per-type accuracy, classification report."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from dataset import VQADataset, build_vocab
from model import VQAModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", default="outputs/metrics.json")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vocab, answers = ckpt["vocab"], ckpt["answers"]
    ds = VQADataset(args.data_dir, "test", vocab, answers)
    loader = DataLoader(ds, batch_size=8)
    model = VQAModel(num_answers=len(answers), vocab_size=len(vocab))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    y_true, y_pred, types, examples = [], [], [], []
    with torch.no_grad():
        for images, q_ids, labels, items in loader:
            pred = model(images, q_ids).argmax(1).tolist()
            # DataLoader collates the item dicts into a dict of lists.
            batch_items = [
                {k: v[i] for k, v in items.items()}
                for i in range(len(labels))
            ]
            for i, item in enumerate(batch_items):
                y_true.append(labels[i].item())
                y_pred.append(pred[i])
                types.append(item["question_type"])
                if len(examples) < 10:
                    examples.append({
                        "image": item["image"], "question": item["question"],
                        "expected": answers[labels[i].item()],
                        "predicted": answers[pred[i]],
                        "correct": answers[labels[i].item()] == answers[pred[i]],
                    })
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / max(len(y_true), 1)
    per_type = {}
    for t in set(types):
        idx = [i for i, x in enumerate(types) if x == t]
        per_type[t] = sum(y_true[i] == y_pred[i] for i in idx) / len(idx)
    metrics = {
        "accuracy": acc,
        "per_question_type_accuracy": per_type,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=answers, output_dict=True, zero_division=0),
        "classes": answers,
        "qualitative_examples": examples,
        "n_test": len(y_true),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2)[:2000])
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
