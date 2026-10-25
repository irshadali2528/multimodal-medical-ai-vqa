"""Ask a question about one image."""
import argparse
import json

import torch
import torch.nn.functional as F
from PIL import Image

from dataset import encode_question, transform
from model import VQAModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = VQAModel(num_answers=len(ckpt["answers"]), vocab_size=len(ckpt["vocab"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    image = transform(Image.open(args.image).convert("RGB")).unsqueeze(0)
    q_ids = torch.tensor([encode_question(args.question, ckpt["vocab"])], dtype=torch.long)
    with torch.no_grad():
        probs = F.softmax(model(image, q_ids), dim=1)[0]
    idx = int(probs.argmax())
    print(f"Question: {args.question}")
    print(f"Answer: {ckpt['answers'][idx]} (confidence {probs[idx]:.2f})")


if __name__ == "__main__":
    main()
