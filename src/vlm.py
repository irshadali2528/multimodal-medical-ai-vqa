"""Optional generative path: pretrained BLIP VQA.

Usage:
    python src/vlm.py --image path/to/image.jpg --question "What is shown?"

Downloads Salesforce/blip-vqa-base on first use (about 1 GB).
This is optional — the fusion prototype in model.py does not need it.
This model is a general VQA model, not a certified medical device.
"""
import argparse

import torch
from PIL import Image
from transformers import BlipForQuestionAnswering, BlipProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--model", default="Salesforce/blip-vqa-base")
    args = ap.parse_args()

    processor = BlipProcessor.from_pretrained(args.model)
    model = BlipForQuestionAnswering.from_pretrained(args.model)
    image = Image.open(args.image).convert("RGB")
    inputs = processor(image, args.question, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_length=20)
    print(processor.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
