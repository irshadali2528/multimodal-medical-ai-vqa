"""Generate synthetic demo images + questions (pipeline test ONLY).

These are computer-generated patterns, NOT real medical images:
- normal: two smooth lung-like ellipses
- pneumonia: ellipses + bright white opacity blobs
- fracture: ellipses + a thin bright diagonal line
Questions are templated; answers are the synthetic label or yes/no.
"""
import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

random.seed(42)
CLASSES = ["normal", "pneumonia", "fracture"]


def make_image(cls, path):
    img = Image.new("L", (224, 224), 20)
    draw = ImageDraw.Draw(img)
    draw.ellipse([30, 50, 100, 180], fill=90)
    draw.ellipse([124, 50, 194, 180], fill=90)
    if cls == "pneumonia":
        for _ in range(12):
            x, y = random.randint(35, 190), random.randint(55, 175)
            r = random.randint(8, 22)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=200)
    if cls == "fracture":
        draw.line([(60, 60), (160, 170)], fill=220, width=4)
    img.convert("RGB").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="demo_data")
    args = ap.parse_args()
    root = Path(args.output_dir)
    for split, n in [("train", 30), ("val", 10), ("test", 10)]:
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        questions = []
        for cls in CLASSES:
            for i in range(n):
                name = f"{cls}_{i}.jpg"
                make_image(cls, split_dir / name)
                questions.append({
                    "image": f"{split}/{name}",
                    "question": "What abnormality is visible?",
                    "answer": cls,
                    "question_type": "abnormality",
                })
                questions.append({
                    "image": f"{split}/{name}",
                    "question": "Is this image normal?",
                    "answer": "yes" if cls == "normal" else "no",
                    "question_type": "yes_no",
                })
        (split_dir / "questions.json").write_text(json.dumps(questions, indent=2))
        print(split, len(questions), "questions")
    print("Demo data written to", root)


if __name__ == "__main__":
    main()
