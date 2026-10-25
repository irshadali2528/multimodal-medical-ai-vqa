# Multimodal Medical AI: Vision-Language Model for Medical Image Question Answering

A multimodal AI prototype that connects medical image understanding with
natural-language question answering (VQA).

- **Task:** Given a medical image and a natural-language question, predict an
  answer from a closed answer set (for example: normal, pneumonia, fracture,
  yes, no). This is a prototype, not a clinical diagnostic tool.
- **Vision encoder:** Pretrained ResNet18 (ImageNet weights) — visual features.
- **Language encoder:** Pretrained DistilBERT (`distilbert-base-uncased`) via
  Hugging Face Transformers — question features. If that download is
  unavailable, the code falls back to a small trainable question encoder so
  the pipeline still runs.
- **Fusion:** Image and question features are projected to a shared space,
  fused (element-wise product + concatenation), and passed to an answer
  classifier.
- **Optional generative path:** `src/vlm.py` wraps a pretrained BLIP VQA model
  (`Salesforce/blip-vqa-base`) for open-ended answers. It is optional and is
  only used when that model is downloaded; it is not required for the
  prototype above.
- **Evaluation:** Accuracy, precision, recall, F1-score, confusion matrix,
  per-question-type accuracy, and qualitative examples (image, question,
  predicted answer, expected answer).

> **No clinical accuracy is claimed here.** The demo run in `DEMO_RUN.md`
> uses synthetic, computer-generated images only, to prove the pipeline
> runs end-to-end. Do not quote demo numbers as medical performance.
> Real evaluation needs a real, licensed medical VQA dataset (for example
> VQA-RAD or PathVQA), proper clinical review, and regulatory approval
> before any clinical use.

## Data format

```json
[
  {"image": "train/normal_0.jpg", "question": "Is this image normal?", "answer": "yes", "question_type": "yes_no"},
  {"image": "train/pneumonia_0.jpg", "question": "What abnormality is visible?", "answer": "pneumonia", "question_type": "abnormality"}
]
```

The demo generator (`src/demo_data.py`) creates synthetic images and a
`questions.json` in this format. Replace `data/` with a real dataset in
the same format to run the real experiment.

## Quick start

```bash
pip install -r requirements.txt

# 1. Create synthetic demo data (pipeline test only, NOT real medical images)
python src/demo_data.py --output-dir demo_data

# 2. Train the fusion model
python src/train.py --data-dir demo_data --epochs 5 --batch-size 8 --output outputs/model.pth

# 3. Evaluate on the test split
python src/evaluate.py --data-dir demo_data --checkpoint outputs/model.pth --output outputs/metrics.json

# 4. Ask a question about one image
python src/predict.py --image demo_data/test/pneumonia_0.jpg \
  --question "What abnormality is visible?" \
  --checkpoint outputs/model.pth

# 5. Optional: open-ended BLIP VQA (downloads the BLIP model on first use)
python src/vlm.py --image path/to/image.jpg --question "What is shown in this image?"
```

## Project structure

```
.
├── README.md
├── DEMO_RUN.md
├── requirements.txt
├── src/
│   ├── dataset.py      # VQA dataset, image + question loading
│   ├── model.py        # Pretrained vision + language encoders, fusion, classifier
│   ├── vlm.py          # Optional pretrained BLIP VQA wrapper
│   ├── demo_data.py    # Synthetic demo image + question generator (test only)
│   ├── train.py        # Training loop, best checkpoint by validation accuracy
│   ├── evaluate.py     # Metrics, confusion matrix, per-type accuracy, examples
│   └── predict.py      # Single image + question -> answer
└── outputs/            # Checkpoints, metrics, qualitative examples
```

## Technologies

Python, PyTorch, Transformers, Computer Vision, Multimodal AI, Generative AI.
