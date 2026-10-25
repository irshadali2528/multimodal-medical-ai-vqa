# Demo run — pipeline test only, NOT real medical images

This file records a test run made to prove the code works end-to-end.
It is **not** a result on real medical data and must not be quoted as
medical VQA accuracy, and this prototype is not a clinical tool.

## What was run

- Synthetic demo images were generated locally (normal: two lung-like
  ellipses; pneumonia: + bright opacity blobs; fracture: + a thin bright line).
  These are computer-generated patterns, not patient images.
- Questions are templated: "What abnormality is visible?" and
  "Is this image normal?", with answers normal / pneumonia / fracture / yes / no.
- Split: train 180 questions, val 60, test 60.
- Model: pretrained ResNet18 vision encoder + trainable question encoder,
  fusion + answer classifier, 5 epochs, batch 8, CPU.
- Commands:
  - `python src/demo_data.py --output-dir demo_data`
  - `python src/train.py --data-dir demo_data --epochs 5 --batch-size 8`
  - `python src/evaluate.py --data-dir demo_data --checkpoint outputs/demo_model.pth`
  - `python src/predict.py --image demo_data/test/pneumonia_0.jpg --question "What abnormality is visible?"`

## Demo results (synthetic data only)

- Best validation accuracy: 1.0000 (epoch 4; validation fluctuated across epochs)
- Test accuracy: 1.0, 60 test questions, per-type accuracy 1.0 for both
  abnormality and yes/no questions
- Qualitative examples: pneumonia image + "What abnormality is visible?"
  -> pneumonia (confidence 1.00); normal image + "Is this image normal?"
  -> yes (confidence 0.99)

These numbers only show that train -> evaluate -> predict runs without
errors on easy synthetic patterns. They say nothing about performance on
real medical images or real clinical questions.

## Next step for real results

Replace `demo_data/` with a real, licensed medical VQA dataset in the same
`questions.json` format (for example VQA-RAD or PathVQA), retrain, and
report only the metrics from `evaluate.py` on that real test split, with
clinical review before any clinical claim.
