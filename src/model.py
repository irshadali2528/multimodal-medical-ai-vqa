"""Vision-language fusion model for medical image question answering.

Vision: pretrained ResNet18 (ImageNet).
Language: pretrained DistilBERT if available, else a small trainable encoder.
Fusion: project both to a shared dimension, element-wise product + concat,
then an answer classifier over a closed answer set.
"""
import torch
import torch.nn as nn
from torchvision import models


class QuestionEncoder(nn.Module):
    def __init__(self, vocab_size, hidden=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.lstm = nn.LSTM(128, hidden, batch_first=True)
        self.hidden = hidden

    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        _, (h, _) = self.lstm(emb)
        return h.squeeze(0)


class PretrainedQuestionEncoder(nn.Module):
    """DistilBERT encoder, frozen except the last layer is not fine-tuned here
    for the small demo; the pooled output is used as the question feature."""

    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        for p in self.model.parameters():
            p.requires_grad = False
        self.hidden = self.model.config.hidden_size

    def encode(self, questions, device):
        toks = self.tokenizer(
            questions, padding=True, truncation=True, max_length=32,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = self.model(**toks).last_hidden_state
        mask = toks["attention_mask"].unsqueeze(-1)
        return (out * mask).sum(1) / mask.sum(1).clamp(min=1)

    def forward(self, input_ids):
        raise NotImplementedError("Use encode() with raw questions.")


class VQAModel(nn.Module):
    def __init__(self, num_answers, vocab_size=5000, shared=256):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.vision = backbone
        self.vision_proj = nn.Linear(512, shared)
        self.question = QuestionEncoder(vocab_size, hidden=shared)
        self.question_proj = nn.Linear(shared, shared)
        self.fusion = nn.Sequential(
            nn.Linear(shared * 2, shared),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(shared, num_answers)

    def forward(self, images, input_ids):
        v = self.vision_proj(self.vision(images))
        q = self.question_proj(self.question(input_ids))
        fused = self.fusion(torch.cat([v * q, v + q], dim=1))
        return self.classifier(fused)
