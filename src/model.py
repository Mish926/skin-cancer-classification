"""
Model Architecture
------------------
ResNet50 with transfer learning for 7-class skin lesion classification.

Design decisions:
  - Pretrained ImageNet weights: leverages low-level texture/color features
    that transfer well to dermatoscopic images.
  - Frozen backbone (first 2 stages) during warm-up phase to avoid
    destroying pretrained features early in training.
  - Custom classifier head: Dropout(0.5) → Linear(2048→512) → ReLU → 
    Dropout(0.3) → Linear(512→7)
  - Supports gradual unfreezing (discriminative fine-tuning).
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


NUM_CLASSES = 7


class SkinLesionClassifier(nn.Module):
    """
    ResNet50 fine-tuned for HAM10000 7-class classification.

    Args:
        num_classes:    number of output classes (default 7)
        dropout_rate:   dropout before first FC layer (default 0.5)
        pretrained:     load ImageNet weights (default True)
        freeze_layers:  number of ResNet layer blocks to freeze (0–4)
                        0 = full fine-tune, 4 = only train classifier head
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout_rate: float = 0.5,
        pretrained: bool = True,
        freeze_layers: int = 2,
    ):
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # ── Selectively freeze backbone layers ─────────────────────────────────
        layers_to_freeze = [
            backbone.conv1, backbone.bn1,
            backbone.layer1, backbone.layer2,
            backbone.layer3, backbone.layer4,
        ]
        for layer in layers_to_freeze[:freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

        # ── Strip original FC head, keep feature extractor ────────────────────
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        in_features = backbone.fc.in_features  # 2048 for ResNet50

        # ── Custom classifier head ─────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate * 0.6),  # lighter second dropout
            nn.Linear(512, num_classes),
        )

        # ── Weight initialization for new layers ──────────────────────────────
        self._init_classifier()

    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        features = torch.flatten(features, 1)
        return self.classifier(features)

    def unfreeze_all(self):
        """Call after warm-up to fine-tune the full network."""
        for param in self.parameters():
            param.requires_grad = True

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(
    num_classes: int = NUM_CLASSES,
    freeze_layers: int = 2,
    device: str = "cuda",
) -> SkinLesionClassifier:
    """Convenience factory with device placement."""
    model = SkinLesionClassifier(num_classes=num_classes, freeze_layers=freeze_layers)
    model = model.to(device)

    total  = model.get_total_params()
    trainable = model.get_trainable_params()
    print(f"Model: ResNet50 | Total params: {total:,} | "
          f"Trainable: {trainable:,} ({100*trainable/total:.1f}%)")

    return model
