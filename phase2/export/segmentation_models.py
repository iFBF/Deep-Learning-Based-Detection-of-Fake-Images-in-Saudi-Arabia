"""FCN-style and U-Net-style binary segmenters sharing a ResNet18 encoder."""
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.models import resnet18

from cohort_data import safe_path
from segmentation_data import transform, flip_for


class SegmentationDataset(Dataset):
    def __init__(self, root, rows, training=False, epoch=0):
        self.root, self.rows, self.training, self.epoch = root, rows, training, epoch

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(safe_path(self.root, row["export_image"])) as opened:
            image = opened.convert("RGB")
        with Image.open(safe_path(self.root, row["export_mask"])) as opened:
            mask = np.asarray(opened, dtype=np.uint8)
        image, mask, valid = transform(image, mask, flip=self.training and flip_for(row["uid"], self.epoch))
        return {"image": torch.from_numpy(image), "mask": torch.from_numpy(mask),
                "valid": torch.from_numpy(valid), "index": index}


class Encoder(nn.Module):
    def __init__(self, pretrained=None):
        super().__init__()
        backbone = resnet18(weights=None)
        if pretrained is not None:
            backbone.load_state_dict(torch.load(pretrained, map_location="cpu", weights_only=True))
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.layer1, self.layer2 = backbone.layer1, backbone.layer2
        self.layer3, self.layer4 = backbone.layer3, backbone.layer4

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x0, x1, x2, x3, x4


class Block(nn.Sequential):
    def __init__(self, inputs, outputs):
        super().__init__(nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
                         nn.GroupNorm(8, outputs), nn.ReLU(inplace=True),
                         nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
                         nn.GroupNorm(8, outputs), nn.ReLU(inplace=True))


class FCNSegmenter(nn.Module):
    def __init__(self, pretrained=None):
        super().__init__()
        self.encoder = Encoder(pretrained)
        self.head = nn.Sequential(Block(512, 128), nn.Dropout2d(0.1), nn.Conv2d(128, 1, 1))

    def forward(self, x):
        logits = self.head(self.encoder(x)[-1])
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


class UNetSegmenter(nn.Module):
    def __init__(self, pretrained=None):
        super().__init__()
        self.encoder = Encoder(pretrained)
        self.blocks = nn.ModuleList([Block(512 + 256, 128), Block(128 + 128, 64),
                                    Block(64 + 64, 32), Block(32 + 64, 16)])
        self.final = nn.Sequential(Block(16, 8), nn.Conv2d(8, 1, 1))

    def forward(self, x):
        x0, x1, x2, x3, current = self.encoder(x)
        for block, skip in zip(self.blocks, [x3, x2, x1, x0], strict=True):
            current = F.interpolate(current, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            current = block(torch.cat([current, skip], dim=1))
        current = F.interpolate(current, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.final(current)


def build_model(architecture, pretrained=None):
    if architecture == "A_FCN_ResNet18":
        return FCNSegmenter(pretrained)
    if architecture == "B_UNet_ResNet18":
        return UNetSegmenter(pretrained)
    raise ValueError("Unknown segmentation architecture: " + architecture)


def segmentation_loss(logits, target, valid):
    """Mean valid-pixel BCE per image + soft Dice on positive resized masks; padding has zero gradient."""
    axes = (1, 2, 3)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = ((bce * valid).sum(axes) / valid.sum(axes).clamp_min(1)).mean()
    probability = logits.sigmoid() * valid
    truth = target * valid
    positives = truth.sum(axes) > 0
    dice = (2 * (probability * truth).sum(axes) + 1e-6) / (probability.sum(axes) + truth.sum(axes) + 1e-6)
    dice_loss = (1 - dice[positives]).mean() if positives.any() else logits.sum() * 0
    return 0.5 * bce + 0.5 * dice_loss
