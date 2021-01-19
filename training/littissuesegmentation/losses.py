from typing import List

import torch
import pytorch_lightning as pl

## GREATLY inspire by https://github.com/ternaus/cloths_segmentation

EPSILON = 1e-4


def binary_mean_iou(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    output = (logits > 0).int()

    if output.shape != targets.shape:
        targets = torch.squeeze(targets, 1)

    intersection = (targets.flatten() * output.flatten()).sum()
    union = targets.sum() + output.sum() - intersection
    assert union >= 0, "Union is less than 0 in binary mean IOU"

    result = (intersection + EPSILON) / (union + EPSILON)

    return result


def dice_from_jaccard(j, jaccard_loss=True):
    if jaccard_loss:
        j = 1 - j

    return 2 * j / (1 + j)


class DiceLossWithLogits(torch.nn.Module):
    def __init__(self, smooth=0.1):
        super().__init__()
        self.smooth = smooth
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, logits, truth):
        probs = self.sigmoid(logits)

        iflat = probs.view(-1)
        tflat = truth.float().view(-1)
        intersection = (iflat * tflat).sum()
        union = iflat.sum() + tflat.sum() + self.smooth
        dice = (2.0 * intersection + self.smooth) / union
        return 1 - dice


class BinaryFocalLossWithLogits(torch.nn.Module):
    def __init__(self, gamma, *args, **kwargs):
        super().__init__()
        self.gamma = gamma
        self.bce = torch.nn.BCEWithLogitsLoss(*args, **kwargs)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, logits, truth):
        if len(truth.shape) == 3:
            truth = truth.unsqueeze(1)
        probs = self.sigmoid(logits)
        loss = self.bce(logits, truth.float())

        modulation_positive = (1 - probs) ** self.gamma
        modulation_negative = probs ** self.gamma

        mask = truth.bool()
        modulation = torch.where(mask, modulation_positive, modulation_negative)

        return (modulation * loss).mean()


# https://github.com/ternaus/iglovikov_helper_functions/blob/67f63742b591347242e2a66ca4d493ac7f15841a/iglovikov_helper_functions/dl/pytorch/lightning.py
def find_average(outputs: List, name: str) -> torch.Tensor:
    if len(outputs[0][name].shape) == 0:
        return torch.stack([x[name] for x in outputs]).mean()
    return torch.cat([x[name] for x in outputs]).mean()
