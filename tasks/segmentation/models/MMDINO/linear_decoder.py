import torch
import torch.nn as nn


class LinearHead(nn.Module):

    def __init__(
        self,
        in_ch: int,
        n_classes: int = 1000,
    ):
        super().__init__()

        self.proj = nn.Conv2d(in_ch, n_classes, 1)
        self.up = nn.Upsample(scale_factor=2,
                              mode="bilinear",
                              align_corners=False)

    def forward(self, *modalities):  # fmap: [B, C, H, W]（步长16）
        if len(modalities) == 1:
            x = modalities[0]
            return self.up(self.proj(x[-1]))  # 输入分辨率的logits
        else:
            features_list = []
            for modality in modalities:
                features = modality
                features_list.append(features[-1])
            return self.up(self.proj(sum(features_list)))
