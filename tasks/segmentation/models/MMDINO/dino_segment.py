import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear_decoder import LinearHead
from .Decoder import Decoder, Decoder_FRM, Decoder_PRN, Decoder_MMFF, Decoder_FRM_MMFF, Decoder_PRN_MMFF, Decoder_FRM_PRN
from .sample_adapter import SampleAdapter
from .lora import LoRA
from .ResNet import ResNet50

# 添加项目根目录到 Python 路径中，以便可以导入 dinov3 模块
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dinov3.hub.backbones import dinov3_vitl16, dinov3_vits16plus, dinov3_vitb16, dinov3_vits16, dinov3_vit7b16

# 添加项目根目录到 Python 路径中，以便可以导入 dinov3 模块
deps_path = os.path.join(os.path.dirname(__file__), "task/segmentation")
sys.path.insert(0, deps_path)

BACKBONE_INTERMEDIATE_LAYERS = {
    "dinov3_vits16": [2, 5, 8, 11],
    "dinov3_vits16plus": [2, 5, 8, 11],
    "dinov3_vitb16": [2, 5, 8, 11],
    "dinov3_vitl16": [4, 11, 17, 23],
    "dinov3_vit7b16": [9, 19, 29, 39],
}


class DINOSegmentModule(nn.Module):

    def __init__(
        self,
        backbone_weights=None,
        freeze_backbone: bool = False,
        n_classes: int = 1000,
        # window_size=(224, 224),
        use_lora: bool = False,
        r: int = 3,
        decoder_type='Decoder',
        adapter_type=None,
        backbone_type='dinov3_vitl16',
        # lora_layers=None,
        num_modalities: int = 1,
    ):
        super().__init__()

        dinov3_vits_dict = {
            "dinov3_vits16": dinov3_vits16,
            "dinov3_vits16plus": dinov3_vits16plus,
            "dinov3_vitb16": dinov3_vitb16,
            "dinov3_vitl16": dinov3_vitl16,
            "dinov3_vit7b16": dinov3_vit7b16
        }
        dinov3_vit = dinov3_vits_dict[backbone_type]
        self.backbone_type = backbone_type
        if backbone_weights is not None:
            self.backbone = dinov3_vit(weights=backbone_weights,
                                       pretrained=True)
        else:
            self.backbone = dinov3_vit(pretrained=False)

        # Important: we freeze the backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)

        embed_dim = self.backbone.embed_dim

        # 根据类型选择适配器
        self.adapter = None
        if adapter_type == 'SampleAdapter':
            self.adapter = SampleAdapter(embed_dim,
                                         num_modalities=num_modalities)

        # 根据类型选择解码器
        decoder_kwargs = {
            "n_classes": n_classes,
            "num_modalities": num_modalities
        }
        if adapter_type is None:
            decoder_kwargs["in_channels"] = [embed_dim] * 4

        if decoder_type == 'LinearHead':
            self.decoder = LinearHead(in_ch=embed_dim, n_classes=n_classes)
        elif decoder_type == 'Decoder':
            self.decoder = Decoder(**decoder_kwargs)
        elif decoder_type == 'Decoder_FRM':
            self.decoder = Decoder_FRM(**decoder_kwargs)
        elif decoder_type == 'Decoder_MMFF':
            self.decoder = Decoder_MMFF(**decoder_kwargs)
        elif decoder_type == 'Decoder_PRN':
            self.decoder = Decoder_PRN(**decoder_kwargs)
        elif decoder_type == 'Decoder_FRM_MMFF':
            self.decoder = Decoder_FRM_MMFF(**decoder_kwargs)
        elif decoder_type == 'Decoder_PRN_MMFF':
            self.decoder = Decoder_PRN_MMFF(**decoder_kwargs)
        elif decoder_type == 'Decoder_FRM_PRN':
            self.decoder = Decoder_FRM_PRN(**decoder_kwargs)
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

        # Add LoRA layers to the encoder
        self.use_lora = use_lora
        if self.use_lora:
            self.lora_layers = list(range(len(self.backbone.blocks)))
            self.w_a = []
            self.w_b = []

            for i, block in enumerate(self.backbone.blocks):
                if i not in self.lora_layers:
                    continue
                w_qkv_linear = block.attn.qkv
                dim = w_qkv_linear.in_features

                w_a_linear_q, w_b_linear_q = self._create_lora_layer(dim, r)
                w_a_linear_v, w_b_linear_v = self._create_lora_layer(dim, r)

                self.w_a.extend([w_a_linear_q, w_a_linear_v])
                self.w_b.extend([w_b_linear_q, w_b_linear_v])

                block.attn.qkv = LoRA(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                )
            self._reset_lora_parameters()

    def _create_lora_layer(self, dim: int, r: int):
        w_a = nn.Linear(dim, r, bias=False)
        w_b = nn.Linear(r, dim, bias=False)
        return w_a, w_b

    def _reset_lora_parameters(self) -> None:
        for w_a in self.w_a:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_b:
            nn.init.zeros_(w_b.weight)

    def forward(self, *modalities):
        if len(modalities) == 0:
            raise ValueError("At least one modality must be provided")

        # 主输入x
        x = modalities[0]
        _, C, H, W = x.shape
        patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16

        scale_factors = [4, 2, 1, 0.5]

        if len(modalities) == 1:
            if self.adapter is not None:
                with torch.autocast("cuda", torch.float32):
                    outputs = self.backbone.get_intermediate_layers(
                        x, n=BACKBONE_INTERMEDIATE_LAYERS[self.backbone_type])
                # 使用适配器处理多尺度特征
                multi_scale_features = self.adapter(outputs,
                                                    patch_h=patch_h,
                                                    patch_w=patch_w)
            else:
                with torch.autocast("cuda", torch.float32):
                    outputs = self.backbone.get_intermediate_layers(
                        x,
                        n=BACKBONE_INTERMEDIATE_LAYERS[self.backbone_type],
                        reshape=True)
                # 直接处理中间层输出
                multi_scale_features = []
                for i, output in enumerate(outputs):
                    # output = output.permute(0, 2, 1).reshape(
                    #     (output.shape[0], output.shape[-1], patch_h, patch_w))

                    if i < len(scale_factors):
                        output = F.interpolate(output,
                                               scale_factor=scale_factors[i],
                                               mode="bilinear",
                                               align_corners=False)
                    multi_scale_features.append(output)

            logits = self.decoder(multi_scale_features)

        else:
            outputs_modalities = []
            for idx, modality_input in enumerate(modalities):
                if modality_input.shape[1] != C and idx > 0:
                    modality_input = modality_input.repeat(1, C, 1, 1)

                outputs_modality = self.backbone.get_intermediate_layers(
                    modality_input,
                    n=BACKBONE_INTERMEDIATE_LAYERS[self.backbone_type])
                outputs_modalities.append(outputs_modality)

            if self.adapter is not None:
                # 使用适配器处理多尺度特征
                processed_outputs_modalities = self.adapter(
                    *outputs_modalities, patch_h=patch_h, patch_w=patch_w)
            else:
                processed_outputs_modalities = []
                for outputs_modality in outputs_modalities:
                    # 直接处理中间层输出
                    processed_outputs_modality = []
                    for i, output in enumerate(outputs_modality):
                        output = output.permute(0, 2, 1).reshape(
                            (output.shape[0], output.shape[-1], patch_h,
                             patch_w))

                        if i < len(scale_factors):
                            output = F.interpolate(
                                output,
                                scale_factor=scale_factors[i],
                                mode="bilinear",
                                align_corners=False)
                        processed_outputs_modality.append(output)

                    processed_outputs_modalities.append(
                        processed_outputs_modality)

            # 将处理后的所有模态特征传递给解码器
            logits = self.decoder(*processed_outputs_modalities)

        _H, _W = logits.shape[2:]
        if _H != H or _W != W:
            # 确保输出大小与输入一致
            pred = F.interpolate(
                logits,
                size=(H, W),
                mode="bilinear",
            )

        return pred


class ResNetSegmentModule(nn.Module):

    def __init__(
        self,
        n_classes: int = 1000,
        use_lora: bool = False,
        r: int = 3,
        decoder_type='Decoder',
        num_modalities: int = 1,
    ):
        super().__init__()

        self.backbone = ResNet50(pretrained=True)

        embed_dim = [256, 512, 1024, 2048]

        # 根据类型选择解码器
        decoder_kwargs = {
            "in_channels": embed_dim,
            "n_classes": n_classes,
            "num_modalities": num_modalities
        }

        if decoder_type == 'LinearHead':
            self.decoder = LinearHead(in_ch=embed_dim, n_classes=n_classes)
        elif decoder_type == 'Decoder':
            self.decoder = Decoder(**decoder_kwargs)
        elif decoder_type == 'Decoder_FRM':
            self.decoder = Decoder_FRM(**decoder_kwargs)
        elif decoder_type == 'Decoder_MMFF':
            self.decoder = Decoder_MMFF(**decoder_kwargs)
        elif decoder_type == 'Decoder_PRN':
            self.decoder = Decoder_PRN(**decoder_kwargs)
        elif decoder_type == 'Decoder_FRM_MMFF':
            self.decoder = Decoder_FRM_MMFF(**decoder_kwargs)
        elif decoder_type == 'Decoder_PRN_MMFF':
            self.decoder = Decoder_PRN_MMFF(**decoder_kwargs)
        elif decoder_type == 'Decoder_FRM_PRN':
            self.decoder = Decoder_FRM_PRN(**decoder_kwargs)
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

        # Add LoRA layers to the encoder
        self.use_lora = use_lora
        if self.use_lora:
            self.lora_layers = list(range(len(self.backbone.blocks)))
            self.w_a = []
            self.w_b = []

            for i, block in enumerate(self.backbone.blocks):
                if i not in self.lora_layers:
                    continue
                w_qkv_linear = block.attn.qkv
                dim = w_qkv_linear.in_features

                w_a_linear_q, w_b_linear_q = self._create_lora_layer(dim, r)
                w_a_linear_v, w_b_linear_v = self._create_lora_layer(dim, r)

                self.w_a.extend([w_a_linear_q, w_a_linear_v])
                self.w_b.extend([w_b_linear_q, w_b_linear_v])

                block.attn.qkv = LoRA(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                )
            self._reset_lora_parameters()

    def _create_lora_layer(self, dim: int, r: int):
        w_a = nn.Linear(dim, r, bias=False)
        w_b = nn.Linear(r, dim, bias=False)
        return w_a, w_b

    def _reset_lora_parameters(self) -> None:
        for w_a in self.w_a:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_b:
            nn.init.zeros_(w_b.weight)

    def forward(self, *modalities):
        if len(modalities) == 0:
            raise ValueError("At least one modality must be provided")

        # 主输入x
        x = modalities[0]
        _, C, H, W = x.shape

        if len(modalities) == 1:
            outputs = self.backbone(x)

            logits = self.decoder(outputs)

        else:
            outputs_modalities = []
            for idx, modality_input in enumerate(modalities):
                if modality_input.shape[1] != C and idx > 0:
                    modality_input = modality_input.repeat(1, C, 1, 1)

                outputs_modality = self.backbone(modality_input)
                outputs_modalities.append(outputs_modality)

            # 将处理后的所有模态特征传递给解码器
            logits = self.decoder(*outputs_modalities)

        _H, _W = logits.shape[2:]
        if _H != H or _W != W:
            # 确保输出大小与输入一致
            pred = F.interpolate(
                logits,
                size=(H, W),
                mode="bilinear",
            )

        return pred


def build_model(
    model_name=None,
    backbone_weights=None,
    n_classes: int = 1000,
    use_lora: bool = False,
    r: int = 3,
    num_modalities: int = 1,
    **kwargs,
):
    if model_name == 'DINOv3' or model_name is None:
        model = DINOSegmentModule(
            backbone_weights=backbone_weights,
            n_classes=n_classes,
            use_lora=use_lora,
            r=r,
            num_modalities=num_modalities,
            adapter_type="SampleAdapter",
            decoder_type="Decoder",
            **kwargs,
        )
    elif model_name == 'DINOv3_Adapter_FRM':
        model = DINOSegmentModule(
            backbone_weights=backbone_weights,
            n_classes=n_classes,
            use_lora=use_lora,
            r=r,
            num_modalities=num_modalities,
            adapter_type="SampleAdapter",
            decoder_type="Decoder_FRM",
            **kwargs,
        )
    elif model_name == 'DINOv3_Adapter_PRN':
        model = DINOSegmentModule(backbone_weights=backbone_weights,
                                  n_classes=n_classes,
                                  use_lora=use_lora,
                                  r=r,
                                  num_modalities=num_modalities,
                                  adapter_type="SampleAdapter",
                                  decoder_type="Decoder_PRN",
                                  **kwargs)
    elif model_name == 'DINOv3_Adapter_MMFF':
        model = DINOSegmentModule(backbone_weights=backbone_weights,
                                  n_classes=n_classes,
                                  use_lora=use_lora,
                                  r=r,
                                  num_modalities=num_modalities,
                                  adapter_type="SampleAdapter",
                                  decoder_type="Decoder_MMFF",
                                  **kwargs)
    elif model_name == 'DINOv3_Adapter_FRM_MMFF':
        model = DINOSegmentModule(backbone_weights=backbone_weights,
                                  n_classes=n_classes,
                                  use_lora=use_lora,
                                  r=r,
                                  num_modalities=num_modalities,
                                  adapter_type="SampleAdapter",
                                  decoder_type="Decoder_FRM_MMFF",
                                  **kwargs)
    elif model_name == 'DINOv3_Adapter_PRN_MMFF':
        model = DINOSegmentModule(backbone_weights=backbone_weights,
                                  n_classes=n_classes,
                                  use_lora=use_lora,
                                  r=r,
                                  num_modalities=num_modalities,
                                  adapter_type="SampleAdapter",
                                  decoder_type="Decoder_PRN_MMFF",
                                  **kwargs)
    elif model_name == 'DINOv3_FRM':
        model = DINOSegmentModule(backbone_weights=backbone_weights,
                                  n_classes=n_classes,
                                  use_lora=use_lora,
                                  r=r,
                                  num_modalities=num_modalities,
                                  decoder_type="Decoder_FRM",
                                  **kwargs)
    elif model_name == 'DINOv3_PRN':
        model = DINOSegmentModule(backbone_weights=backbone_weights,
                                  n_classes=n_classes,
                                  use_lora=use_lora,
                                  r=r,
                                  num_modalities=num_modalities,
                                  decoder_type="Decoder_PRN",
                                  **kwargs)
    elif model_name == 'DINOv3_MMFF':
        model = DINOSegmentModule(backbone_weights=backbone_weights,
                                  n_classes=n_classes,
                                  use_lora=use_lora,
                                  r=r,
                                  num_modalities=num_modalities,
                                  decoder_type="Decoder_MMFF",
                                  **kwargs)
    elif model_name == 'DINOv3_FRM_MMFF':
        model = DINOSegmentModule(backbone_weights=backbone_weights,
                                  n_classes=n_classes,
                                  use_lora=use_lora,
                                  r=r,
                                  num_modalities=num_modalities,
                                  decoder_type="Decoder_FRM_MMFF",
                                  **kwargs)
    elif model_name == 'DINOv3_PRN_MMFF':
        model = DINOSegmentModule(backbone_weights=backbone_weights,
                                  n_classes=n_classes,
                                  use_lora=use_lora,
                                  r=r,
                                  num_modalities=num_modalities,
                                  decoder_type="Decoder_PRN_MMFF",
                                  **kwargs)
    elif model_name == 'DINOv3_Baseline':
        model = DINOSegmentModule(backbone_weights=backbone_weights,
                                  n_classes=n_classes,
                                  use_lora=use_lora,
                                  r=r,
                                  num_modalities=num_modalities,
                                  decoder_type="LinearHead",
                                  **kwargs)
    elif model_name == 'DINOv3_Adapter':
        model = DINOSegmentModule(backbone_weights=backbone_weights,
                                  n_classes=n_classes,
                                  use_lora=use_lora,
                                  r=r,
                                  num_modalities=num_modalities,
                                  adapter_type="SampleAdapter",
                                  decoder_type="LinearHead",
                                  **kwargs)
    elif model_name == 'DINOv3_FRM_MMFF_PRN':
        model = DINOSegmentModule(backbone_weights=backbone_weights,
                                  n_classes=n_classes,
                                  use_lora=use_lora,
                                  r=r,
                                  num_modalities=num_modalities,
                                  decoder_type="Decoder",
                                  **kwargs)
    elif model_name == 'DINOv3_FRM_PRN':
        model = DINOSegmentModule(backbone_weights=backbone_weights,
                                  n_classes=n_classes,
                                  use_lora=use_lora,
                                  r=r,
                                  num_modalities=num_modalities,
                                  decoder_type="Decoder_FRM_PRN",
                                  **kwargs)
    elif model_name == 'DINOv3_Adapter_FRM_PRN':
        model = DINOSegmentModule(backbone_weights=backbone_weights,
                                  n_classes=n_classes,
                                  use_lora=use_lora,
                                  r=r,
                                  num_modalities=num_modalities,
                                  adapter_type="SampleAdapter",
                                  decoder_type="Decoder_FRM_PRN",
                                  **kwargs)
    elif model_name == 'DINOv3_ResNet50':
        model = ResNetSegmentModule(n_classes=n_classes,
                                    use_lora=use_lora,
                                    r=r,
                                    num_modalities=num_modalities,
                                    decoder_type="Decoder")

    return model
