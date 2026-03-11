import torch.optim as optim

from losses import *
from .common_cfg import *
from models.MMDINO.dino_segment import build_model

# 导入分布式训练相关模块
import dinov3.distributed as distributed


def get_cfg(model_name=None, dataset_name=None, **kwargs):
    if dataset_name is None:
        raise ValueError("Dataset name must be specified")

    base_lr = 1e-4
    batch_size = 8
    epochs = 50
    window_size = (512, 512)
    labels = get_labels(dataset_name)
    ignore_index = len(labels)
    loss_fn = JointLoss(
        SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
        DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)

    backbone_type = kwargs.get('backbone_type', "dinov3_vits16")
    backbone_weights_dict = {
        "dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        "dinov3_vits16plus":
        "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
        "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        "dinov3_vitl16": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
        "dinov3_vit7b16": "dinov3_vit7b16_pretrain_sat493m-a6675841.pth",
    }
    backbone_weights = f"{MS_ROOT_DIR}/Checkpoints/facebook/" + backbone_weights_dict[
        backbone_type]
    if model_name is not None:
        model = build_model(model_name=model_name,
                            backbone_weights=backbone_weights,
                            backbone_type=backbone_type,
                            freeze_backbone=True,
                            n_classes=len(labels),
                            use_lora=kwargs.get('use_lora'),
                            r=kwargs.get('r'),
                            num_modalities=kwargs.get('num_modalities', 1))
    else:
        raise ValueError("Model name not recognized")

    # 根据GPU数量调整学习率
    if distributed.is_enabled():
        base_lr = base_lr * distributed.get_world_size()

    # 分别为backbone和其他部分设置不同的学习率
    backbone_params = []
    other_params = []

    # 如果backbone中有需要训练的参数（如LoRA参数）
    if hasattr(model, 'backbone'):
        backbone_params = [
            p for p in model.backbone.parameters() if p.requires_grad
        ]

    # 其他所有需要训练的参数
    other_params = []
    for name, param in model.named_parameters():
        # 排除backbone中的参数，剩下的都是其他参数
        if not name.startswith('backbone') and param.requires_grad:
            other_params.append(param)

    # 为不同部分设置不同的学习率
    param_groups = [
        {
            'params': backbone_params,
            'lr': base_lr
        },
        {
            'params': other_params,
            'lr': base_lr
        }  # 其他部分使用正常学习率
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=epochs,
                                                     eta_min=1e-7)

    return dict(batch_size=batch_size,
                epochs=epochs,
                window_size=window_size,
                labels=labels,
                loss_fn=loss_fn,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler)
