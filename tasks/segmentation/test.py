import os
import sys
import torch
import numpy as np

from tqdm import tqdm

# 添加项目根目录到 Python 路径中，以便可以导入 dinov3 模块
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import dinov3.distributed as distributed

from utils.metrics import metrics_print_version as metrics
from utils.inference import slide_inference
from utils import save_prediction_as_image, plot_confusion_matrix

DATASET_NAME = ""
MODEL_NAME = ""
NUM_MODALITIES = -1
CHECKPOINT_PATH = None

from datasets import build_dataset
from configs import get_cfg
from configs.common_cfg import MS_ROOT_DIR


def get_local_rank():
    """获取本地rank"""
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    else:
        return 0


def main(**kwargs):
    try:
        # 初始化分布式训练环境
        distributed.enable(overwrite=True)
    except Exception as e:
        print(f"Failed to initialize distributed training: {e}")
        print("Falling back to single GPU training")
        # 手动设置环境以进行单GPU训练
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

    # 获取模型配置
    # cfg = cfg_module.get_cfg(MODEL_NAME, DATASET_NAME)
    cfg = get_cfg(MODEL_NAME, DATASET_NAME, **kwargs)
    window_size = cfg.get('window_size')
    model = cfg.get('model')

    test_dataset = build_dataset(
        DATASET_NAME,
        "test",
        window_size=window_size,
        model_name=MODEL_NAME,
        modality="multi" if NUM_MODALITIES > 1 else None,
        backbone_type=kwargs.get('backbone_type'),
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

    # 将模型移到GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.load_state_dict(torch.load(CHECKPOINT_PATH)["model"], strict=True)

    # 如果分布式训练可用，则包装为分布式模型
    if distributed.is_enabled():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[get_local_rank()]
            if torch.cuda.is_available() else None,
            output_device=get_local_rank()
            if torch.cuda.is_available() else None,
            find_unused_parameters=True,  # 这将允许模型在某些参数未参与损失计算时仍能正常工作
        )

    test(model, test_loader, cfg)


def test(model, test_loader, cfg):
    # 清理缓存
    torch.cuda.empty_cache()
    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    preds = []
    labels = []

    window_size = cfg.get("window_size")
    classes = cfg.get("labels")

    # 创建保存结果的目录
    modality = "uni" if NUM_MODALITIES == 1 else "multi"
    save_dir = f"./vis_results/{MODEL_NAME}_{DATASET_NAME}_{modality}"
    os.makedirs(save_dir, exist_ok=True)
    sample_index = 0

    iterations = tqdm(test_loader, disable=not distributed.is_main_process())
    for batch in iterations:
        if NUM_MODALITIES > 1:
            input, dsm, label = batch
            input, dsm = input.to(device), dsm.to(device)

            with torch.no_grad():
                s_w = int(window_size[0] * 2 / 3)
                pred = slide_inference(input,
                                       model,
                                       dsm=dsm,
                                       n_output_channels=len(classes),
                                       crop_size=window_size,
                                       stride=(s_w, s_w),
                                       batch_size=cfg.get("batch_size", 4))
        else:
            input, label = batch
            input = input.to(device)

            with torch.no_grad():
                s_w = int(window_size[0] * 2 / 3)
                pred = slide_inference(input,
                                       model,
                                       n_output_channels=len(classes),
                                       crop_size=window_size,
                                       stride=(s_w, s_w),
                                       batch_size=cfg.get("batch_size", 4))

        pred = np.argmax(pred, axis=1)
        preds.append(pred)
        labels.append(label)

        # 保存预测结果为图像
        save_prediction_as_image(pred,
                                 label.numpy(),
                                 save_dir,
                                 sample_index,
                                 dataset_name=DATASET_NAME)
        sample_index += 1

    MIoU, F1, Kappa, Acc, cm = metrics(
        np.concatenate([p.ravel() for p in preds]),
        np.concatenate([p.ravel() for p in labels]).ravel(), classes)

    plot_confusion_matrix(cm, classes[:-1],
                          os.path.join(save_dir, "confusion_matrix.png"))

    # 构建详细指标字典
    detailed_metrics = {"MIoU": MIoU, "F1": F1, "Kappa": Kappa, "Acc": Acc}

    return detailed_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train segmentation model')
    parser.add_argument('--model-name',
                        type=str,
                        default='DINOv3',
                        help='Name of the model to train')
    parser.add_argument('--dataset-name',
                        type=str,
                        default='WHU',
                        help='Dataset of the model to train')
    parser.add_argument('--num-modalities',
                        type=int,
                        default=1,
                        help='Number of modality to train')
    parser.add_argument('--use-lora',
                        type=bool,
                        default=False,
                        help='use lora or not')
    parser.add_argument('--r', type=int, default=3, help='lora r')
    parser.add_argument('--backbone-type',
                        type=str,
                        default='dinov3_vits16',
                        help='backbone type')
    parser.add_argument('--checkpoint-path',
                        type=str,
                        default=None,
                        help='checkpoint path')
    args = parser.parse_args()

    # 如果提供了模型名称参数，使用它；否则使用默认值
    if args.model_name:
        MODEL_NAME = args.model_name
    if args.num_modalities:
        NUM_MODALITIES = args.num_modalities
    if args.dataset_name:
        DATASET_NAME = args.dataset_name

    if args.checkpoint_path is None:
        raise ValueError('Please provide a checkpoint path.')
    else:
        CHECKPOINT_PATH = args.checkpoint_path

    main(
        num_modalities=NUM_MODALITIES,
        use_lora=args.use_lora,
        r=args.r,
        backbone_type=args.backbone_type,
    )
