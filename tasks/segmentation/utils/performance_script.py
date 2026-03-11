import os
import sys

import torch
# 添加项目根目录到 Python 路径中，以便可以导入 dinov3 模块
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

segmentation_path = os.path.join(project_root, "segmentation")
if segmentation_path not in sys.path:
    sys.path.insert(0, segmentation_path)

from configs import get_cfg

from utils import measure_model_performance, measure_training_time

if __name__ == "__main__":
    window_size = (512, 512)
    num_modalities = 1

    cfg = get_cfg("DINOv3", "WHU", num_modalities=num_modalities)
    model = cfg.get('model')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    measure_model_performance(model, window_size, num_modalities)
    measure_training_time(model, window_size, num_modalities)
