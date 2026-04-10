import json
import logging
import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime

# 添加项目根目录到 Python 路径中，以便可以导入 dinov3 模块
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dinov3.logging import setup_logging
# 导入分布式训练相关模块
import dinov3.distributed as distributed

deps_path = os.path.join(os.path.dirname(__file__), "task/segmentation")
if deps_path not in sys.path:
    sys.path.insert(0, deps_path)
from utils.metrics import metrics
from utils.inference import slide_inference
from utils.utils import set_seed
from utils.move_files import move_files
from utils.clean_logs import clean_logs

from configs import get_cfg
from configs.common_cfg import MS_ROOT_DIR

from datasets import build_dataset

DATASET_NAME = ""
MODEL_NAME = ""
NUM_MODALITIES = -1


def get_local_rank():
    """获取本地rank"""
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    else:
        return 0


def setup_nccl_environment():
    """设置NCCL环境变量以提高稳定性"""
    # 增加NCCL超时时间
    os.environ['NCCL_TIMEOUT'] = '1200'  # 20分钟
    # os.environ['NCCL_BLOCKING_WAIT'] = '1'  # 启用阻塞等待
    # os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'  # 启用异步错误处理
    # os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'  # PyTorch 2.2+版本

    # # 设置NCCL通信参数
    # os.environ['NCCL_DEBUG'] = 'INFO'  # 调试信息
    # os.environ['NCCL_SOCKET_IFNAME'] = 'lo'  # 使用回环接口（单机多卡）

    # # 减少NCCL操作的并发性以提高稳定性
    # os.environ['NCCL_P2P_LEVEL'] = 'LOC'  # 限制P2P通信级别
    # os.environ['NCCL_SHM_DISABLE'] = '1'  # 禁用共享内存

    print("NCCL环境变量已设置完成")


def main(**kwargs):
    # 在初始化分布式训练前设置NCCL环境
    setup_nccl_environment()

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
    cfg = get_cfg(MODEL_NAME, DATASET_NAME, **kwargs)
    window_size = cfg.get('window_size')
    batch_size = cfg.get('batch_size')
    model = cfg.get('model')
    optimizer = cfg.get('optimizer')
    scheduler = cfg.get('scheduler')

    set_seed(42)
    train_dataset = build_dataset(
        DATASET_NAME,
        "train",
        window_size=window_size,
        model_name=MODEL_NAME,
        modality="multi" if NUM_MODALITIES > 1 else None,
        backbone_type=kwargs.get('backbone_type'),
    )

    # 根据分布式训练设置调整采样器
    if distributed.is_enabled():
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        batch_size = batch_size  # 保持batch size不变
        shuffle = False  # 使用sampler时需要设置为False
    else:
        train_sampler = None
        batch_size = batch_size
        shuffle = True

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle,
                                               sampler=train_sampler,
                                               num_workers=4,
                                               pin_memory=False,
                                               persistent_workers=True)

    test_dataset = build_dataset(
        DATASET_NAME,
        "test",
        window_size=window_size,
        model_name=MODEL_NAME,
        modality="multi" if NUM_MODALITIES > 1 else None,
        backbone_type=kwargs.get('backbone_type'),
    )

    if distributed.is_enabled():
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, shuffle=False)
    else:
        test_sampler = None

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              sampler=test_sampler,
                                              num_workers=1,
                                              persistent_workers=True)

    # 将模型移到GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

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

    # 创建日志目录
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    src_dict = f"{MS_ROOT_DIR}/SS-projects/dinov3/tasks/segmentation"
    dst_dict = f"{src_dict}/logs/{MODEL_NAME}/{DATASET_NAME}_{date_time}"
    detection_log_dir = os.path.join(f"{src_dict}/logs", f"{MODEL_NAME}")

    # 只在主进程上创建目录和保存文件
    if distributed.is_main_process():
        clean_logs(detection_log_dir, 2)
        print(f"正在将文件移动到 {dst_dict}...")
        move_files(src_dict, os.path.join(dst_dict, 'proj_files'),
                   ['logs', '__pycache__', '.pyc'])
        print("=====文件移动完成=====")
        # 初始化日志系统
        setup_logging(output=dst_dict, level=logging.INFO, name='dinov3seg')

    # 打印所有可学习参数
    # if distributed.is_main_process():
    #     print("Trainable parameters:")
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             print(f"  {name}: {param.shape}")

    train(model,
          train_loader,
          test_loader,
          optimizer,
          scheduler,
          save_dir=dst_dict if distributed.is_main_process() else None,
          cfg=cfg)
    # test(model, test_loader, cfg)


def train(model,
          train_loader,
          test_loader,
          optimizer,
          scheduler,
          save_dir,
          cfg=None):
    logger = logging.getLogger("dinov3seg")
    epochs = cfg.get("epochs")
    best_IoU = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = cfg.get("loss_fn")

    # 初始化用于记录训练和测试指标的文件
    train_metrics_file = None
    test_metrics_file = None
    if save_dir is not None and distributed.is_main_process():
        train_metrics_file = os.path.join(save_dir, "train_metrics.json")
        test_metrics_file = os.path.join(save_dir, "test_metrics.json")

        # 初始化空的JSON文件
        with open(train_metrics_file, 'w') as f:
            f.write("[\n")

        with open(test_metrics_file, 'w') as f:
            f.write("[\n")

    for e in range(1, epochs + 1):
        model.train()

        # 在分布式训练中设置采样器
        if distributed.is_enabled():
            train_loader.sampler.set_epoch(e)

        total_loss = 0.0
        num_batches = 0

        iterations = tqdm(train_loader,
                          disable=not distributed.is_main_process())
        for batch in iterations:
            if NUM_MODALITIES > 1:
                input, dsm, label = batch
                input, dsm, label = input.to(device), dsm.to(device), label.to(
                    device)
                optimizer.zero_grad()
                logits = model(input, dsm)
            else:
                input, label = batch
                input, label = input.to(device), label.to(device)
                optimizer.zero_grad()
                logits = model(input)

            if MODEL_NAME == 'MultiSenseSeg':
                loss = loss_fn(logits, label, e)
            else:
                loss = loss_fn(logits, label)

            # 添加调试信息来帮助定位问题
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN or Inf detected in loss:")
                print(f"logits shape: {logits.shape}")
                print(f"label shape: {label.shape}")
                print(f"label min: {label.min()}, label max: {label.max()}")
                print(f"unique labels: {torch.unique(label)}")
                logger.error(
                    f"logits min: {logits.min()}, logits max: {logits.max()}")
                logger.error(
                    f"logits mean: {logits.mean()}, logits std: {logits.std()}"
                )
                sys.exit(1)

            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.data
            num_batches += 1

            if distributed.is_main_process():
                iterations.set_description("Epoch: {}/{} Loss: {:.4f}".format(
                    e, epochs, loss.data))

        # 计算并打印epoch的平均loss
        avg_loss = total_loss / num_batches

        # 在分布式训练中同步损失
        if distributed.is_enabled():
            avg_loss_tensor = torch.tensor(
                avg_loss, device=device) if not isinstance(
                    avg_loss,
                    torch.Tensor) else avg_loss.detach().clone().to(device)
            torch.distributed.all_reduce(avg_loss_tensor)
            avg_loss_tensor /= distributed.get_world_size()
            avg_loss = avg_loss_tensor.item()

        if distributed.is_main_process():
            logger.info(f"Epoch {e}/{epochs} - Average Loss: {avg_loss:.4f}")

            # 记录训练指标到JSON文件
            if train_metrics_file is not None:
                train_record = {"epoch": e, "avg_loss": float(avg_loss)}

                # 添加逗号（如果不是第一条记录）
                if e > 1:
                    with open(train_metrics_file, 'a') as f:
                        f.write(",\n")

                with open(train_metrics_file, 'a') as f:
                    json.dump(train_record, f, indent=2)

        if scheduler is not None:
            scheduler.step()

        # 每隔{save_interval}个epoch保存一次模型
        save_interval = 5
        if e % save_interval == 0:
            # 在所有进程间设置屏障,确保同步
            if distributed.is_enabled():
                torch.distributed.barrier()

            test_metrics = test(model,
                                test_loader,
                                cfg=cfg,
                                is_distributed=distributed.is_enabled())

            if distributed.is_main_process():
                if isinstance(test_metrics, dict):
                    mIoU = test_metrics.get('MIoU', 0.0)

                if mIoU > best_IoU:
                    best_IoU = mIoU
                    # 保存模型时考虑分布式包装
                    model_state = model.module.state_dict() if hasattr(
                        model, 'module') else model.state_dict()
                    torch.save({
                        "model": model_state
                    }, f"{save_dir}/{MODEL_NAME}_{DATASET_NAME}_e{e}_mIoU{round(mIoU*100, 2)}.pth"
                               )

                # 记录测试指标到JSON文件
                if test_metrics_file is not None and isinstance(
                        test_metrics, dict):
                    test_record = test_metrics.copy()
                    test_record["epoch"] = e

                    # 添加逗号（如果不是第一条记录）
                    with open(test_metrics_file, 'a') as f:
                        if e > save_interval:  # 第一条记录是第{save_interval}个epoch
                            f.write(",\n")
                        json.dump(test_record, f, indent=2)

                # 清理多余的 .pth 文件
                if save_dir is not None:
                    model_files = [
                        f for f in os.listdir(save_dir) if f.endswith(".pth")
                    ]
                    if len(model_files) > 5:  # 设置最大保留的模型数量
                        # 按文件创建时间排序，保留最新的 5 个模型
                        model_files.sort(key=lambda x: os.path.getmtime(
                            os.path.join(save_dir, x)))
                        for file_name in model_files[:-5]:
                            os.remove(os.path.join(save_dir, file_name))
                            print(f"Deleted old model: {file_name}")

            # 再次设置屏障,确保所有进程等待主进程完成保存
            if distributed.is_enabled():
                torch.distributed.barrier()

        # 保存检查点
        if save_dir is not None and distributed.is_main_process():
            model_path = save_dir + "/" + DATASET_NAME + "_checkpoint.pth"
            # 保存模型时考虑分布式包装
            model_state = model.module.state_dict() if hasattr(
                model, 'module') else model.state_dict()
            torch.save(
                {
                    "model": model_state,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": e,
                },
                model_path,
            )

    # 完成训练后关闭JSON数组
    if save_dir is not None and distributed.is_main_process():
        if train_metrics_file is not None:
            with open(train_metrics_file, 'a') as f:
                f.write("\n]")

        if test_metrics_file is not None:
            with open(test_metrics_file, 'a') as f:
                f.write("\n]")


def test(model, test_loader, cfg, is_distributed=False):
    # 清理缓存
    torch.cuda.empty_cache()
    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    preds = []
    labels = []

    window_size = cfg.get("window_size")
    classes = cfg.get("labels")

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
                                       batch_size=cfg.get("batch_size", 4) * 4)
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
                                       batch_size=cfg.get("batch_size", 4) * 4)

        pred = np.argmax(pred, axis=1)
        preds.append(pred)
        labels.append(label)

    # 如果是分布式训练，需要收集所有进程的预测结果
    if is_distributed:
        # 将当前进程的preds和labels转换为tensor并填充到相同长度
        local_preds = np.concatenate([p.ravel() for p in preds
                                      ]) if preds else np.array([])
        local_labels = np.concatenate([p.ravel() for p in labels
                                       ]) if labels else np.array([])

        # 获取所有进程的数据大小
        local_size = torch.tensor([len(local_preds)],
                                  device=device,
                                  dtype=torch.long)
        all_sizes = [
            torch.zeros_like(local_size)
            for _ in range(distributed.get_world_size())
        ]
        torch.distributed.all_gather(all_sizes, local_size)

        max_size = max(size.item() for size in all_sizes)

        # 填充到最大长度
        padded_preds = np.zeros(max_size, dtype=local_preds.dtype)
        padded_labels = np.zeros(max_size, dtype=local_labels.dtype)
        padded_preds[:len(local_preds)] = local_preds
        padded_labels[:len(local_labels)] = local_labels

        # 转换为tensor
        preds_tensor = torch.from_numpy(padded_preds).to(device)
        labels_tensor = torch.from_numpy(padded_labels).to(device)

        # 收集所有进程的结果
        all_preds_list = [
            torch.zeros_like(preds_tensor)
            for _ in range(distributed.get_world_size())
        ]
        all_labels_list = [
            torch.zeros_like(labels_tensor)
            for _ in range(distributed.get_world_size())
        ]

        torch.distributed.all_gather(all_preds_list, preds_tensor)
        torch.distributed.all_gather(all_labels_list, labels_tensor)

        # 合并并去除填充部分
        all_preds = []
        all_labels = []
        for i, size in enumerate(all_sizes):
            actual_size = size.item()
            if actual_size > 0:
                all_preds.append(all_preds_list[i][:actual_size].cpu().numpy())
                all_labels.append(
                    all_labels_list[i][:actual_size].cpu().numpy())

        if all_preds:
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
        else:
            all_preds = np.array([])
            all_labels = np.array([])

        # 只在主进程计算指标
        if distributed.is_main_process():
            MIoU, F1, Kappa, Acc = metrics(all_preds, all_labels, classes)
        else:
            MIoU, F1, Kappa, Acc = 0.0, 0.0, 0.0, 0.0

        # 将指标广播到所有进程
        metrics_tensor = torch.tensor([MIoU, F1, Kappa, Acc], device=device)
        torch.distributed.broadcast(metrics_tensor, src=0)
        MIoU, F1, Kappa, Acc = metrics_tensor.tolist()
    else:
        # 非分布式情况，直接计算
        MIoU, F1, Kappa, Acc = metrics(
            np.concatenate([p.ravel() for p in preds]),
            np.concatenate([p.ravel() for p in labels]).ravel(), classes)

    # 构建详细指标字典，并转换为Python原生类型以支持JSON序列化
    detailed_metrics = {
        "MIoU": float(MIoU),  # 转换为Python float
        "F1": float(F1),  # 转换为Python float
        "Kappa": float(Kappa),  # 转换为Python float
        "Acc": float(Acc)  # 转换为Python float
    }

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
    args = parser.parse_args()

    # 如果提供了模型名称参数，使用它；否则使用默认值
    if args.model_name:
        MODEL_NAME = args.model_name
    if args.num_modalities:
        NUM_MODALITIES = args.num_modalities
    if args.dataset_name:
        DATASET_NAME = args.dataset_name

    main(num_modalities=NUM_MODALITIES,
         use_lora=args.use_lora,
         r=args.r,
         backbone_type=args.backbone_type)
