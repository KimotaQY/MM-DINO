import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix


def CrossEntropy2d(input, target, weight=None, size_average=True):
    """ 2D version of the cross entropy loss """
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight, size_average)
    elif dim == 4:
        output = input.view(input.size(0), input.size(1), -1)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        return F.cross_entropy(output, target, weight, size_average)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))


def DiceLoss(inputs, targets):
    smooth = 1e-6
    # 将输入经过softmax处理
    inputs = torch.softmax(inputs, dim=1)

    # 转换为目标格式
    # 根据输入维度动态处理
    if inputs.dim() == 4:  # batch_size x channels x height x width
        targets_one_hot = F.one_hot(targets,
                                    num_classes=inputs.shape[1]).permute(
                                        0, 3, 1, 2).float()
    else:
        raise ValueError(f"Unsupported input dimensions: {inputs.dim()}")

    # 展平
    inputs = inputs.reshape(-1)
    targets_one_hot = targets_one_hot.reshape(-1)

    intersection = (inputs * targets_one_hot).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() +
                                           targets_one_hot.sum() + smooth)

    return 1 - dice


class DiceLossOptimized(nn.Module):
    """
    优化版本的Dice Loss，可以作为模块使用
    """

    def __init__(self, smooth=1e-6):
        super(DiceLossOptimized, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 对输入应用softmax
        inputs = torch.softmax(inputs, dim=1)

        # 获取类别数
        num_classes = inputs.shape[1]

        # 将目标转换为one-hot编码
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)

        # 调整维度: 从 [B, H, W, C] 转换为 [B, C, H, W]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # 展平输入和目标
        inputs = inputs.reshape(-1)
        targets_one_hot = targets_one_hot.reshape(-1)

        # 计算交集
        intersection = (inputs * targets_one_hot).sum()

        # 计算Dice系数
        dice = (2. * intersection + self.smooth) / (
            inputs.sum() + targets_one_hot.sum() + self.smooth)

        return 1 - dice


class CombinedLoss(nn.Module):
    """
    组合损失函数：结合交叉熵和Dice损失
    """

    def __init__(self, weight_ce=0.5, weight_dice=0.5, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.dice_loss = DiceLossOptimized(smooth=smooth)

    def forward(self, inputs, targets, weight=None):
        if inputs.dim() == 4:
            output = inputs.view(inputs.size(0), inputs.size(1), -1)
            output = torch.transpose(output, 1, 2).contiguous()
            output = output.view(-1, output.size(2))
            ce = F.cross_entropy(output,
                                 targets.view(-1),
                                 weight=weight,
                                 size_average=True)
        else:
            ce = F.cross_entropy(inputs,
                                 targets,
                                 weight=weight,
                                 size_average=True)

        dice = self.dice_loss(inputs, targets)
        # return self.weight_ce * ce + self.weight_dice * dice
        return ce, dice


def confusion_matrix_gpu(y_true, y_pred, labels=None, sample_weight=None):
    """
    GPU版本的混淆矩阵计算，基于sklearn的实现原理
    """
    # 转换为torch tensor并确保在GPU上
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true).long()
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred).long()

    # 确保在GPU上（如果有GPU）
    if torch.cuda.is_available():
        y_true = y_true.cuda()
        y_pred = y_pred.cuda()
        device = 'cuda'
    else:
        device = 'cpu'

    # 处理labels - 向量化方式
    if labels is None:
        # 使用torch.unique获取所有唯一标签
        all_labels = torch.cat([y_true, y_pred])
        labels = torch.unique(all_labels).sort()[0]
    else:
        labels = torch.as_tensor(labels, device=device).long()

    n_labels = len(labels)

    # 创建标签到索引的映射（向量化）
    # 使用searchsorted替代Python字典查找
    y_true_indices = torch.searchsorted(labels, y_true)
    y_pred_indices = torch.searchsorted(labels, y_pred)

    # 处理不在labels中的值（标记为n_labels）
    y_true_invalid = ~torch.isin(y_true, labels)
    y_pred_invalid = ~torch.isin(y_pred, labels)

    y_true_indices[y_true_invalid] = n_labels
    y_pred_indices[y_pred_invalid] = n_labels

    # 过滤有效索引
    valid_mask = (y_true_indices < n_labels) & (y_pred_indices < n_labels)
    y_true_filtered = y_true_indices[valid_mask]
    y_pred_filtered = y_pred_indices[valid_mask]

    # 处理sample_weight
    if sample_weight is None:
        weights = torch.ones(len(y_true_filtered),
                             dtype=torch.float32,
                             device=device)
    else:
        sample_weight = torch.as_tensor(sample_weight,
                                        dtype=torch.float32,
                                        device=device)
        weights = sample_weight[valid_mask]

    if len(y_true_filtered) == 0:
        return torch.zeros((n_labels, n_labels),
                           dtype=torch.float32,
                           device=device)

    # 核心优化：使用scatter_add_替代bincount（更高效）
    cm = torch.zeros((n_labels, n_labels), dtype=torch.float32, device=device)

    # 创建展平索引
    flat_indices = y_true_filtered * n_labels + y_pred_filtered

    # 使用scatter_add_累加权重
    cm_flat = torch.zeros(n_labels * n_labels,
                          dtype=torch.float32,
                          device=device)
    cm_flat.scatter_add_(0, flat_indices, weights)
    cm = cm_flat.reshape(n_labels, n_labels)

    return cm


def metrics(predictions, gts, label_values):
    logger = logging.getLogger("dinov3seg")

    cm = confusion_matrix(gts, predictions, labels=range(len(label_values)))
    # 使用GPU计算混淆矩阵
    # cm_tensor = confusion_matrix_gpu(gts,
    #                                  predictions,
    #                                  labels=range(len(label_values)))
    # cm = cm_tensor.cpu().numpy()  # 转回numpy用于后续计算

    logger.info("Confusion matrix :")
    print(cm)

    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    logger.info("%d pixels processed" % (total))
    logger.info("Total accuracy : %.2f" % (accuracy))

    Acc = np.diag(cm) / cm.sum(axis=1)
    for l_id, score in enumerate(Acc):
        logger.info("%s: %.4f" % (label_values[l_id], score))
    logger.info("---")

    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    logger.info("F1Score :")
    for l_id, score in enumerate(F1Score):
        logger.info("%s: %.4f" % (label_values[l_id], score))
    if "undefined" in label_values or "clutter" in label_values:
        F1Score = np.nanmean(F1Score[:(len(label_values) - 1)])
    else:
        F1Score = np.nanmean(F1Score[:(len(label_values))])
    logger.info('mean F1Score: %.4f' % (F1Score))
    logger.info("---")

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    logger.info("Kappa: %.4f" % (kappa))

    # Compute MIoU coefficient
    MIoU = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) -
                          np.diag(cm))
    print(MIoU)
    logger.info("MIoU: ")
    for l_id, score in enumerate(MIoU):
        logger.info("%s: %.4f" % (label_values[l_id], score))
    if "undefined" in label_values or "clutter" in label_values:
        MIoU = np.nanmean(MIoU[:(len(label_values) - 1)])
    else:
        MIoU = np.nanmean(MIoU[:(len(label_values))])
    logger.info('mean MIoU: %.4f' % (MIoU))
    logger.info("---")

    return MIoU, F1Score, kappa, accuracy


def metrics_print_version(predictions, gts, label_values):
    cm = confusion_matrix(gts, predictions, labels=range(len(label_values)))

    print("Confusion matrix :")
    print(cm)

    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("%d pixels processed" % (total))
    print("Total accuracy : %.2f" % (accuracy))

    Acc = np.diag(cm) / cm.sum(axis=1)
    for l_id, score in enumerate(Acc):
        print("%s: %.4f" % (label_values[l_id], score))
    print("---")

    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("%s: %.4f" % (label_values[l_id], score))
    if "undefined" in label_values or "clutter" in label_values:
        F1Score = np.nanmean(F1Score[:(len(label_values) - 1)])
    else:
        F1Score = np.nanmean(F1Score[:(len(label_values))])
    print('mean F1Score: %.4f' % (F1Score))
    print("---")

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    print("Kappa: %.4f" % (kappa))

    # Compute MIoU coefficient
    MIoU = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) -
                          np.diag(cm))
    print(MIoU)
    print("MIoU: ")
    for l_id, score in enumerate(MIoU):
        print("%s: %.4f" % (label_values[l_id], score))
    if "undefined" in label_values or "clutter" in label_values:
        MIoU = np.nanmean(MIoU[:(len(label_values) - 1)])
    else:
        MIoU = np.nanmean(MIoU[:(len(label_values))])
    print('mean MIoU: %.4f' % (MIoU))
    print("---")

    return MIoU, F1Score, kappa, accuracy, cm
