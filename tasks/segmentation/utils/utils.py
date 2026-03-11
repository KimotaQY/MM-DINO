import os
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)  # 设置 Python 内部的随机种子
    np.random.seed(seed)  # 设置 NumPy 的随机种子
    torch.manual_seed(seed)  # 设置 PyTorch 的随机种子（CPU）
    torch.cuda.manual_seed(seed)  # 设置 PyTorch 的随机种子（单 GPU）
    torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU 设置所有 GPU 的随机种子

    # 确保 CuDNN 的确定性操作
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import time


def measure_model_performance(model, window_size, modality, batch_size=1):
    """测量模型的参数量、FLOPs、显存占用和FPS"""
    try:
        from thop import profile
        thop_available = True
    except ImportError:
        print("警告: thop库未安装，无法计算FLOPs")
        thop_available = False

    model.cuda()
    model.eval()
    device = next(model.parameters()).device

    # 创建测试输入
    if modality > 1:
        inputs = []
        for i in range(modality):
            input_tensor = torch.randn(batch_size, 3, window_size[0],
                                       window_size[1]).to(device)
            inputs.append(input_tensor)
    else:
        input_tensor = torch.randn(batch_size, 3, window_size[0],
                                   window_size[1]).to(device)
        inputs = (input_tensor, )

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)

    # 测试显存占用和FPS
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 预热
    with torch.no_grad():
        for _ in range(5):
            _ = model(*inputs)

    # 正式测试
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()  # 确保GPU操作完成
    start_memory = torch.cuda.max_memory_allocated()  # 获取当前最大显存占用

    start_time = time.time()
    with torch.no_grad():
        for _ in range(20):  # 运行20次取平均
            _ = model(*inputs)
            torch.cuda.synchronize()  # 确保GPU操作完成
    end_time = time.time()

    peak_memory = torch.cuda.max_memory_allocated()
    memory_usage = peak_memory - start_memory

    # 计算FPS
    total_time = end_time - start_time
    fps = 20 / total_time

    # 计算FLOPs
    flops = None
    if thop_available:
        try:
            flops, params = profile(model, inputs, verbose=False)
        except Exception as e:
            print(f"警告: 使用thop计算FLOPs时出错: {e}")
            flops = None

    # 对于PyTorch 2.0+版本，尝试使用flop_counter
    if flops is None and hasattr(torch, 'utils') and hasattr(
            torch.utils, 'flop_counter'):
        try:
            flop_count = torch.utils.flop_counter.flop_count(model, inputs)
            flops = flop_count.total()
            print(f"使用torch.utils.flop_counter计算FLOPs: {flops / 1e9:.2f}G")
        except Exception as e:
            print(f"使用torch.utils.flop_counter计算FLOPs时出错: {e}")

    # 输出结果
    print("=" * 50)
    print("模型性能指标:")
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")
    if flops:
        print(f"FLOPs: {flops / 1e9:.2f}G")
    print(f"显存占用: {peak_memory / 1024**2:.2f} MB")
    print(f"FPS: {fps:.2f}")
    print("=" * 50)

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "flops": flops,
        "memory_usage": memory_usage,
        "fps": fps
    }


def measure_training_time(model,
                          window_size,
                          modality,
                          batch_size=1,
                          num_steps=10):
    """测量模型训练时间"""
    model.cuda()
    device = next(model.parameters()).device

    # 创建训练输入和标签
    if modality > 1:
        inputs = []
        for i in range(modality):
            input_tensor = torch.randn(batch_size, 3, window_size[0],
                                       window_size[1]).to(device)
            inputs.append(input_tensor)
    else:
        input_tensor = torch.randn(batch_size, 3, window_size[0],
                                   window_size[1]).to(device)
        # 创建对应的标签 (batch_size, height, width)
        label_tensor = torch.randint(
            0, 6, (batch_size, window_size[0], window_size[1])).to(device)
        inputs = (input_tensor, )

    # 设置模型为训练模式
    model.train()

    # 创建简单的优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 预热几步
    for _ in range(3):
        outputs = model(*inputs)

        # 创建简单损失函数 (模拟分割任务)
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            loss = torch.mean(outputs[0])  # 假设第一个输出是分割结果
        else:
            loss = torch.mean(outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 正式测试训练时间
    start_time = time.time()

    for _ in range(num_steps):
        outputs = model(*inputs)

        # 创建简单损失函数 (模拟分割任务)
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            loss = torch.mean(outputs[0])  # 假设第一个输出是分割结果
        else:
            loss = torch.mean(outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    end_time = time.time()

    # 计算平均训练时间
    total_time = end_time - start_time
    avg_time_per_step = total_time / num_steps
    steps_per_second = num_steps / total_time

    # 输出结果
    print("=" * 50)
    print("模型训练时间指标:")
    print(f"总训练时间 ({num_steps} steps): {total_time:.4f} 秒")
    print(f"平均每个step时间: {avg_time_per_step:.4f} 秒")
    print(f"每秒处理steps数: {steps_per_second:.4f}")
    print("=" * 50)

    # 将模型恢复为评估模式
    model.eval()

    return {
        "total_training_time": total_time,
        "avg_time_per_step": avg_time_per_step,
        "steps_per_second": steps_per_second
    }


import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['axes.unicode_minus'] = False  # 处理坐标轴负号显示问题


def plot_confusion_matrix(cm, class_names, save_path):
    """绘制并保存混淆矩阵图像"""
    # 归一化混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import os

    # 尝试设置支持中文的字体
    try:
        # 定义常见的中文字体路径
        zh_font_paths = [
            # Noto Sans CJK 字体路径
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
            '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
            '/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc',
            # 文泉驿字体路径
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
            # 其他可能的路径
            '/System/Library/Fonts/Helvetica.ttc',  # macOS
            'C:/Windows/Fonts/simhei.ttf',  # Windows 黑体
            'C:/Windows/Fonts/msyh.ttc',  # Windows 微软雅黑
        ]

        # 首先尝试直接指定的字体路径
        font_found = False
        for font_path in zh_font_paths:
            if os.path.exists(font_path):
                font_prop = font_manager.FontProperties(fname=font_path)
                font_found = True
                break

        # 如果直接路径找不到，再尝试搜索字体名称
        if not font_found:
            # 搜索包含中文字符的字体
            zh_fonts = []
            for font in font_manager.fontManager.ttflist:
                font_name_lower = font.name.lower()
                # 检查是否包含中文字体关键词
                if any(keyword in font_name_lower for keyword in [
                        'noto', 'cjk', 'wqy', 'micro', 'hei', 'song', 'kai',
                        'sans', 'serif'
                ]):
                    # 额外验证字体是否支持中文字符（通过检查family name）
                    if any(chinese_indicator in font.name
                           for chinese_indicator in
                           ['SC', 'TC', 'JP', 'KR', 'HK', 'CJK', 'CHINESE']):
                        zh_fonts.append(font)
                    # 或者检查文件名是否包含中文指示符
                    elif any(indicator in font.fname.lower() for indicator in
                             ['chinese', 'cjk', 'noto', 'wqy']):
                        zh_fonts.append(font)

            if zh_fonts:
                # 使用找到的第一个中文字体
                font_prop = font_manager.FontProperties(
                    fname=zh_fonts[0].fname)
                font_found = True
            else:
                # 如果没找到特定的中文字体，尝试设置支持中文字体的通用属性
                plt.rcParams['font.sans-serif'] = [
                    'SimHei', 'DejaVu Sans', 'Liberation Sans',
                    'Noto Sans CJK SC'
                ]
                plt.rcParams[
                    'axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
                font_prop = font_manager.FontProperties()
    except:
        font_prop = font_manager.FontProperties()

    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制热力图
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # 设置标签
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)

    # 使用字体属性设置标签
    ax.set_xticklabels(class_names,
                       rotation=45,
                       ha='right',
                       fontproperties=font_prop)
    ax.set_yticklabels(class_names, fontproperties=font_prop)

    # 在每个格子中显示数值
    fmt = '.2f'
    thresh = cm_normalized.max() / 2.
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            ax.text(j,
                    i,
                    format(cm_normalized[i, j], fmt),
                    ha="center",
                    va="center",
                    fontproperties=font_prop,
                    color="white" if cm_normalized[i, j] > thresh else "black")

    # 设置标题和轴标签（使用字体属性）
    ax.set_title("归一化混淆矩阵", fontproperties=font_prop)
    ax.set_xlabel("预测标签", fontproperties=font_prop)
    ax.set_ylabel("真实标签", fontproperties=font_prop)

    # 自动调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


from PIL import Image


def save_prediction_as_image(pred, label, save_dir, index, dataset_name):
    """将预测结果保存为图像"""

    if dataset_name == "Vaihingen" or dataset_name == "Potsdam":
        palette = np.array(
            [
                [255, 255, 255],  # 0: Impervious surfaces (white)
                [0, 0, 255],  # 1: Buildings (blue)
                [0, 255, 255],  # 2: Low vegetation (cyan)
                [0, 255, 0],  # 3: Trees (green)
                [255, 255, 0],  # 4: Cars (yellow)
                [255, 0, 0],  # 5: Clutter (red)
                [0, 0, 0]  # 6: Undefined (black)
            ],
            dtype=np.uint8)
    elif dataset_name == "YYYJ":
        palette = np.array(
            [
                # 地基建设 → 棕色系
                [102, 50, 18],  # 0: 深褐色
                [175, 117, 71],  # 1: 赭石色
                [214, 171, 131],  # 2: 浅土黄色
                [231, 212, 190],  # 3: 极浅的米黄色

                # 施工道路 → 灰色系
                [64, 64, 64],  # 4: 深灰色
                [153, 153, 153],  # 5: 中灰色

                # 风电施工 → 蓝色系
                [18, 74, 143],  # 6: 深蓝色
                [125, 177, 230],  # 7: 天蓝色

                # 独立类别 - 高区分度配色
                [255, 191, 0],  # 8: 琥珀色/金黄色
                [54, 140, 48],  # 9: 深绿色
                [212, 50, 125],  # 10: 洋红色/品红色
                [128, 78, 191],  # 11: 中等深度的紫色
                [188, 155, 218],  # 12: 浅薰衣草紫
                [220, 60, 60],  # 13: 鲜红色
                [255, 255, 255],  # 14: 白色
                [0, 0, 0]  # 15: 黑色
            ],
            dtype=np.uint8)
    elif dataset_name == "WHU":
        palette = np.array(
            [
                [204, 102, 1],  # 0: farmland
                [255, 0, 0],  # 1: city
                [255, 255, 0],  # 2: village
                [0, 0, 255],  # 3: water
                [85, 166, 1],  # 4: forest
                [93, 255, 255],  # 5: road
                [152, 102, 153],  # 6: others
                [0, 0, 0]
            ],
            dtype=np.uint8)

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 应用颜色映射到预测结果
    pred_colored = palette[pred]
    label_colored = palette[label]

    # 转换为PIL图像并保存
    pred_img = Image.fromarray(pred_colored.squeeze().astype(np.uint8), 'RGB')
    label_img = Image.fromarray(label_colored.squeeze().astype(np.uint8),
                                'RGB')

    pred_img.save(os.path.join(save_dir, f"{index}_prediction.png"))
    label_img.save(os.path.join(save_dir, f"{index}_ground_truth.png"))
