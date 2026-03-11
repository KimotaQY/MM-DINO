import os
import random
import sys
import numpy as np
import torch
from PIL import Image
from PIL import ImageOps
import torchvision.transforms.functional as TF

from utils.transform import *

palette = {
    # 【地基建设】→ 棕色系
    0: (102, 50, 18),  # 深褐色
    1: (175, 117, 71),  # 赭石色
    2: (214, 171, 131),  # 浅土黄色
    3: (231, 212, 190),  # 极浅的米黄色

    # 【施工道路】→ 灰色系
    4: (64, 64, 64),  # 深灰色
    5: (153, 153, 153),  # 中灰色

    # 【风电施工】→ 蓝色系
    6: (18, 74, 143),  # 深蓝色
    7: (125, 177, 230),  # 天蓝色

    # 【独立类别 - 高区分度配色】
    8: (255, 191, 0),  # 琥珀色/金黄色
    9: (54, 140, 48),  # 深绿色
    10: (212, 50, 125),  # 洋红色/品红色
    11: (128, 78, 191),  # 中等深度的紫色
    12: (188, 155, 218),  # 浅薰衣草紫
    13: (220, 60, 60),  # 鲜红色
    14: (255, 255, 255)  # 白色
}

invert_palette = {v: k for k, v in palette.items()}


class YYYJ_Dataset(torch.utils.data.Dataset):

    def __init__(self,
                 ids,
                 data_dir,
                 label_dir,
                 data_type,
                 window_size=(224, 224),
                 normalize_type=None,
                 cache_size=500):
        super(YYYJ_Dataset, self).__init__()

        self.data_type = data_type
        self.window_size = window_size
        self.cache_size = cache_size  # 限制缓存数量

        # List of files
        self.data_files = [data_dir.format(id) for id in ids]
        self.label_files = [label_dir.format(id) for id in ids]

        # Sanity check : raise an error if some files do not exist
        for file in self.data_files + self.label_files:
            if not os.path.exists(file) and not os.path.isfile(file):
                raise ValueError(f"File {file} does not exist")

        # Initialize cache dicts
        self.data_cache = {}
        self.label_cache = {}

        if normalize_type == "geo":
            self.imagenet_mean = (0.430, 0.411, 0.296)
            self.imagenet_std = (0.213, 0.156, 0.143)
        elif normalize_type == "common":
            self.imagenet_mean = (0.485, 0.456, 0.406)
            self.imagenet_std = (0.229, 0.224, 0.225)
        else:
            self.imagenet_mean = None
            self.imagenet_std = None

    def __len__(self):
        interval_num = (256**2 / self.window_size[0]**2) * 1000  # 256尺寸时为*1000
        data_len = len(self.data_files
                       ) * interval_num if self.data_type == 'train' else len(
                           self.data_files)
        return int(data_len)

    def __getitem__(self, idx):
        if self.data_type == 'train':
            random_idx = random.randint(0, len(self.data_files) - 1)

            # 限制缓存大小，防止内存溢出
            if random_idx not in self.data_cache.keys():
                if len(self.data_cache) >= self.cache_size:
                    # 移除最早的缓存项
                    oldest_key = next(iter(self.data_cache))
                    del self.data_cache[oldest_key]
                    if oldest_key in self.label_cache:
                        del self.label_cache[oldest_key]

                data = Image.open(self.data_files[random_idx]).convert('RGB')
                label = Image.open(self.label_files[random_idx]).convert('RGB')

                # 缓存图像
                self.data_cache[random_idx] = data
                self.label_cache[random_idx] = label
            else:
                data = self.data_cache[random_idx]
                label = self.label_cache[random_idx]

            # 检查图像尺寸是否小于窗口尺寸，如果是则使用padding
            img_width, img_height = data.size
            pad_width = max(0, self.window_size[0] - img_width)
            pad_height = max(0, self.window_size[1] - img_height)

            if pad_width > 0 or pad_height > 0:
                # 使用PIL的expand方法进行padding
                data = ImageOps.expand(data,
                                       border=(0, 0, pad_width, pad_height),
                                       fill=(0, 0, 0))
                label = ImageOps.expand(label,
                                        border=(0, 0, pad_width, pad_height),
                                        fill=(255, 255, 255))

            # Get a random patch
            x1, x2, y1, y2 = self.get_random_pos(data, self.window_size)
            if isinstance(data, np.ndarray):
                data = data[:, x1:x2, y1:y2]
                label = label[x1:x2, y1:y2]
            elif isinstance(data, Image.Image):
                data = data.crop(
                    (y1, x1, y2, x2))  # PIL使用(left, upper, right, lower)
                label = label.crop((y1, x1, y2, x2))

            # 弱增强
            # data, label = resize(data, label, (0.5, 2.0))
            # data, label = crop(data, label, self.window_size[0])
            data, label = hflip(data, label, p=0.5)
            data, label = vflip(data, label, p=0.5)
            # data, label = rotate(data, label, p=0.5)

            # data = color_jitter(data, p=0.8)
            # data = grayscale(data, p=0.2)
            # data = blur(data, p=0.5)

            # convert to np.array
            # data = np.array(data, dtype='float32').transpose((2, 0, 1))
            label = np.array(label)
            label = np.asarray(self.convert_from_color(label), dtype='int64')
        else:
            data = Image.open(self.data_files[idx]).convert('RGB')
            # data = np.array(data, dtype='float32').transpose((2, 0, 1))

            label = Image.open(self.label_files[idx]).convert('RGB')

            # 检查图像尺寸是否小于窗口尺寸，如果是则使用padding
            img_width, img_height = data.size
            pad_width = max(0, self.window_size[0] - img_width)
            pad_height = max(0, self.window_size[1] - img_height)

            if pad_width > 0 or pad_height > 0:
                # 使用PIL的expand方法进行padding
                data = ImageOps.expand(data,
                                       border=(0, 0, pad_width, pad_height),
                                       fill=(0, 0, 0))
                label = ImageOps.expand(label,
                                        border=(0, 0, pad_width, pad_height),
                                        fill=(255, 255, 255))

            label_arr = np.array(label)
            label = np.asarray(self.convert_from_color(label_arr),
                               dtype='int64')

        data = TF.to_tensor(data)  # Convert image to tensor
        if self.imagenet_mean is not None:
            data = TF.normalize(
                data, self.imagenet_mean,
                self.imagenet_std)  # Normalize with ImageNet mean and std

        return data, label

    @staticmethod
    def convert_from_color(arr_3d, palette=invert_palette):
        """ RGB-color encoding to grayscale labels """
        arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

        for c, i in palette.items():
            m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
            arr_2d[m] = i

        return arr_2d

    @staticmethod
    def get_random_pos(img, window_shape):
        """ Extract of 2D random patch of shape window_shape in the image """
        w, h = window_shape
        if isinstance(img, np.ndarray):
            W, H = img.shape[-2:]
        elif isinstance(img, Image.Image):
            W, H = img.size

        # 确保不会出现负的随机数范围
        x1 = random.randint(0, max(0, W - w)) if W > w else 0
        x2 = x1 + min(w, W)  # 确保不超出边界
        y1 = random.randint(0, max(0, H - h)) if H > h else 0
        y2 = y1 + min(h, H)  # 确保不超出边界
        return x1, x2, y1, y2
