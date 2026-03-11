import os
import random
import sys
import numpy as np
import torch
from PIL import Image
from skimage.io import imread
import torchvision.transforms.functional as TF
from collections import OrderedDict

deps_path = os.path.join(os.path.dirname(__file__), "task/segmentation")
sys.path.insert(0, deps_path)
from utils.transform import *

palette = {
    0: (255, 255, 255),  # Background (white)
    1: (255, 0, 0),  # 农田 (red)
    2: (255, 255, 0),  # 城市 (yellow)
    3: (0, 0, 255),  # 村庄 (blue)
    4: (159, 129, 183),  # 水体 (purple)
    5: (0, 255, 0),  # 森林 (green)
    6: (255, 195, 128),  # 道路 (deeper yellow)
    7: (165, 0, 165),  # 其他 (deeper purple)
    8: (0, 0, 0)
}  # Undefined (black)

invert_palette = {v: k for k, v in palette.items()}


class LRUCache:
    """LRU缓存实现"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key in self.cache:
            # 移动到最前面（最近使用）
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            # 更新值并移动到最前面
            self.cache.move_to_end(key)
        self.cache[key] = value
        # 如果超出容量，删除最久未使用的项
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class WHU_Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        filenames,
        rgb_dir,
        label_dir,
        data_type,
        window_size=(224, 224),
        normalize_type=None,
        sar_dir=None,
        cache_size=100,
    ):
        super(WHU_Dataset, self).__init__()

        self.data_type = data_type
        self.window_size = window_size
        self.cache_size = cache_size

        # List of files
        self.rgb_files = []
        self.sar_files = []
        self.label_files = []

        self.rgb_files = [rgb_dir.format(filename) for filename in filenames]
        self.label_files = [
            label_dir.format(filename) for filename in filenames
        ]
        self.sar_files = [sar_dir.format(filename) for filename in filenames
                          ] if sar_dir is not None else []

        # Sanity check : raise an error if some files do not exist
        for file in self.rgb_files + self.label_files + self.sar_files:
            if not os.path.exists(file) and not os.path.isfile(file):
                raise ValueError(f"File {file} does not exist")

        # 初始化LRU缓存
        self.rgb_cache = LRUCache(cache_size)
        self.label_cache = LRUCache(cache_size)
        self.sar_cache = LRUCache(cache_size)

        if normalize_type == "geo":
            # self.imagenet_mean = (0.430, 0.411, 0.296)
            # self.imagenet_std = (0.213, 0.156, 0.143)
            self.imagenet_mean = (0.485, 0.456, 0.406)
            self.imagenet_std = (0.229, 0.224, 0.225)
        elif normalize_type == "common":
            self.imagenet_mean = (0.485, 0.456, 0.406)
            self.imagenet_std = (0.229, 0.224, 0.225)
        else:
            self.imagenet_mean = None
            self.imagenet_std = None

    def __len__(self):
        interval_num = (256**2 / self.window_size[0]**2) * 160  # 256尺寸时为*16
        data_len = len(self.rgb_files
                       ) * interval_num if self.data_type == 'train' else len(
                           self.rgb_files)
        return int(data_len)

    def __getitem__(self, idx):
        if self.data_type == 'train':
            random_idx = random.randint(0, len(self.rgb_files) - 1)

            # 使用LRU缓存
            cached_data = self.rgb_cache.get(random_idx)
            if cached_data is not None:
                data = cached_data
            else:
                data = imread(self.rgb_files[random_idx])
                if len(data.shape) == 3 and data.shape[2] >= 3:
                    data = data[:, :, :3]
                self.rgb_cache.put(random_idx, data)

            cached_label = self.label_cache.get(random_idx)
            if cached_label is not None:
                label = cached_label
            else:
                label = imread(self.label_files[random_idx]).astype(np.int32)
                label = label / 10 - 1
                label[label == -1] = 7
                label = label.astype(np.int32)
                self.label_cache.put(random_idx, label)

            sar = None
            cached_sar = self.sar_cache.get(random_idx)
            if cached_sar is not None:
                sar = cached_sar
            elif len(self.sar_files) > 0:
                sar = imread(self.sar_files[random_idx])
                self.sar_cache.put(random_idx, sar)

            # Get a random patch
            x1, x2, y1, y2 = self.get_random_pos(data, self.window_size)
            if isinstance(data, np.ndarray):
                data = data[x1:x2, y1:y2, :]
                label = label[x1:x2, y1:y2]
                sar = sar[x1:x2, y1:y2] if sar is not None else None
            elif isinstance(data, Image.Image):
                data = data.crop(
                    (y1, x1, y2, x2))  # PIL使用(left, upper, right, lower)
                label = label.crop((y1, x1, y2, x2))
                sar = sar.crop((y1, x1, y2, x2)) if sar is not None else None

            # 弱增强
            data, label, sar = resize(data, label, sar, ratio_range=(0.5, 2.0))
            data, label, sar = crop(data, label, sar, size=self.window_size[0])
            data, label, sar = hflip(data, label, sar, p=0.5)
            data, label, sar = vflip(data, label, sar, p=0.5)
            # data, label = rotate(data, label, p=0.5)

            # data = color_jitter(data, p=0.8)
            # data = grayscale(data, p=0.2)
            # data = blur(data, p=0.5)
        else:
            data = imread(self.rgb_files[idx])
            if len(data.shape) == 3 and data.shape[2] >= 3:
                data = data[:, :, :3]

            label = imread(self.label_files[idx]).astype(np.int32)
            label = label / 10 - 1
            label[label == -1] = 7
            label = label.astype(np.int32)

            sar = imread(self.sar_files[idx]) if len(
                self.sar_files) > 0 else None

        # 最终转换为tensor前确保数据连续且格式正确
        if isinstance(data, np.ndarray):
            # 确保数据连续且转换为(C,H,W)格式用于tensor转换
            data = np.ascontiguousarray(data)
            label = np.ascontiguousarray(label)

        data = TF.to_tensor(data)  # Convert image to tensor
        if self.imagenet_mean is not None:
            data = TF.normalize(
                data, self.imagenet_mean,
                self.imagenet_std)  # Normalize with ImageNet mean and std

        if sar is not None:
            if isinstance(sar, np.ndarray):
                sar = sar / 255.0
                sar = sar.astype(np.float32)
                sar = np.ascontiguousarray(sar)
            sar = TF.to_tensor(sar)
            # sar = TF.normalize(sar, self.sar_mean, self.sar_std)
            return data, sar, label
        else:
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
            W, H = img.shape[:2]
        elif isinstance(img, Image.Image):
            W, H = img.size

        x1 = random.randint(0, W - w - 1) if W - w - 1 > 0 else 0
        x2 = x1 + w
        y1 = random.randint(0, H - h - 1) if H - h - 1 > 0 else 0
        y2 = y1 + h
        return x1, x2, y1, y2
