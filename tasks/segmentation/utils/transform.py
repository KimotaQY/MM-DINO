import random
import numpy as np
from PIL import Image, ImageFilter, ImageOps

from torchvision import transforms
import torchvision.transforms.functional as F

import cv2

# def normalize(img_a, img_b):
#     compose = transforms.Compose(
#         [
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#         ]
#     )
#     img_a = compose(img_a)
#     img_b = compose(img_b)

#     return img_a, img_b


def resize(img_a, mask=None, dsm=None, ratio_range=(0.5, 2.0)):
    w, h = img_a.shape[:2] if isinstance(img_a, np.ndarray) else img_a.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]),
                               int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    if isinstance(img_a, np.ndarray):
        img_a = cv2.resize(img_a, (ow, oh), interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
        if dsm is not None:
            dsm = cv2.resize(dsm, (ow, oh), interpolation=cv2.INTER_LINEAR)
    else:
        img_a = img_a.resize((ow, oh), Image.BILINEAR)
        if mask is not None:
            mask = mask.resize((ow, oh), Image.NEAREST)
        if dsm is not None:
            dsm = dsm.resize((ow, oh), Image.BILINEAR)

    return img_a, mask, dsm


def crop(img_a, mask, dsm, size, ignore_value=0):
    w, h = img_a.shape[:2] if isinstance(img_a, np.ndarray) else img_a.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0

    if isinstance(img_a, np.ndarray):
        if len(img_a.shape) == 3:  # 彩色图像
            img_a = np.pad(img_a, ((0, padh), (0, padw), (0, 0)),
                           mode='constant',
                           constant_values=0)
        else:  # 灰度图像
            img_a = np.pad(img_a, ((0, padh), (0, padw)),
                           mode='constant',
                           constant_values=0)

        if mask is not None:
            mask = np.pad(mask, ((0, padh), (0, padw)),
                          mode='constant',
                          constant_values=ignore_value)

        if dsm is not None:
            dsm = np.pad(dsm, ((0, padh), (0, padw)),
                         mode='constant',
                         constant_values=ignore_value)

        # 随机裁剪
        h, w = img_a.shape[:2]
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)

        img_a = img_a[y:y + size, x:x + size]
        if mask is not None:
            mask = mask[y:y + size, x:x + size]
        if dsm is not None:
            dsm = dsm[y:y + size, x:x + size]
    else:
        img_a = ImageOps.expand(img_a, border=(0, 0, padw, padh), fill=0)
        if mask is not None:
            mask = ImageOps.expand(mask,
                                   border=(0, 0, padw, padh),
                                   fill=ignore_value)
        if dsm is not None:
            dsm = ImageOps.expand(dsm,
                                  border=(0, 0, padw, padh),
                                  fill=ignore_value)

        w, h = img_a.size
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        img_a = img_a.crop((x, y, x + size, y + size))
        if mask is not None:
            mask = mask.crop((x, y, x + size, y + size))
        if dsm is not None:
            dsm = dsm.crop((x, y, x + size, y + size))

    return img_a, mask, dsm


def hflip(img_a, mask, dsm=None, p=0.5):
    if random.random() < p:
        if isinstance(img_a, np.ndarray):
            # numpy array水平翻转
            img_a = np.fliplr(img_a)
            if mask is not None:
                mask = np.fliplr(mask)
            if dsm is not None:
                dsm = np.fliplr(dsm)
        else:
            # PIL Image水平翻转
            img_a = transforms.functional.hflip(img_a)
            if mask is not None:
                mask = transforms.functional.hflip(mask)
            if dsm is not None:
                dsm = transforms.functional.hflip(dsm)

    if dsm is not None:
        return img_a, mask, dsm
    else:
        return img_a, mask, dsm


def vflip(img_a, mask, dsm=None, p=0.5):
    if random.random() < p:
        if isinstance(img_a, np.ndarray):
            # numpy array垂直翻转
            img_a = np.flipud(img_a)
            if mask is not None:
                mask = np.flipud(mask)
            if dsm is not None:
                dsm = np.flipud(dsm)
        else:
            # PIL Image垂直翻转
            img_a = transforms.functional.vflip(img_a)
            if mask is not None:
                mask = transforms.functional.vflip(mask)
            if dsm is not None:
                dsm = transforms.functional.vflip(dsm)

    if dsm is not None:
        return img_a, mask, dsm
    else:
        return img_a, mask, dsm


def rotate(img_a, mask, p=0.5):
    if random.random() < p:
        angle = random.randint(-90, 90)
        img_a = transforms.functional.rotate(img_a, angle)
        if mask is not None:
            mask = transforms.functional.rotate(mask, angle)
    return img_a, mask


def color_jitter(img_a, p=0.8):
    if random.random() < p:
        # cj = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)
        cj = transforms.ColorJitter(0.5, (0.8, 1.2), (0.8, 1.2), 0.25)
        img_a = cj(img_a)
    return img_a


def grayscale(img_a, p=0.2):
    num_output_channels, _, _ = F.get_dimensions(img_a)
    if random.random() < p:
        img_a = F.rgb_to_grayscale(img_a,
                                   num_output_channels=num_output_channels)
    return img_a


def blur(img_a, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img_a = img_a.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img_a


# def cutout(image_A,
#            image_B,
#            mask,
#            output_size,
#            scale=(0.08, 0.4),
#            ratio=(3 / 4, 4 / 3)):
#     original_width, original_height = image_A.size
#     area = original_width * original_height

#     # 随机选择裁剪区域的面积
#     target_area = random.uniform(scale[0], scale[1]) * area

#     # 随机选择裁剪区域的宽高比
#     aspect_ratio = random.uniform(ratio[0], ratio[1])

#     # 计算裁剪区域的宽度和高度
#     w = int(round((target_area * aspect_ratio)**0.5))
#     h = int(round((target_area / aspect_ratio)**0.5))

#     # 确保裁剪区域不超过原始图像的大小
#     if w > original_width or h > original_height:
#         return cutout(image_A, image_B, mask, output_size, scale, ratio)

#     # 随机选择裁剪区域的位置
#     x1 = random.randint(0, original_width - w)
#     y1 = random.randint(0, original_height - h)

#     # 裁剪图像
#     cropped_image_A = image_A.crop((x1, y1, x1 + w, y1 + h))
#     cropped_image_B = image_B.crop((x1, y1, x1 + w, y1 + h))
#     cropped_mask = mask.crop((x1, y1, x1 + w, y1 + h))

#     # 调整大小
#     resized_image_A = cropped_image_A.resize(output_size, Image.BILINEAR)
#     resized_image_B = cropped_image_B.resize(output_size, Image.BILINEAR)
#     resized_mask = cropped_mask.resize(output_size, Image.BILINEAR)

#     return resized_image_A, resized_image_B, resized_mask

# def cutmix(image_A,
#            image_B,
#            output_size,
#            scale=(0.08, 0.4),
#            ratio=(3 / 4, 4 / 3)):
#     original_width, original_height = image_A.size
#     area = original_width * original_height

#     # 随机选择裁剪区域的面积
#     target_area = random.uniform(scale[0], scale[1]) * area

#     # 随机选择裁剪区域的宽高比
#     aspect_ratio = random.uniform(ratio[0], ratio[1])

#     # 计算裁剪区域的宽度和高度
#     w = int(round((target_area * aspect_ratio)**0.5))
#     h = int(round((target_area / aspect_ratio)**0.5))

#     # 确保裁剪区域不超过原始图像的大小
#     if w > original_width or h > original_height:
#         return cutmix(image_A, image_B, output_size, scale, ratio)

#     # 随机选择裁剪区域的位置
#     x1 = random.randint(0, original_width - w)
#     y1 = random.randint(0, original_height - h)

#     # 创建一个与原始图像大小相同的空白图像
#     cutmix_image_A = image_A.copy()
#     cutmix_image_B = image_B.copy()

#     # 将image_A的裁剪区域替换为image_B的相应区域
#     cutmix_image_A.paste(image_B.crop((x1, y1, x1 + w, y1 + h)), (x1, y1))
#     # 将image_B的裁剪区域替换为image_A的相应区域
#     cutmix_image_B.paste(image_A.crop((x1, y1, x1 + w, y1 + h)), (x1, y1))

#     # 调整大小
#     resized_image_A = cutmix_image_A.resize(output_size, Image.BILINEAR)
#     resized_image_B = cutmix_image_B.resize(output_size, Image.BILINEAR)

#     return resized_image_A, resized_image_B
