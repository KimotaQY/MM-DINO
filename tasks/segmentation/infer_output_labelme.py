import json
import os
import sys
import cv2
import numpy as np
import torch

# 添加项目根目录到 Python 路径中，以便可以导入 dinov3 模块
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import dinov3.distributed as distributed

from utils.metrics import metrics_print_version as metrics
from utils.inference import slide_inference

# 添加GeoTIFF处理和WKT生成所需库
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from shapely import wkt
import json

# 添加geopandas用于shapefile支持
try:
    import geopandas as gpd
    from shapely.geometry import Polygon
    SHAPEFILE_SUPPORT = True
except ImportError:
    SHAPEFILE_SUPPORT = False
    print(
        "Warning: geopandas not installed. Shapefile export will not be available."
    )

from configs import get_cfg

DATASET_NAME = "Vaihingen"
MODEL_NAME = "DINOv3"
MODALITY = "uni"
N_CLASSES = None


def get_local_rank():
    """获取本地rank"""
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    else:
        return 0


def save_labelme_format(pred_class, image_path, output_json_path, labels_map):
    """
    将分割结果保存为labelme的JSON格式
    
    Args:
        pred_class: 分割结果数组 (H, W)，每个像素值代表类别ID
        image_path: 原始图像路径
        output_json_path: 输出的JSON文件路径
        labels_map: 类别ID到标签名称的映射字典
    """
    import base64
    from PIL import Image

    # 读取原始图像以获取图像尺寸和路径信息
    with Image.open(image_path) as img:
        image_width, image_height = img.size
        image_format = img.format

    # 读取图像数据并进行base64编码
    with open(image_path, "rb") as img_file:
        image_data = base64.b64encode(img_file.read()).decode('utf-8')

    # 初始化labelme JSON结构
    labelme_json = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(image_path),
        "imageData": image_data,
        "imageHeight": image_height,
        "imageWidth": image_width
    }

    # 为每个类别生成多边形
    unique_classes = np.unique(pred_class)

    for class_id in unique_classes:
        class_id = int(class_id)

        # 跳过背景类（如果需要的话）
        if class_id == 0:
            continue

        # 获取当前类别的掩码
        class_mask = (pred_class == class_id).astype(np.uint8)

        # 使用cv2找到轮廓
        contours, _ = cv2.findContours(
            class_mask,
            mode=cv2.RETR_EXTERNAL,  # 只检测外轮廓
            method=cv2.CHAIN_APPROX_SIMPLE  # 压缩水平、垂直和对角线方向的元素，只保留端点
        )

        # 为每个轮廓创建一个shape
        for contour in contours:
            # 轮廓可能是n x 1 x 2的形状，需要重塑为n x 2
            if len(contour.shape) == 3:
                contour = contour.reshape(-1, 2)

            # 转换为labelme格式的points
            points = []
            for point in contour:
                x, y = float(point[0]), float(point[1])
                points.append([x, y])

            # 确保多边形闭合（首尾点相同）
            if points and points[0] != points[-1]:
                points.append(points[0])

            # 创建shape对象
            shape = {
                "label": labels_map.get(class_id, f"class_{class_id}"),
                "points": points,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }

            labelme_json["shapes"].append(shape)

    # 保存JSON文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(labelme_json, f, ensure_ascii=False, indent=2)

    print(f"Labelme JSON格式文件已保存至: {output_json_path}")


def generate_wkt_polygons(classification_result, profile):
    """将分类结果转换为WKT多边形格式
    
    Args:
        classification_result: 分类结果数组 (H, W)
        profile: GeoTIFF元数据
        
    Returns:
        dict: 每个类别的WKT多边形列表
    """
    print("Generating WKT polygons from classification results")
    # 获取变换矩阵和坐标系
    transform = profile['transform']
    crs = profile['crs']

    polygons_by_class = {}

    # 为每个类别生成多边形
    for class_id in np.unique(classification_result):
        class_id = int(class_id)  # 转换为Python原生int类型
        # if class_id == 0:  # 跳过背景类（如果需要）
        #     continue

        # 创建该类别的二值掩码
        mask = (classification_result == class_id).astype(np.uint8)

        # 生成多边形
        results = ({
            'properties': {
                'class_id': class_id,
                'id': i
            },
            'geometry': s
        } for i, (
            s, v) in enumerate(shapes(mask, mask=mask, transform=transform)))

        # 转换为WKT格式
        polygons = []
        for result in results:
            geom = shape(result['geometry'])
            polygons.append({
                'wkt': wkt.dumps(geom),
                'class_id': class_id,
                'id': int(result['properties']['id'])
            })

        if polygons:
            polygons_by_class[LABELS[int(class_id)]] = polygons

    print(f"Generated WKT polygons for {len(polygons_by_class)} classes")
    return polygons_by_class


def generate_shapefile_from_wkt(wkt_polygons, output_shp_path, profile):
    """将WKT多边形转换为Shapefile格式并保存
    
    Args:
        wkt_polygons: WKT多边形字典
        output_shp_path: 输出Shapefile路径
        profile: GeoTIFF元数据，包含坐标系统等信息
    """
    if not SHAPEFILE_SUPPORT:
        print("Geopandas not available, cannot save Shapefile")
        return False

    print("Converting WKT polygons to Shapefile")

    geometries = []
    labels = []
    class_ids = []

    # 遍历所有类别和对应的多边形
    for label, polygons in wkt_polygons.items():
        for poly_data in polygons:
            # 从WKT创建几何对象
            geom = wkt.loads(poly_data['wkt'])

            geometries.append(geom)
            labels.append(label)
            class_ids.append(poly_data['class_id'])

    # 创建GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'label': labels,
        'class_id': class_ids,
        'geometry': geometries
    })

    # 设置坐标参考系统
    if 'crs' in profile:
        gdf.crs = profile['crs']
    else:
        # 如果没有CRS信息，默认使用WGS84
        gdf.crs = 'EPSG:4326'

    # 保存为Shapefile
    try:
        gdf.to_file(output_shp_path, driver='ESRI Shapefile')
        print(f"Shapefile saved successfully to {output_shp_path}")
        return True
    except Exception as e:
        print(f"Error saving Shapefile: {str(e)}")
        return False


def load_model(model_cfg):
    MODEL_NAME = model_name = model_cfg.get("model_name")
    DATASET_NAME = dataset_name = model_cfg.get("dataset_name")
    MODALITY = modality = model_cfg.get("modality")
    classification_model_path = model_cfg.get("classification_model_path")

    print("=" * 50)
    print(f"Testing model: {model_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Modality: {modality}")
    print("=" * 50)

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
    cfg = get_cfg(model_name, dataset_name)
    window_size = cfg.get('window_size')
    model = cfg.get('model')

    # 将模型移到GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.load_state_dict(torch.load(classification_model_path)["model"],
                          strict=False)

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

    return model, cfg


def predict(input, model):
    model.eval()
    with torch.no_grad():
        pred = slide_inference(input, model, n_output_channels=N_CLASSES)

    return pred


if __name__ == "__main__":
    img_path = "/home/yyyjvm/SS-datasets/YYYJ_dataset/train_tmp/1638431685714513920BJ220181031.tif"
    if not os.path.exists(img_path):
        print("图像不存在")

    # 读取GeoTIFF图像
    with rasterio.open(img_path) as src:
        # 读取图像数据
        image_data = src.read()  # 读取所有波段
        profile = src.profile.copy()  # 获取图像的元数据
        print(f"tif元数据：{profile}")

        # 如果是多波段图像，选择前3个波段作为RGB
        if image_data.shape[0] >= 3:
            image_data = image_data[:3, :, :]  # 取前3个波段
        elif image_data.shape[0] == 1:
            # 如果是单波段，复制为3个波段
            image_data = np.repeat(image_data, 3, axis=0)

        # 转换为模型所需的格式 (H, W, C) -> (C, H, W)
        image_data = np.transpose(image_data, (1, 2, 0))

        # 归一化到0-1范围
        image_data = image_data.astype('float32') / 255.0 if image_data.max(
        ) > 1.0 else image_data.astype('float32')

        # 转换为模型所需的格式
        data = np.transpose(image_data, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = torch.from_numpy(data).unsqueeze(0).to(device)

    model_cfg = {
        "model_name":
        "DINOv3",
        "dataset_name":
        "YYYJ",
        "modality":
        "uni",
        "classification_model_path":
        "/home/yyyjvm/SS-projects/dinov3/tasks/segmentation/logs/DINOv3/YYYJ_20251226_083701/DINOv3_YYYJ_e9_mIoU54.5.pth"
    }

    # 进行推理
    model, cfg = load_model(model_cfg)
    # colored_pred, overlay = predict(data, model)
    N_CLASSES = len(cfg.get("labels"))
    LABELS = cfg.get("labels")
    pred = predict(data, model)

    # 将预测结果转换为GeoTIFF格式并保存
    pred_np = pred.cpu().numpy()[0]  # 移除batch维度
    pred_class = np.argmax(pred_np, axis=0)  # 获取每个像素的分类结果

    # 更新profile以适应单波段分类结果
    profile.update({'count': 1, 'dtype': rasterio.uint8})

    # 生成输出文件路径
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    output_geotiff_path = os.path.join(os.path.dirname(img_path),
                                       f"{base_name}_classified.tif")
    output_wkt_path = os.path.join(os.path.dirname(img_path),
                                   f"{base_name}_polygons.json")
    output_labelme_path = os.path.join(os.path.dirname(img_path),
                                       f"{base_name}_labelme.json")
    output_shp_path = os.path.join(os.path.dirname(img_path),
                                   f"{base_name}_polygons.shp")

    # 保存分类结果为GeoTIFF
    # with rasterio.open(output_geotiff_path, 'w', **profile) as dst:
    #     dst.write(pred_class.astype(rasterio.uint8), 1)

    # 生成WKT格式的多边形
    wkt_polygons = generate_wkt_polygons(pred_class, profile)

    # 生成Shapefile
    if SHAPEFILE_SUPPORT:
        success = generate_shapefile_from_wkt(wkt_polygons, output_shp_path,
                                              profile)
        if success:
            print(f"Shapefile saved to {output_shp_path}")
        else:
            print(f"Failed to save Shapefile to {output_shp_path}")
    else:
        print("Shapefile support not available due to missing geopandas")

    # 生成labelme格式的JSON
    # 创建标签映射字典
    labels_map = {i: LABELS[i] for i in range(len(LABELS))}
    save_labelme_format(pred_class, img_path, output_labelme_path, labels_map)
