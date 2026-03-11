import os
from .ISPRS_dataset import ISRPS_Dataset
from .YYYJ_dataset import YYYJ_Dataset
from .EarthMiss_dataset import EarthMiss_Dataset
from .WHU_dataset import WHU_Dataset

from configs.common_cfg import MS_ROOT_DIR


def build_dataset(dataset_name, data_type="test", **kwargs):
    # model_name = kwargs.get("model_name")
    # if model_name in ['DINOv3', 'SegDINO', 'Mask2Former']:
    #     normalize_type = "geo"
    # elif model_name in ['']:
    #     normalize_type = None
    # else:
    #     normalize_type = "common"
    normalize_type = "common"

    if dataset_name == "Potsdam":
        ids = ['4_10', '5_11', '2_11', '3_10', '6_11', '7_12'
               ] if data_type == "test" else [
                   '6_10', '7_10', '2_12', '3_11', '2_10', '7_8', '5_10',
                   '3_12', '5_12', '7_11', '7_9', '6_9', '7_7', '4_12', '6_8',
                   '6_12', '6_7', '4_11'
               ]
        root_dir = f"{MS_ROOT_DIR}/SS-datasets/ISPRS_dataset/"
        data_dir = root_dir + "Potsdam/2_Ortho_RGB/top_potsdam_{}_RGB.tif"  # RGB
        # data_dir = root_dir + "Potsdam/4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif"  # RGBIR
        # label_dir = root_dir + "Potsdam/5_Labels_for_participants/top_potsdam_{}_label.tif"
        label_dir = root_dir + "Potsdam/5_Labels_for_participants_no_Boundary/top_potsdam_{}_label_noBoundary.tif"
        dsm_dir = root_dir + 'Potsdam/1_DSM_normalisation/dsm_potsdam_{}_normalized_lastools.jpg' if kwargs.get(
            "modality") == "multi" else None
        return ISRPS_Dataset(ids=ids,
                             data_dir=data_dir,
                             label_dir=label_dir,
                             dsm_dir=dsm_dir,
                             dataset_name=dataset_name,
                             data_type=data_type,
                             window_size=kwargs.get("window_size", (224, 224)),
                             normalize_type=normalize_type)
    elif dataset_name == "Vaihingen":
        ids = [5, 21, 15, 30] if data_type == "test" else [
            1, 3, 23, 26, 7, 11, 13, 28, 17, 32, 34, 37
        ]
        root_dir = f"{MS_ROOT_DIR}/SS-datasets/ISPRS_dataset/"
        data_dir = root_dir + "Vaihingen/top/top_mosaic_09cm_area{}.tif"
        # label_dir = root_dir + "Vaihingen/gts_complete/top_mosaic_09cm_area{}.tif"
        label_dir = root_dir + "Vaihingen/gts_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif"
        dsm_dir = root_dir + "Vaihingen/dsm/dsm_09cm_matching_area{}.tif" if kwargs.get(
            "modality") == "multi" else None
        return ISRPS_Dataset(ids=ids,
                             data_dir=data_dir,
                             label_dir=label_dir,
                             dsm_dir=dsm_dir,
                             dataset_name=dataset_name,
                             data_type=data_type,
                             window_size=kwargs.get("window_size", (224, 224)),
                             normalize_type=normalize_type)
    elif dataset_name == "YYYJ":
        train_dir = f"{MS_ROOT_DIR}/SS-datasets/YYYJ_dataset/new_train"
        test_dir = f"{MS_ROOT_DIR}/SS-datasets/YYYJ_dataset/new_test"
        # 读取文件夹中所有tif文件名
        ids = [
            f.split(".")[0] for f in os.listdir(test_dir) if f.endswith(".tif")
        ] if data_type == "test" else [
            f.split(".")[0] for f in os.listdir(train_dir)
            if f.endswith(".tif")
        ]
        data_dir = test_dir + "/{}.tif" if data_type == "test" else train_dir + "/{}.tif"
        label_dir = test_dir + "/label_masks/{}.tif" if data_type == "test" else train_dir + "/label_masks/{}.tif"
        return YYYJ_Dataset(ids=ids,
                            data_dir=data_dir,
                            label_dir=label_dir,
                            data_type=data_type,
                            window_size=kwargs.get("window_size", (224, 224)),
                            normalize_type=normalize_type)
    elif dataset_name == "EarthMiss":
        citys = [
            "Singapore", "Nanjing", "America-Eugene", "America-Louisville",
            "French-Paris", "Netherlands-Rotterdam", "Morocco-Casablanca"
        ] if data_type == "train" else [
            "Japan-Hakodate", "America-NewYork", "Peru-Callao"
        ]
        root_dir = f"{MS_ROOT_DIR}/SS-datasets/EarthMiss/"
        rgb_dir = root_dir + "{}/images/RGB/"
        sar_dir = root_dir + "{}/images/SAR/" if kwargs.get(
            "modality") == "multi" else None
        label_dir = root_dir + "{}/masks/"
        return EarthMiss_Dataset(citys=citys,
                                 rgb_dir=rgb_dir,
                                 sar_dir=sar_dir,
                                 label_dir=label_dir,
                                 data_type=data_type,
                                 window_size=kwargs.get(
                                     "window_size", (224, 224)),
                                 normalize_type=normalize_type)
    elif dataset_name == "WHU":
        root_dir = f"{MS_ROOT_DIR}/SS-datasets/whu-opt-sar/"

        if data_type == "train":
            train_txt_path = f"{root_dir}/train_list.txt"
            with open(train_txt_path, 'r') as f:
                filenames = [
                    line.strip() for line in f.readlines() if line.strip()
                ]
        else:
            test_txt_path = f"{root_dir}/test_list.txt"
            with open(test_txt_path, 'r') as f:
                filenames = [
                    line.strip() for line in f.readlines() if line.strip()
                ]

        data_dir = root_dir + "optical/{}"
        label_dir = root_dir + "lbl/{}"
        sar_dir = root_dir + "sar/{}" if kwargs.get(
            "modality") == "multi" else None
        return WHU_Dataset(filenames=filenames,
                           rgb_dir=data_dir,
                           label_dir=label_dir,
                           sar_dir=sar_dir,
                           data_type=data_type,
                           window_size=kwargs.get("window_size", (224, 224)),
                           normalize_type=normalize_type)
