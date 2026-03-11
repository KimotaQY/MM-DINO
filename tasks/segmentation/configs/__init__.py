from .MMDINO import get_cfg as get_DINOv3_cfg


def get_cfg(model_name=None, dataset_name=None, **kwargs):
    if model_name is None:
        raise ValueError("Model name must be specified")
    if dataset_name is None:
        raise ValueError("Dataset name must be specified")

    if 'DINOv3' in model_name:
        cfg = get_DINOv3_cfg(model_name, dataset_name, **kwargs)
    else:
        raise ValueError("Model name is not supported")

    return cfg
