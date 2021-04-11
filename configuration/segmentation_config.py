from configuration import base_config
from modeling.api import DeepLab


class SegmentationConfig(base_config.BaseConfig):
    model = DeepLab
    num_classes = 2
    output_stride = 8
    device = "cuda"
    EXPERIMENT_NAME = "OPTIC_DISC"
    extension_image = "jpg"
    extension_mask = "jpg"
    path = {
        base_config.DataMode.train: "./data/refugee/train/imgs",
        base_config.DataMode.eval: "./data/refugee/validate/imgs"
    }
    mask_path = {
        base_config.DataMode.train: "./data/refugee/train/masks",
        base_config.DataMode.eval: "./data/refugee/validate/masks"
    }