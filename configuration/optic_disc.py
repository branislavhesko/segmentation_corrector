from configuration.base_config import BaseConfig, DataMode

from modeling.deeplab import DeepLab

class BaseConfigOpticDisc(BaseConfig):
    model = DeepLab
    device = "cuda"
    EXPERIMENT_NAME = "OPTIC_DISC"
    extension_image = "jpg"
    extension_mask = "jpg"
    path = {
        DataMode.train: "./data/refugee/train/imgs",
        DataMode.eval: "./data/refugee/validate/imgs"
    }
    mask_path = {
        DataMode.train: "./data/refugee/train/masks",
        DataMode.eval: "./data/refugee/validate/masks"
    }