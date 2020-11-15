from enum import auto, Enum


class DataMode(Enum):
    train = auto()
    eval = auto()


class Config:
    extension_image = "tif"
    extension_mask = "png"
    path = "./data/pop1/train/imgs"
    mask_path = "./data/pop1/train/masks"
    crop_size = (256, 256)
    device = "cuda"
    num_random_crops_per_image = 10