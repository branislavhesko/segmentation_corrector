from enum import auto, Enum

from transforms import (
    ComposeTransforms, Normalize, RandomHorizontalFlip, RandomRotate, 
    RandomSquaredCrop, RandomVerticalFlip, Resize, ToTensor, Transpose)


class DataProps:
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

class DataMode(Enum):
    train = auto()
    eval = auto()


class Config:
    EXPERIMENT_NAME = "VESSELS"
    num_epochs = 10
    batch_size = 2
    num_workers = 4
    extension_image = "tif"
    extension_mask = "png"
    path = {
        DataMode.train: "./data/pop1/train/imgs",
        DataMode.eval: "./data/pop1/validate/imgs"
    }
    mask_path = {
        DataMode.train: "./data/pop1/train/masks",
        DataMode.eval: "./data/pop1/validate/masks"
    }
    crop_size = (384, 384)
    device = "cuda"
    num_random_crops_per_image = 2
    lr = 1e-4

    augmentation = ComposeTransforms([
        RandomRotate(0.6),
        RandomSquaredCrop(0.85),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        Transpose(),
        ToTensor()
    ])
    val_augmentation = ComposeTransforms([
        Transpose(),
        ToTensor()
    ])
    live_visualization = True
    frequency_visualization = {
        DataMode.train: 100,
        DataMode.eval: 50
    }

    checkpoint_path = "./ckpt"
    alfa = 2
    beta = 4


class ConfigOpticDisc(Config):
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