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
    crop_size = (256, 256)
    device = "cpu"
    num_random_crops_per_image = 10
    lr = 1e-4

    augmentation = ComposeTransforms([
        Normalize(DataProps.MEAN, DataProps.STD),
        RandomRotate(0.6),
        RandomSquaredCrop(0.85),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        Transpose(),
        ToTensor()
    ])
    val_augmentation = ComposeTransforms([
        Normalize(DataProps.MEAN, DataProps.STD),
        Transpose(),
        ToTensor()
    ])