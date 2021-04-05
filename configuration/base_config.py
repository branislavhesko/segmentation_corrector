from enum import auto, Enum

from data_tools.transforms import (
    ComposeTransforms, RandomAdditiveNoise, RandomContrastBrightness,
    RandomHorizontalFlip, RandomMultiplicativeNoise, RandomRotate,
    RandomSquaredCrop, RandomVerticalFlip, ToTensor, Transpose)
from modeling.api import DeepLab


class DataProps:
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]


class DataMode(Enum):
    train = auto()
    eval = auto()


class BaseConfig:
    EXPERIMENT_NAME = "VESSELS"
    num_epochs = 10
    batch_size = 2
    num_workers = 0
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
    lr = 1e-3

    augmentation = ComposeTransforms([
        RandomRotate(0.6),
        RandomSquaredCrop(0.85),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomContrastBrightness(),
        RandomMultiplicativeNoise(),
        RandomAdditiveNoise(),
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
    border_limit = 0.5




class SegmentationConfig(ConfigOpticDisc):
    model = DeepLab
    num_classes = 2
    output_stride = 8