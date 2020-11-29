from collections import namedtuple
import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import Config, DataMode


CurrentlyOpened = namedtuple("CurrentlyOpened", ["image", "mask", "id"])


class SmartRandomDataSet(Dataset):
    MASK_LOAD_TYPE = cv2.IMREAD_GRAYSCALE

    def __init__(self, config, img_files, mask_files,
                 crop_size, transforms, **_):
        self._config = config
        self._img_files = img_files
        self._mask_files = mask_files
        self._crop_size = crop_size
        self._transforms = transforms
        self._num_random_crops = self._config.num_random_crops_per_image
        self._currently_opened = CurrentlyOpened(None, None, None)

    def __len__(self):
        assert len(self._img_files) == len(self._mask_files)
        return len(self._img_files) * self._num_random_crops

    def __getitem__(self, item):
        image_id = item // self._num_random_crops
        if self._currently_opened.id != image_id:
            self.assign_currently_opened(image_id)
        rand_row, rand_col = self._get_random_crop(self._currently_opened.image.shape, self._crop_size)
        image_crop, mask_crop = self._crop(rand_col, rand_row)
        data = (
            *self._transforms(image_crop, mask_crop),
            (rand_row, rand_col, rand_row + self._crop_size[0], rand_col + self._crop_size[1]),
            self._img_files[self._currently_opened.id], self._mask_files[self._currently_opened.id]
        )
        return data

    def _crop(self, rand_col, rand_row):
        image_crop = np.copy(self._currently_opened.image[rand_row: rand_row + self._crop_size[0],
                                                          rand_col: rand_col + self._crop_size[1], :])
        mask_crop = np.copy(self._currently_opened.mask[rand_row: rand_row + self._crop_size[0],
                                                        rand_col: rand_col + self._crop_size[1]])
        return image_crop, mask_crop

    def assign_currently_opened(self, image_id):
        self._currently_opened = CurrentlyOpened(
            image=cv2.cvtColor(cv2.imread(
                self._img_files[image_id], cv2.IMREAD_COLOR).astype(np.float32) / 255., cv2.COLOR_BGR2RGB),
            mask=self.process_mask(cv2.imread(self._mask_files[image_id], self.MASK_LOAD_TYPE)),
            id=image_id
        )

    def process_mask(self, mask):
        mask = (mask > 0).astype(np.uint8)
        mask_dilated = cv2.dilate(mask, np.ones((5, 5)))
        return mask_dilated - mask

    def _get_random_crop(self, image_size, crop_size):
        rand_row = torch.randint(low=0, high=image_size[0] - crop_size[0], size=[1])
        rand_col = torch.randint(low=0, high=image_size[1] - crop_size[1], size=[1])
        return rand_row.item(), rand_col.item()


class SmartRandomDataSetIdrid(SmartRandomDataSet):
    CLASS_VALUE = 10
    def process_mask(self, mask):
        mask[mask <= self.CLASS_VALUE] = 1
        mask[mask > self.CLASS_VALUE] = 0
        return (cv2.distanceTransform(mask, cv2.DIST_L1, 0) == 1).astype(np.uint8)


def get_data_loaders(config: Config):
    images = sorted(glob.glob(os.path.join(config.path[DataMode.train], "*." + config.extension_image)))
    masks = sorted(glob.glob(os.path.join(config.mask_path[DataMode.train], "*." + config.extension_mask)))
    images_val = sorted(glob.glob(os.path.join(config.path[DataMode.eval], "*." + config.extension_image)))
    masks_val = sorted(glob.glob(os.path.join(config.mask_path[DataMode.eval], "*." + config.extension_mask)))
    data_loader = DataLoader(
        SmartRandomDataSetIdrid(config=config, img_files=images, mask_files=masks, 
                           crop_size=config.crop_size, transforms=config.augmentation),
        batch_size=config.batch_size, num_workers=config.num_workers)
    data_loader_val = DataLoader(
        SmartRandomDataSetIdrid(config=config, img_files=images_val, mask_files=masks_val, 
                           crop_size=config.crop_size, transforms=config.val_augmentation),
        batch_size=1, num_workers=2)
    return {
        DataMode.eval: data_loader_val,
        DataMode.train: data_loader
    }


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from config import ConfigOpticDisc
    loader = get_data_loaders(ConfigOpticDisc())
    for idx, data in enumerate(loader[DataMode.train]):
        plt.imshow(data[1].squeeze().numpy()[0, :, :])
        plt.show()

