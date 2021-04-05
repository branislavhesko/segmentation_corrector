import numpy as np
import torch.utils.data as data

from configuration.config import Config


class SmartSegmentationLoader(data.Dataset):

    def __init__(self, config, img_files, mask_files, transforms):
        super().__init__()
        self._img_files = img_files
        self._mask_files = mask_files
        self._config = config
        self._transforms = transforms

    def __len__(self):
        assert len(self._img_files) == len(self._mask_files), "Num masks and num images should be equal"
        return len(self._img_files)

    def __getitem__(self, item):
        img = self._img_files[item]
        mask = self._mask_files[item]
        image_crop, mask_crop = self._crop(image, mask)
        image, mask = self._transforms(image, mask)
        return image, mask


    def _crop(self, image, mask):
        rand_row, rand_col = self._get_random_crop(img.shape, self._config.crop_size)

        image_crop = np.copy(image[rand_row: rand_row + self._config.crop_size[0],
                                   rand_col: rand_col + self._config.crop_size[1], :])
        mask_crop = np.copy(mask[rand_row: rand_row + self._config.crop_size[0],
                                   rand_col: rand_col + self._config.crop_size[1], :])
        return image_crop, mask_crop

    def _get_random_crop(self, image_size, crop_size):
        rand_row = torch.randint(low=0, high=image_size[0] - crop_size[0], size=[1])
        rand_col = torch.randint(low=0, high=image_size[1] - crop_size[1], size=[1])
        return rand_row.item(), rand_col.item()



def get_data_loaders(config: Config):
    images = sorted(glob.glob(os.path.join(config.path[DataMode.train], "*")))
    masks = sorted(glob.glob(os.path.join(config.mask_path[DataMode.train], "*")))
    images_val = sorted(glob.glob(os.path.join(config.path[DataMode.eval], "*")))
    masks_val = sorted(glob.glob(os.path.join(config.mask_path[DataMode.eval], "*")))
    data_loader = DataLoader(
        SmartSegmentationLoader(config=config, img_files=images, mask_files=masks, 
                           crop_size=config.crop_size, transforms=config.augmentation),
        batch_size=config.batch_size, num_workers=config.num_workers)
    data_loader_val = DataLoader(
        SmartSegmentationLoader(config=config, img_files=images_val, mask_files=masks_val, 
                           crop_size=config.crop_size, transforms=config.val_augmentation),
        batch_size=1, num_workers=2)
    return {
        DataMode.eval: data_loader_val,
        DataMode.train: data_loader
    }
