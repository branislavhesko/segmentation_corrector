import albumentations
import numpy as np
import cv2
from skimage.transform import resize
import torch
from scipy.ndimage.interpolation import rotate


class Normalize:
    def __init__(self, means, stds):
        self._means = means
        self._stds = stds

    def __call__(self, img, border, mask):
        assert img.shape[2] == len(self._means)
        img[:, :, 0] = (img[:, :, 0] - self._means[0]) / self._stds[0]
        img[:, :, 1] = (img[:, :, 1] - self._means[1]) / self._stds[1]
        img[:, :, 2] = (img[:, :, 2] - self._means[2]) / self._stds[2]
        return img, border, mask


class RandomHorizontalFlip:

    def __init__(self, p=0.5):
        self._p = p

    def __call__(self, img, border, mask):
        if self._p < np.random.rand():
            return img[:, ::-1, :], border[:, ::-1], mask[:, ::-1]
        else:
            return img, border, mask


class RandomVerticalFlip:

    def __init__(self, p=0.5):
        self._p = p

    def __call__(self, img, border, mask):
        if self._p < np.random.rand():
            return img[::-1, :, :], border[::-1, :], mask[::-1, :]
        else:
            return img, border, mask


class RandomSquaredCrop:

    def __init__(self, minimal_relative_crop_size=0.5):
        self._minimal_relative_crop_size = minimal_relative_crop_size

    def __call__(self, img, border, mask):
        h, w = mask.shape
        size = np.random.randint(int(self._minimal_relative_crop_size * h), h)
        crop_x_origin = np.random.randint(0, h - size)
        crop_y_origin = np.random.randint(0, w - size)
        # print("Crop_x: {}, crop_y: {}, size: {}".format(crop_x_origin, crop_y_origin, size))
        image = cv2.resize(img[crop_x_origin: crop_x_origin + size, crop_y_origin: crop_y_origin + size, :], (w, h)),
        mask_ = cv2.resize(mask[crop_x_origin: crop_x_origin + size,
                           crop_y_origin: crop_y_origin + size], (w, h), interpolation=cv2.INTER_NEAREST)
        border_ = cv2.resize(border[crop_x_origin: crop_x_origin + size,
                           crop_y_origin: crop_y_origin + size], (w, h), interpolation=cv2.INTER_NEAREST)
        return image[0], border_, mask_


class RandomContrastBrightness:

    def __init__(self, probability=0.5):
        self._probability = probability
        self._transform = albumentations.RandomBrightnessContrast()

    def __call__(self, img, mask):
        if np.random.rand(1) > self._probability:
            return img, mask
        return self._transform(image=img)["image"], mask


class RandomAdditiveNoise:
    def __init__(self, probability=0.5):
        self._probability = probability

    def __call__(self, img, mask):
        if np.random.rand(1) > self._probability:
            return img, mask
        sigma = 0.01 * 255 if img.dtype == np.uint8 else 0.01
        return img + np.random.randn(*img.shape) * sigma


class RandomMultiplicativeNoise:
    def __init__(self, probability=0.5):
        self._probability = probability

    def __call__(self, img, mask):
        if np.random.rand(1) > self._probability:
            return img, mask
        sigma = 0.01 * 255 if img.dtype == np.uint8 else 0.01
        return img * (1 + np.random.randn(*img.shape) * sigma)


class RandomRotate:
    def __init__(self, p=0.5, std_dev=10):
        self._p = p
        self._std_dev = std_dev

    def __call__(self, img, border, mask):
        if np.random.rand() > self._p:
            return img, border, mask
        else:
            # angle = np.random.randint(0, 180)
            angle = np.random.normal(0, self._std_dev / 3.)  # normal distribution
            return rotate(img, angle, reshape=False), rotate(border, angle, reshape=False, order=0), rotate(mask, angle, reshape=False, order=0)
            

class Resize:
    def __init__(self, size):
        self._size = size

    def __call__(self, img, border, mask):
        return resize(img, self._size), resize(border, self._size), resize(mask, self._size)


class ComposeTransforms:
    def __init__(self, transforms):
        self._transforms = transforms
    
    def __call__(self, img, border, mask):
        for transform in self._transforms:
            img, border, mask = transform(img, border, mask)
        return img, border, mask

    def __str__(self):
        return "|".join(map(lambda x: x.__class__.__name__, self._transforms))


class Clahe:
    def __init__(self):
        self._clage = None

    def __call__(self, img, border, mask):
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        print(img.dtype)
        return self._clahe.apply(img), border, mask
        

class ToTensor:

    def __call__(self, img, border, mask):
        return torch.from_numpy(img.astype(np.float32)), torch.from_numpy(border.astype(np.int64)), torch.from_numpy(mask.astype(np.int64))


class Transpose:

    def __call__(self, img, border, mask):
        if len(img.shape) == 2:
            img = img[np.newaxis, :, :]
            return img, mask
        return np.transpose(img, [2, 0, 1]), border, mask
        

if __name__ == "__main__":
    img = cv2.imread("./data/train/imgs/aria_aria_a_14_11.png", cv2.IMREAD_COLOR)[:, :, ::-1]
    mask = cv2.imread("./data/train/masks/aria_aria_a_14_11.png", cv2.IMREAD_GRAYSCALE)
    mask[mask > 0] = 1
    transform = RandomSquaredCrop(0.8)
    from matplotlib import pyplot
    for i in range(20):
        image, mask_t = transform(img, mask)
        pyplot.subplot(1, 2, 1)
        pyplot.imshow(image)
        pyplot.subplot(1, 2, 2)
        pyplot.imshow(mask_t)
        pyplot.show()