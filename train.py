from glob import glob
import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config, DataMode
from data_loader import get_data_loaders
from focal_loss import FocalLoss
from modeling.fovea_net import FoveaNet


class Trainer:

    def __init__(self, config: Config):
        self._config = config
        self._model = FoveaNet(num_classes=1).to(self._config.device)
        self._loss = FocalLoss()
        self._loaders = get_data_loaders(config)
        self._writer = SummaryWriter()
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._config.lr)

    def train(self):
        for epoch in range(self._config.num_epochs):
            t = tqdm(self._loaders[DataMode.train])
            for idx, data in enumerate(t):
                self._optimizer.zero_grad()
                imgs, masks, crop_info, opened_img, opened_mask = data
                imgs, masks = imgs.to(self._config.device), masks.to(self._config.device)
                masks = masks.unsqueeze(1)
                output = self._model(imgs)
                loss = self._loss(masks, output)
                t.set_description(f"LOSS: {loss.item()}")
                loss.backward()
                self._optimizer.step()

                if idx % self._config.frequency_visualization[DataMode.train] == 0:
                    self._tensorboard_visualization(loss, epoch, idx, imgs, masks, output)

                if self._config.live_visualization:
                    self._live_visualization(masks, output)

    def _tensorboard_visualization(self, loss, epoch, idx, imgs, masks, output):
        self._writer.add_scalar("Loss/training", loss.item(), 
                                global_step=epoch * len(self._loaders[DataMode.train]) + idx)
        self._writer.add_images("Images/training", imgs, idx)
        self._writer.add_images("Masks/training", masks, idx)
        self._writer.add_images("Outputs/training", output, idx)

    @staticmethod
    def _live_visualization(masks, output):
        show = np.zeros((masks.shape[2], 2 * masks.shape[3]), dtype=np.float32)
        show[:, :masks.shape[3]] = masks[1, 0, :, :].detach().cpu().numpy().astype(np.float32)
        show[:, masks.shape[3]:] = output[1, 0, :, :].detach().cpu().numpy().astype(np.float32)
        cv2.imshow("training", show)
        cv2.waitKey(10)

    def validate(self):
        pass


if __name__ == "__main__":
    Trainer(Config()).train()
