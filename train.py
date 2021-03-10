from glob import glob
import os
import cv2
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config, ConfigOpticDisc, DataMode
from data_tools.data_loader import get_data_loaders
from modeling.focal_loss import FocalLoss
from modeling.deeplab import DeepLab


class Trainer:

    def __init__(self, config: Config):
        self._config = config
        self._model = DeepLab(num_classes=1, output_stride=8, sync_bn=False).to(self._config.device)
        self._loss = FocalLoss()
        self._loaders = get_data_loaders(config)
        self._writer = SummaryWriter()
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._config.lr)
        self._scheduler = torch.optim.lr_scheduler.ExponentialLR(self._optimizer, gamma=0.97)
        self._load()

    def train(self):
        for epoch in range(self._config.num_epochs):
            t = tqdm(self._loaders[DataMode.train])
            self._model.train()
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
                    self._live_visualization(imgs, masks, output)
            self._save()

    def validate(self, epoch):
        t = tqdm(self._loaders[DataMode.eval])
        self._model.eval()
        for idx, data in enumerate(t):
            imgs, masks, _, _, _ = data
            imgs, masks = imgs.to(self._config.device), masks.to(self._config.device)
            output = self._model(imgs)
            loss = self._loss(masks, output)

    def _tensorboard_visualization(self, loss, epoch, idx, imgs, masks, output):
        self._writer.add_scalar("Loss/training", loss.item(), 
                                global_step=epoch * len(self._loaders[DataMode.train]) + idx)
        self._writer.add_images("Images/training", imgs, idx)
        self._writer.add_images("Masks/training", masks, idx)
        self._writer.add_images("Outputs/training", output, idx)

    def _save(self):
        path = os.path.join(self._config.checkpoint_path, self._config.EXPERIMENT_NAME)
        self.check_and_mkdir(path)
        torch.save(self._model.state_dict(), os.path.join(path, "weight.pth"))

    def _load(self):
        path = os.path.join(self._config.checkpoint_path, self._config.EXPERIMENT_NAME)
        weights = glob(os.path.join(path, "*.pth"))
        if len(weights):
            state_dict = torch.load(weights[0])
            self._model.load_state_dict(state_dict)

    @staticmethod
    def check_and_mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    @staticmethod
    def _live_visualization(imgs, masks, output):
        out = np.expand_dims((output[1, 0, :, :].detach().cpu().numpy()).astype(np.float32), axis=2)
        # from matplotlib import pyplot as plt
        # plt.imshow(out[:, :, 0])
        # plt.show()
        show = np.zeros((masks.shape[2], 3 * masks.shape[3], 3), dtype=np.float32)
        show[:, :masks.shape[3]] = cv2.applyColorMap(
            (masks[1, 0, :, :] * 255).detach().cpu().numpy().astype(
                np.uint8), cv2.COLORMAP_BONE)
        show[:, masks.shape[3]: 2 * masks.shape[3], :] = np.concatenate([out, out, out], axis=2)
        show[:, 2 * masks.shape[3]:] = cv2.cvtColor(
            imgs[1, ...].permute(1, 2, 0).cpu().detach().numpy(), cv2.COLOR_BGR2RGB)
        cv2.imshow("training", show)
        cv2.waitKey(10)


if __name__ == "__main__":
    Trainer(ConfigOpticDisc()).train()
