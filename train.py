from glob import glob
import os
import cv2
import numpy as np
import torch
from torch.nn import BCELoss, CrossEntropyLoss
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from configuration.base_config import BaseConfig, DataMode
from configuration.optic_disc import BaseConfigOpticDisc
from data_tools.data_loader import get_data_loaders
from modeling.focal_loss import FocalLoss, TotalLoss
from modeling.deeplab import DeepLab


class Trainer:

    def __init__(self, config: BaseConfig):
        self._config = config
        self._model = DeepLab(num_classes=9, output_stride=8, sync_bn=False).to(self._config.device)
        self._border_loss = TotalLoss(self._config)
        self._direction_loss = CrossEntropyLoss()
        self._loaders = get_data_loaders(config)
        self._writer = SummaryWriter()
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=self._config.lr, weight_decay=1e-4, nesterov=True, momentum=0.9)
        self._scheduler = torch.optim.lr_scheduler.ExponentialLR(self._optimizer, gamma=0.97)

        if self._config.parallel:
            self._model = DistributedDataParallel(self._model, device_ids=[self._config.device, ])
        # self._load()

    def train(self):
        for epoch in range(self._config.num_epochs):
            t = tqdm(self._loaders[DataMode.train])
            self._model.train()
            for idx, data in enumerate(t):
                self._optimizer.zero_grad()
                imgs, borders, masks, crop_info, opened_img, opened_mask = data
                imgs, borders, masks = imgs.to(self._config.device), borders.to(self._config.device), masks.to(self._config.device)
                # borders = borders.unsqueeze(1)
                output = self._model(imgs)
                border_output = output[:, :1, :, :].squeeze()
                direction_output = output[:, 1:, :, :]
                seg_loss = self._direction_loss(direction_output, masks)
                loss = self._border_loss(border_output, borders) + seg_loss
                t.set_description(f"LOSS: {seg_loss.item()}")
                loss.backward()
                self._optimizer.step()
                self._writer.add_scalar("Loss/training", loss.item(), 
                                        global_step=epoch * len(self._loaders[DataMode.train]) + idx)

                if idx % self._config.frequency_visualization[DataMode.train] == 0:
                    self._tensorboard_visualization(
                        loss=loss, epoch=epoch, idx=idx, imgs=imgs, 
                        border_gt=borders.unsqueeze(1), border=border_output.unsqueeze(1),
                        direction_gt=masks, direction=direction_output)

                if self._config.live_visualization:
                    self._live_visualization(imgs, borders, output)
        self._save()

    @staticmethod
    def _get_mask(masks):
        masks_new = torch.zeros(masks.shape[0], 2, masks.shape[1], masks.shape[2], device=masks.device)
        for idx in range(2):
            masks_new[:, idx, :, :][masks == idx] = 1
        return masks_new

    def validate(self, epoch):
        t = tqdm(self._loaders[DataMode.eval])
        self._model.eval()
        for idx, data in enumerate(t):
            imgs, masks, _, _, _ = data
            imgs, masks = imgs.to(self._config.device), masks.to(self._config.device)
            output = self._model(imgs)
            loss = self._border_loss(masks, output)

    def _tensorboard_visualization(self, loss, epoch, idx, imgs, border_gt, direction_gt, border, direction):
        self._writer.add_images(f"Images{idx}/training", imgs, epoch)
        self._writer.add_images(f"Border{idx}/training", border, epoch)
        self._writer.add_images(f"BorderGT{idx}/training", border_gt, epoch)
        self._writer.add_images(f"DirectionGT{idx}/training", direction_gt.unsqueeze(1), epoch)
        self._writer.add_images(f"Direction{idx}/training", direction.argmax(1).unsqueeze(1), epoch)

    def _save(self):
        path = os.path.join(self._config.checkpoint_path, self._config.EXPERIMENT_NAME)
        self.check_and_mkdir(path)
        torch.save(self._model.state_dict(), os.path.join(path, "corrector.pth"))

    def _load(self):
        path = os.path.join(self._config.checkpoint_path, self._config.EXPERIMENT_NAME)
        weights = glob(os.path.join(path, "*.pth"))
        if len(weights):
            state_dict = torch.load(weights[0])
            try:
                self._model.load_state_dict(state_dict)
            except RuntimeError as e:
                print("ERROR while loading weights: {}".format(e))

    @staticmethod
    def check_and_mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    @staticmethod
    def _live_visualization(imgs, masks, output):
        out = np.expand_dims((output[1, 0, :, :].detach().cpu().numpy()).astype(np.float32), axis=2) > 0.7
        out2 = np.expand_dims(np.argmax((output[1, 1:, :, :].detach().cpu().numpy()).astype(np.float32), axis=0), axis=2) / 7.
        out3 = np.expand_dims((output[1, 2, :, :].detach().cpu().numpy()).astype(np.float32), axis=2)

        show = np.zeros((masks.shape[2], 5 * masks.shape[3], 3), dtype=np.float32)
        show[:, :masks.shape[3]] = cv2.applyColorMap(
            (masks[1, 0, :, :] * 255).detach().cpu().numpy().astype(
                np.uint8), cv2.COLORMAP_BONE)
        show[:, masks.shape[3]: 2 * masks.shape[3], :] = np.concatenate([out, out, out], axis=2)
        show[:, 2 * masks.shape[3]:3 * masks.shape[3]] = cv2.cvtColor(
            imgs[1, ...].permute(1, 2, 0).cpu().detach().numpy(), cv2.COLOR_BGR2RGB)
        show[:, 3 * masks.shape[3]:4 * masks.shape[3]] = np.concatenate([out2, out2, out2], axis=2)
        show[:, 4 * masks.shape[3]:5 * masks.shape[3]] = np.concatenate([out3, out3, out3], axis=2)
        cv2.imshow("training", show)
        cv2.waitKey(10)


if __name__ == "__main__":
    Trainer(BaseConfigOpticDisc()).train()
