import os
from time import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configuration.base_config import DataMode
from configuration.segmentation_config import SegmentationConfig
from data_tools.segmentation_loader import get_data_loaders
from evaluation import iou, dice


class SegmentationTrainer:

    def __init__(self, config: SegmentationConfig):
        self._cfg = config
        self._model = self._cfg.model(
            num_classes=self._cfg.num_classes, 
            output_stride=self._cfg.output_stride, 
            sync_bn=False).to(self._cfg.device)
        self._optimizer = torch.optim.SGD(
            self._model.parameters(), 
            lr=self._cfg.lr, 
            momentum=0.95, 
            nesterov=True, 
            weight_decay=1e-4)
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, T_max=self._cfg.num_epochs)
        self._data_loader = get_data_loaders(self._cfg)
        self._loss = torch.nn.CrossEntropyLoss()
        self._writer = SummaryWriter()
        self.load()

    def train(self):
        for epoch in range(self._cfg.num_epochs):
            self._train_single_epoch(epoch)
            if epoch % self._cfg.validation_frequency == 0:
                self.validate(epoch)
            self._scheduler.step()
        self.save()

    def _train_single_epoch(self, epoch):
        self._model.train()
        progress_bar = tqdm(self._data_loader[DataMode.train])

        for index, data in enumerate(progress_bar):
            start = time()
            self._optimizer.zero_grad()
            images, masks = [d.to(self._cfg.device) for d in data]
            model_output = self._model(images)
            loss = self._loss(model_output, masks)
            loss.backward()
            self._optimizer.step()
            dice_value, iou_value = self._get_metrics(masks, model_output)
            progress_bar.set_description("TRAINING - Epoch: {}, inference time: {:.2f} ms, iou: {:.2f}, dice: {:.2f} "
                                         "loss: {:.2f}".format(epoch, (time() - start) * 1000,
                                                               iou_value, dice_value, loss.item()))
            self._write_scalar_to_tensorboard(dice_value, iou_value, epoch, index, loss)

    @torch.no_grad()
    def validate(self, epoch):
        self._model.eval()
        progress_bar = tqdm(self._data_loader[DataMode.eval])

        for index, data in enumerate(progress_bar):
            start = time()
            images, masks = [d.to(self._cfg.device) for d in data]
            model_output = self._model(images)
            loss = self._loss(model_output, masks)
            dice_value, iou_value = self._get_metrics(masks, model_output)
            progress_bar.set_description("VALIDATION - Epoch: {}, inference time: {:.2f} ms, iou: {:.2f}, dice: {:.2f} "
                                         "loss: {:.2f}".format(epoch, (time() - start) * 1000,
                                                               iou_value, dice_value, loss.item()))
            self._write_scalar_to_tensorboard(dice_value, iou_value, epoch, index, loss)

    @staticmethod
    def _get_metrics(masks, model_output):
        iou_value = iou(masks, torch.argmax(model_output, dim=1))
        dice_value = dice(masks, torch.argmax(model_output, dim=1))
        return dice_value, iou_value

    def _write_scalar_to_tensorboard(self, dice_value, iou_value, epoch, index, loss, mode=DataMode.train):
        mode = "Training" if mode == DataMode.train else "Validation"
        self._writer.add_scalar(f"{mode}Loss", loss.item(), epoch * len(self._data_loader[DataMode.train]) + index)
        self._writer.add_scalar(f"{mode}IOU", iou_value, epoch * len(self._data_loader[DataMode.train]) + index)
        self._writer.add_scalar(f"{mode}Dice", dice_value, epoch * len(self._data_loader[DataMode.train]) + index)

    def save(self):
        path = os.path.join(self._cfg.checkpoint_path, self._cfg.EXPERIMENT_NAME, "checkpoint.pth")
        if not os.path.exists(os.path.split(path)[0]):
            os.makedirs(os.path.split(path)[0], exist_ok=True)
        model_info = {
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "scheduler": self._scheduler.state_dict(),
            "metrics": []
        }
        torch.save(model_info, path)

    def load(self):
        path = os.path.join(self._cfg.checkpoint_path, self._cfg.EXPERIMENT_NAME, "checkpoint.pth")
        if not os.path.isfile(path):
            return
        model_info = torch.load(path)
        self._model.load_state_dict(model_info["model"])
        if "optimizer" in model_info:
            self._optimizer.load_state_dict(model_info["optimizer"])
        if "scheduler" in model_info:
            self._scheduler.load_state_dict(model_info["scheduler"])


if __name__ == "__main__":
    SegmentationTrainer(SegmentationConfig()).train()
