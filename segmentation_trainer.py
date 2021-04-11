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

    def train(self):
        for epoch in range(self._cfg.num_epochs):
            self._train_single_epoch(epoch)
            if epoch % self._cfg.validation_frequency == 0:
                self.validate(epoch)

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
            iou_value = iou(masks, torch.argmax(model_output, dim=1))
            dice_value = dice(masks, torch.argmax(model_output, dim=1))
            progress_bar.set_description("Epoch: {}, inference time: {:.2f} ms, iou: {:.2f}, "
                                         "loss: {:.2f}".format(epoch, (time() - start) * 1000, iou_value, loss.item()))
            self._writer.add_scalar("TrainingLoss", loss.item(), epoch * len(self._data_loader[DataMode.train]) + index)
            self._writer.add_scalar("TrainingIOU", iou, epoch * len(self._data_loader[DataMode.train]) + index)
            self._writer.add_scalar("TrainingDice", dice_value, epoch * len(self._data_loader[DataMode.train]) + index)

    def validate(self, epoch):
        self._model.eval()

    def save(self):
        pass

    def load(self):
        pass


if __name__ == "__main__":
    SegmentationTrainer(SegmentationConfig()).train()
