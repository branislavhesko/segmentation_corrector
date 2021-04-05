import torch

from configuration.config import SegmentationConfig


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
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer)

    def train(self):
        for epoch in range(self._cfg.num_epochs):
            pass

    def _train_single_epoch(self, epoch):
        pass

    def validate(self, epoch):
        pass

    def save(self):
        pass

    def load(self):
        pass
