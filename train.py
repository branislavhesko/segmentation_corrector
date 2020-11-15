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
        self._model = FoveaNet(num_classes=1)
        self._loss = FocalLoss()
        self._loaders = get_data_loaders(config)
        self._writer = SummaryWriter()
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._config.lr)

    def train(self):
        for idx, data in enumerate(self._loaders[DataMode.train]):
            print(data)
            self._optimizer.zero_grad()
            imgs, masks, crop_info, opened_img, opened_mask = data
            imgs, masks = imgs.to(self._config.device), masks.to(self._config.device)
            output = self._model(imgs)
            loss = self._loss(output, masks)
            loss.backwards()
            self._optimizer.step()


    def validate(self):
        pass


if __name__ == "__main__":
    Trainer(Config()).train()