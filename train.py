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

    def train(self):
        for idx, data in enumerate(self._loaders[DataMode.train]):
            data = [d.to(self._config.device) for d in data]

    def validate(self):
        pass