import cv2
import os
import torch

from configuration.base_config import BaseConfig
from modeling.fovea_net import FoveaNet


class PredictorSingleImage:

    def __init__(self, config: BaseConfig) -> None:
        self._config = config
        self._model = FoveaNet(num_classes=1).to(self._config.device)
        self._load()
    
    @torch.no_grad()
    def execute(self, image):
        img = self._to_tensor(image, self._config.device)
        return self._model(img).squeeze().cpu().numpy()

    @staticmethod
    def _to_tensor(image, device):
        return torch.from_numpy(image).permute(2, 0, 1).to(device).float().unsqueeze_(0)

    def _load(self):
        path = os.path.join(self._config.checkpoint_path, self._config.EXPERIMENT_NAME)
        assert os.path.isdir(path), "Path does not exists!"
        weights = os.path.join(path, "w.pth")
        assert os.path.exists(weights), "Weights does not exists!"
        self._model.load_state_dict(torch.load(weights))
        self._model.eval()


class PredictorSingleFile(PredictorSingleImage):
    def execute(self, image_file):
        return super().execute(cv2.cvtColor(cv2.imread(image_file, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) / 255.)


if __name__ == "__main__":
    predictor = PredictorSingleFile(config=BaseConfig())
    output = predictor.execute("./Study01_00040_003.tif")
    from matplotlib import pyplot as plt
    plt.imshow(output, vmin=0.2, vmax=0.6, cmap="jet")
    plt.show()
