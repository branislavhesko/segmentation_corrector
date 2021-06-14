import enum
import os

import torch
import tqdm

from configuration.base_config import DataMode
from configuration.optic_disc import BaseConfigOpticDisc
from configuration.segmentation_config import SegmentationConfig
from data_tools.segmentation_loader import get_data_loaders
from modeling import deeplab


def iou(gt, pred):
    return torch.sum(gt & pred) / torch.sum(gt | pred)


def dice(gt, pred):
    return 2 * torch.sum(gt & pred) / (torch.sum(gt) + torch.sum(pred))



class MetricAggregator:
    
    def __init__(self) -> None:
        self.seg_iou = []
        self.cor_iou = []
        self.seg_dice = []
        self.cor_dice = []

    def update(self, seg_iou, seg_dice, cor_iou, cor_dice):
        self.seg_iou.append(seg_iou)
        self.seg_dice.append(seg_dice)
        self.cor_iou.append(cor_iou)
        self.cor_dice.append(cor_dice)

class Evaluator:
    
    def __init__(self, segmentation_model, correction_model):
        self._segmentation_model = segmentation_model
        self._correction_model = correction_model
        self._cfg = SegmentationConfig()
        self._data_loader = get_data_loaders(self._cfg)[DataMode.eval]
        
    @classmethod
    def from_checkpoints(cls, config_corrector: BaseConfigOpticDisc, 
                         config_segmentation: SegmentationConfig):
        segmentation_model = config_segmentation.model(
            num_classes=config_segmentation.num_classes,
            output_stride=config_segmentation.output_stride,
            sync_bn=False
        ).to(config_segmentation.device)
        correction_model = config_corrector.model(
            num_classes=9,
            output_stride=8,
            sync_bn=False
        ).to(config_corrector.device)
        ckpt_path_segmentation = os.path.join(config_segmentation.checkpoint_path, config_segmentation.EXPERIMENT_NAME, "checkpoint.pth")
        ckpt_path_corrector = os.path.join(config_corrector.checkpoint_path, config_corrector.EXPERIMENT_NAME, "checkpoint.pth")
        assert os.path.isfile(ckpt_path_segmentation), "Checkpoint segmentation needed"
        assert os.path.isfile(ckpt_path_corrector), "Checkpoint corrector needed"
        segmentation_model.load_state_dict(torch.load(ckpt_path_segmentation))
        correction_model.load_state_dict(torch.load(ckpt_path_corrector))
        return cls(segmentation_model=segmentation_model, 
                   correction_model=correction_model)

    def execute(self):
        
        for idx, data in enumerate(tqdm.tqdm(self._data_loader)):
            image, labels = [d.to(self._cfg.device) for d in data]
            seg_out = self._segmentation_model(image)
            correction_out = self._correction_model(image)
