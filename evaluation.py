import torch

from configuration.config import Config, ConfigOpticDisc


def iou(gt, pred):
    return torch.sum(gt & pred) / torch.sum(gt | pred)


def dice(gt, pred):
    return 2 * torch.sum(gt & pred) / (torch.sum(gt) + torch.sum(pred))



class Evaluator:
    
    def __init__(self, segmentation_model, correction_model):
        pass