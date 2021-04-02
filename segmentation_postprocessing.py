import torch


class PostProcessor:

    def __init__(self, map_threshold: float = 0.5):
        self._threshold = map_threshold

    def execute(self, image: torch.Tensor, segmentation_output: torch.Tensor, corrector_output: torch.Tensor):
        corrector_output_thresholded = (segmentation_output > 0) & (corrector_output > self._threshold)
        in_out_mask = self._get_in_out_mask(corrector_output)
        correction = corrector_output_thresholded * in_out_mask
        return correction

    def _get_in_out_mask(self, corrector_output):
        pass


class Visualization:

    @staticmethod
    def visualize(image, segmentation_output, corrector_output):
        pass