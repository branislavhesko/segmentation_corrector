import logging

import torch


class FocalLoss(torch.nn.Module):

    def __init__(self, alfa=2., beta=4., eps=1e-5):
        super().__init__()
        self._alfa = alfa
        self._beta = beta
        self._eps = eps

    def forward(self, labels, output):
        loss_point = torch.mean((1 - output[
            labels == 1.]) ** self._alfa * torch.log(output[labels == 1.] + self._eps))
        loss_background = torch.mean((1 - labels) ** self._beta * output ** self._alfa * torch.log(1 - output + self._eps))
        return -1 * (loss_point + 5 * loss_background)


class HardNegativeFocalLoss(torch.nn.Module):
    kernel_size = 11
    def __init__(self, alfa=2., beta=4., eps=1e-5) -> None:
        super().__init__()
        self._alfa = alfa
        self._beta = beta
        self._eps = eps

    def forward(self, labels, output):
        pool = torch.nn.MaxPool2d(kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
        labels_enlarged = pool(labels.float()).long()
        labels_enlarged = labels_enlarged - labels
        loss_background = torch.mean((1 - labels[labels_enlarged == 1]) ** self._beta * output[
            labels_enlarged == 1] ** self._alfa * torch.log(1 - output[labels_enlarged == 1] + self._eps))
        return -1 * loss_background


class TotalLoss(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self._focal_loss = FocalLoss(alfa=config.alfa, beta=config.beta)
        self._focal_loss_hard = HardNegativeFocalLoss(alfa=config.alfa, beta=config.beta)

    def forward(self, output, labels):
        return self._focal_loss(labels, output) # + self._focal_loss_hard(labels, output)


if __name__ == "__main__":
    x = FocalLoss()
    labels = torch.zeros(1, 32, 32)
    labels[0, 11, 10] = 1.
    output = torch.zeros(1, 32, 32)
    output[0, 11, 10] = 10.

    print(x(labels, output))