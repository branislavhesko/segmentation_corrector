import os

import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group

from configuration.optic_disc import BaseConfigOpticDisc
from train import Trainer


class Params:
    world_size = torch.cuda.device_count()


def train(gpu, args):
    init_process_group(backend='nccl', init_method='env://', world_size=2, rank=gpu)
    torch.manual_seed(42)
    config = _get_config(args, gpu)
    trainer = Trainer(config=config)
    trainer.train()


def _get_config(args, gpu):
    config = BaseConfigOpticDisc()
    config.device = gpu
    config.parallel = True
    config.world_size = args.worls_size
    return config


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29502"
    mp.spawn(train, nprocs=Params.world_size, args=(Params(),), join=True, daemon=False)
    print("This is segmentation corrector distributed training. Thank you!")
