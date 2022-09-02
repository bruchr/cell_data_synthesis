from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.utils import tensor2np


class Visualizer():
    """Visualize the loss of CycleGAN"""

    def __init__(self, opt):
        logdir = Path("Optimization/logs").joinpath(opt["experiment"])
        self.writer = SummaryWriter(logdir)
        self.loss_groups = {}

    def write_losses(self, loss, epoch):
        for group, loss_names in self.loss_groups.items():
            loss_group = {}
            for key, loss_name in enumerate(loss_names):
                loss_group[loss_name] = loss[loss_name]
                self.writer.add_scalar(group+"/"+loss_name, loss[loss_name], epoch)
            #self.writer.add_scalars(group, loss_group, epoch)
            #self.writer.close()
        self.writer.close()

    def write_images(self, images, epoch):
        for key, image in images.items():
            image_np = tensor2np(torch.squeeze(image[0,:]), "uint8")
            if image_np.ndim == 2:
                self.writer.add_image(key, image_np, epoch, dataformats="HW")
            elif image_np.ndim == 3:
                mid_ind = int(np.floor(image_np.shape[0]/2))
                self.writer.add_image(key, image_np[mid_ind,...], epoch, dataformats="HW")
            else:
                self.writer.add_image(key, image_np, epoch)
        self.writer.close()

    def define_loss_group(self, group_name, group_items):
        self.loss_groups[group_name] = group_items
