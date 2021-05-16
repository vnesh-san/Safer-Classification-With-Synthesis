import pytorch_lightning as pl
import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric

class Accuracy(Metric):
    """Accuracy Metric with a hack."""
    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # update metric states
        # preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        # compute final result
        return self.correct.float() / self.total


class ImageGeneratingLogger(pl.Callback):
    def __init__(self, num_samples=10):
        super().__init__()
        self.num_samples = num_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        noise = torch.randn(self.num_samples, pl_module.feature_dim, device=pl_module.device)
        class_dcgans = pl_module.class_noise_convertor
        for key, model in class_dcgans.items():
            gen_imgs = pl_module.generator(model(noise).view(self.num_samples, -1, 1, 1))
            pred = pl_module(gen_imgs).cpu().numpy()
            trainer.logger.experiment.log({
                "Generator_"+key: [wandb.Image(img, caption=f"Pred:{y_pred}") for img, y_pred in zip(gen_imgs, pred)]
            })


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()
