"""Module for sequence folding"""
import torch
from torch import nn
import lightning as l


class Folder(l.LightningModule):

    def __init__(self,
                 target_distance_matrix,
                 learning_rate=5e-2,
                 log_positions=True):
        super().__init__()
        self.learning_rate = learning_rate
        self.length = target_distance_matrix.shape[0]
        positions = [torch.zeros(
            (1, 2))] + [torch.rand(1, 2) for _ in range(self.length - 1)]
        self.positions = nn.Parameter(torch.cat(positions))
        self.target_distance_matrix = target_distance_matrix

        self.log_positions = log_positions
        self.positions_history = []
        if self.log_positions:
            self.positions_history.append(
                self.positions.clone().detach().numpy())

    def forward(self):
        return torch.norm(self.positions[:, None] - self.positions[None, :],
                          dim=-1)

    def training_step(self, _, __):
        training_loss = self._compute_loss()
        self.log('loss',
                 training_loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True)

        if self.log_positions:
            self.positions_history.append(
                self.positions.clone().detach().numpy())
        return training_loss

    def _compute_loss(self):
        distances = self.forward()
        return nn.MSELoss()(distances, self.target_distance_matrix)

    def configure_optimizers(self):
        return torch.optim.LBFGS(self.parameters(),
                                 lr=self.learning_rate,
                                 max_iter=1)
