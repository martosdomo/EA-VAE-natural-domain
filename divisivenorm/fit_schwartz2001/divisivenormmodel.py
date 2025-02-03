###############################################################################
# Copyright 2024 Ferenc Csikor
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
###############################################################################
import torch
import pytorch_lightning as pl


class DivisiveNormModel(pl.LightningModule):
    def __init__(self, xdim, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.optimizer = getattr(torch.optim, self.hparams.optimizer)

        self.w_unsign = getattr(torch, self.hparams.w_unsign)

        self.signed_w = torch.nn.Parameter(
            torch.normal(self.hparams.signed_w_initial_mean,
                         self.hparams.signed_w_initial_std,
                         size=(xdim, xdim)))

        self.sigma_square_unsign = getattr(torch,
                                           self.hparams.sigma_square_unsign)

        self.signed_sigma_square = torch.nn.Parameter(
            torch.normal(self.hparams.signed_sigma_square_initial_mean,
                         self.hparams.signed_sigma_square_initial_std,
                         size=(1,)))

        self.validation_step_outputs = []

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.validation_step_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        all_losses = torch.stack(self.validation_step_outputs)
        self.log('val_loss', torch.mean(all_losses), prog_bar=True,
                 logger=True)
        self.validation_step_outputs.clear()

    def _common_step(self, batch, batch_idx):
        x, y = batch

        w = self.w_unsign(self.signed_w)
        x_square = torch.mul(x, x)
        w_times_x_square = torch.matmul(x_square, w)

        sigma_square = self.sigma_square_unsign(self.signed_sigma_square)

        denominator = w_times_x_square + sigma_square
        denominator = torch.sqrt(denominator)

        x = torch.div(x, denominator)

        loss = torch.nn.MSELoss()
        return loss(x, y)

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.hparams.lr)
        return optimizer
