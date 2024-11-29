import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np


class PositionalEncoding(nn.Module):
    """Positional encoding as used in transformers.

    NOTE: This implementation was copied from pytorch's official documentation:
    https://pytorch.org/tutorials/beginner/translation_transformer.html

    """

    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class SeqTransformer(pl.LightningModule):

    def __init__(self,
                 ntoken,
                 d_model,
                 nhead,
                 d_hid,
                 nlayers,
                 learning_rate,
                 max_sequence_len,
                 distance_mlp_layer_sizes,
                 dropout=0.1):
        super().__init__()
        self.save_hyperparameters()

        self.pos_encoder = PositionalEncoding(d_model,
                                              dropout,
                                              max_len=max_sequence_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid,
                                                    dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)
        # self.final_activation = nn.LogSoftmax(dim=2)

        self.distance_mlp = nn.Sequential()
        for i, o in distance_mlp_layer_sizes:
            self.distance_mlp.append(nn.Linear(i, o))
            self.distance_mlp.append(nn.ReLU())

        self.n_tokens = ntoken
        self.learning_rate = learning_rate

    def forward(self, source, temperature=1):
        # Because our data comes batch first
        source = torch.transpose(source, 0, 1)

        source = self.embedding(source)
        source = self.pos_encoder(source)
        # TODO: padding_mask
        output = self.transformer_encoder(source)
        output = self.linear(output)
        output = torch.transpose(output, 0, 1)
        pairwise_sums = output[:, :, None, :] + output[:, None, :, :]
        distance_predictions = self.distance_mlp(pairwise_sums)
        return distance_predictions

    def training_step(self, batch, _):
        training_loss = self._compute_loss(batch)
        self.log('loss',
                 training_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        return {'loss': training_loss}

    def validation_step(self, batch, _):
        """Equal to training step but used for validation."""
        val_loss = self._compute_loss(batch)
        # Log the validation loss to the progress bar.
        self.log('val_loss',
                 val_loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        return {'val_loss': val_loss}

    def _compute_loss(self, batch):
        sequences = batch['tokenized_sequences']
        distances = batch['distances']
        preds = self.foward(sequences)
        return nn.MSELoss()(preds, distances)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
