# %%
import torch
import torch.nn as nn
from dataclasses import dataclass


class Seq2SeqGRU(nn.Module):
    def __init__(self, 
                 enc_input_size: int, 
                 dec_input_size: int, 
                 hidden_size: int = 64, 
                 num_layers: int = 1, 
                 horizon: int = 24,
                 teacher_threshold: float = 0.5,
                 device: str = 'cpu'):
        super().__init__()
        self.encoder = nn.GRU(input_size=enc_input_size, hidden_size=hidden_size, batch_first=True, device=device)
        self.decoder = nn.GRU(input_size=dec_input_size, hidden_size=hidden_size, batch_first=True, device=device)
        self.fc = nn.Linear(hidden_size, dec_input_size, device=device)
        self.teacher_threshold = teacher_threshold
        self.horizon = horizon # horizon: number of time steps into the future to forecast


    def forward(self, X: torch.Tensor, y: torch.Tensor | None = None):

        # encode input
        enc_output, hidden = self.encoder(X)

        # seed for decoder: the last observed price
        dec_input = X[:, -1:, -1:] # dims: (batch, 1, 1)

        predictions = []
        for time_step in range(self.horizon):
            # decode
            dec_output, hidden = self.decoder(dec_input, hidden)

            # make prediction from output of decoder
            prediction = self.fc(dec_output)
            predictions.append(prediction)

            apply_teacher_forcing = (self.training and y is not None 
                                     and torch.rand(1).item() < self.teacher_threshold)

            if apply_teacher_forcing:
                dec_input = y[:, time_step:time_step+1]
            else:
                dec_input = prediction

        # concatinate over dim 1 so that horizon is on second dimension and batches on first
        return torch.cat(predictions, dim=1)

    def predict(self, X: torch.tensor):
        with torch.no_grad():
            y_pred = self.forward(X)
        return y_pred

    def loss_targets(self, X, y):
        """For model agnostic training; return the target that the prediction should be compared to"""
        return y
    
class Transformer(nn.Module):
    def __init__(self, 
                 enc_input_size: int, # number of features in the input sequence
                 dim_model: int, # attention dimensions, also referred to as embedding dim
                 num_heads: int,
                 num_layers: int,
                 horizon: int = 1,
                 activation_fun: str = 'relu',
                 learning_rate: float = 1E-3):
        super().__init__()
        self.enc_input_size = enc_input_size
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.horizon = horizon
        self.activation_fun = activation_fun
        self.learning_rate = learning_rate
        self._mask = None

        self.input_project = nn.Linear(self.enc_input_size, self.dim_model, bias=False) # bias is false because it is essentially just a reshape
        self.positional_encoder = PositionalEncoding(self.dim_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.dim_model,
            nhead = self.num_heads,
            activation=self.activation_fun,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers = self.num_layers
        ) # copy the encoder layer num_layers times

        self.decoder = nn.Sequential(
            nn.Linear(self.dim_model, 100),
            nn.ReLU(),
            nn.Linear(100, self.horizon)
        )

    def _create_square_mask(self, seq_len):
        if self._mask is None:
            mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
            mask[mask == 0] = float("-inf")
            mask[mask == 1] = float(0.0)
            self._mask = mask
        return self._mask

    def forward(self, X: torch.Tensor, y: torch.Tensor | None = None):
        mask = self._create_square_mask(X.shape[1]).to(X.device)

        X_ = self.input_project(X)

        X_ = self.positional_encoder(X_)

        X_ = self.transformer_encoder(X_, mask)

        dec_output = self.decoder(X_)        

        return dec_output

    def loss_targets(self, X: torch.Tensor, y: torch.Tensor):
        """
        For model agnostic training; return the target that the prediction should be compared t. 
        Want to compare all predictions to all targets, not just last prediction to last target
        """
        y = torch.cat([X[:, 1:, -1:], y], dim=1).squeeze(-1).unfold(1, y.size(1), 1)
        return y

    def predict(self, X: torch.Tensor):
        with torch.no_grad():
            y_pred = self.forward(X)
            y_pred = y_pred[:, -1, :].unsqueeze(1) # for prediction we only need the prediction of the last position
        return y_pred

class PositionalEncoding(nn.Module):

    def __init__(self, dim_model, max_len = 1000):
        super().__init__()

        self.positional_enc = torch.zeros(1, max_len, dim_model) # TODO: why the singleton dimension in the beginning?

        numerator = torch.arange(0, max_len, dtype=torch.float32).reshape(-1,1) # TODO: should this start at 1?
        denominator = torch.pow(10000, torch.arange(0, dim_model, 2, dtype=torch.float32) / dim_model)

        self.positional_enc[:, :, 0::2] = torch.sin(numerator/denominator) # for every 2i (even) position
        self.positional_enc[:, :, 1::2] = torch.cos(numerator/denominator) # for every 2i + 1 (odd) position

    def forward(self, X):
        X = X + self.positional_enc[:, :X.shape[1], :].to(X.device)
        return X
