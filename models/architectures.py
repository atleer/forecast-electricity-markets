# %%
import torch
import torch.nn as nn
from dataclasses import dataclass


class Seq2SeqGRU(nn.Module):
    def __init__(self, enc_input_size: int, dec_input_size: int, hidden_size: int = 64, num_layers: int = 1, device: str = 'cpu'):
        super().__init__()
        self.encoder = nn.GRU(input_size=enc_input_size, hidden_size=hidden_size, batch_first=True, device=device)
        self.decoder = nn.GRU(input_size=dec_input_size, hidden_size=hidden_size, batch_first=True, device=device)
        self.fc = nn.Linear(hidden_size, dec_input_size, device=device)

    def forward(self, X: torch.Tensor, horizon: int, y_teacher: torch.Tensor | None = None, teacher_threshold: float = 0.5):

        # encode input
        enc_output, hidden = self.encoder(X)

        # seed for decoder: the last observed price
        dec_input = X[:, -1:, -1:] # dims: (batch, 1, 1)

        predictions = []
        for time_step in range(horizon):
            # decode
            dec_output, hidden = self.decoder(dec_input, hidden)

            # make prediction from output of decoder
            prediction = self.fc(dec_output)
            predictions.append(prediction)

            apply_teacher_forcing = (y_teacher is not None 
                                     and torch.rand(1).item() < teacher_threshold)

            if apply_teacher_forcing:
                dec_input = y_teacher[:, time_step:time_step+1].unsqueeze(1)
            else:
                dec_input = prediction


        # concatinate over dim 1 so that horizon is on second dimension and batches on first
        return torch.cat(predictions, dim=1)
    
class Transformer(nn.Module):
    def __init__(self, 
                 enc_input_size: int,
                 dims_model: int, # attention dimensions, also referred to as embedding dim
                 num_heads: int,
                 num_layers: int,
                 horizon: int = 1,
                 activation_fun: str = 'relu',
                 learning_rate: float = 1E-3):
        super().__init__()
        self.input_project = nn.Linear(self.enc_input_size, self.dims_model, bias=False) # TODO: Check why bias needs to be false
        self.positional_encoder = PositionalEncoding(self.dims_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            dim_model = self.dim_model,
            nhead = self.num_heads,
            activation=self.activation_function,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers = self.num_layers
        ) # copy the encoder layer num_layers times

        self.decoder = nn.Sequential(
            nn.Linear(self.dim_model, 64),
            nn.ReLU(),
            nn.Linear(100, self.horizon)
        )

    def forward(self, X: torch.Tensor):
        X_ = self.input_project(X)

        X_ = self.positional_encoder(X_)

        X_ = self.transformer_encoder(X_)

        dec_output = self.decoder(X_)

        # unfold
        y = torch.cat(X[:, 1:, :], dim=1).squeeze(-1).unfold(1, y.size(1), 1)

        return dec_output, y

class PositionalEncoding(nn.Module):

    def __init__(self, dims_model, max_len = 1000):
        super().__init__()

        self.positional_enc = torch.zeros(1, max_len, dims_model) # TODO: why the singleton dimension in the beginning?

        numerator = torch.arange(1, max_len, dtype=torch.float32).reshape(-1,1)
        denominator = torch.pow(10000, torch.arange(0, dims_model, 2, dtype=torch.float32) / dims_model)

        self.positional_enc[:, :, 0::2] = torch.sin(numerator/denominator) # for every 2i (even) position
        self.positional_enc[:, :, 1::2] = torch.cos(numerator/denominator) # for every 2i + 1 (odd) position

    def forward(self, X):
        X += self.positional_enc[:, :X.shape[1], 1].to(X.device)
        return X
