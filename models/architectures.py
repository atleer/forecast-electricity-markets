import torch
import torch.nn as nn

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