import torch
import torch.nn as nn
import math


class MCDropout(nn.Module):
    """MonteCarlo Dropout"""

    def __init__(self, p=0.3):
        super().__init__()
        self.p = p
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        return self.dropout(x)


class ARDLinear(nn.Module):
    """Linear with ARD normalization"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        return nn.functional.linear(x, self.weight)

class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, d_model=64, nhead=4, num_layers=6, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.input_size = input_size

        # init ARD parameter
        self.log_lambda = nn.Parameter(torch.zeros(input_size))

        # input linear with ard
        # inputSize ->64
        self.ard_linear = ARDLinear(input_size, d_model)
        self.input_proj = nn.Sequential(
            self.ard_linear,
            MCDropout(p=dropout),
            nn.LeakyReLU()
        )
        # Position encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # attention block
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # output mlp
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LeakyReLU(),
            MCDropout(p=dropout),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            MCDropout(p=dropout),
            nn.Linear(32, output_size)
        )

    def reg_loss(self):
        """Loss of ARD normalization"""
        lambda_params = torch.exp(self.log_lambda)  # (input_size,)
        W = self.ard_linear.weight  # (d_model, input_size)

        #  0.5*(sum(λ||W||^2) - sum(logλ))
        reg_term = 0.5 * (
                torch.sum(lambda_params * torch.sum(W ** 2, dim=0)) -
                torch.sum(self.log_lambda)
        )
        return reg_term

    def forward(self, src):
        #  (batch, 1, input_size) -> (1, batch, d_model)
        src = self.input_proj(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src.permute(1, 0, 2))  # (seq_len, batch, d_model)
        memory = self.transformer(src)
        output = self.output_layer(memory.mean(dim=0))
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x):
        return x + self.pe[:x.size(0)]


class CombinedArdTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = TransformerModel(input_size=22, output_size=7)

    def forward(self, features, x, x_norm):
        features = features.unsqueeze(1)

        a_m_b = self.transformer(features)
        a = a_m_b[:, 0].unsqueeze(-1)
        m = a_m_b[:, 1].unsqueeze(-1)
        b = a_m_b[:, 2].unsqueeze(-1)
        c = a_m_b[:, 3].unsqueeze(-1)
        d = a_m_b[:, 4].unsqueeze(-1)
        e = a_m_b[:, 5].unsqueeze(-1)

        x = torch.clamp(x, min=1e-10)
        output = a * (x ** m) + b * x + c * x_norm ** 3 + d * x_norm ** 2 + e
        # output = a * (x ** m) + b * x_norm**3 + c*x_norm**2 + d*x+e
        # output = a * (x ** m) + b
        return output

    def reg_loss(self):
        return 0.0001 * self.transformer.reg_loss()

