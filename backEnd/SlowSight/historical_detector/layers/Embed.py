import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Classic positional embedding for sequence data.

        Args:
            d_model: Dimension of the model (embedding dimension)
            max_len: Maximum length of input sequences
        """
        super(PositionalEmbedding, self).__init__()
        # Compute positional encodings once in log space and register as buffer
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass to return positional embeddings.

        Args:
            x: Input tensor with shape [B, T, D]
        Returns:
            Positional embedding tensor of shape [1, T, D]
        """
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        """
        Token Embedding using Conv1d layer.

        Args:
            c_in: Input feature dimension
            d_model: Output embedding dimension
        """
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        """
        Forward pass for token embedding.

        Args:
            x: Input tensor [Batch, Time, Features]
        Returns:
            Embedded tensor [Batch, Time, d_model]
        """
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        """
        Fixed sinusoidal embedding for time-related features.

        Args:
            c_in: Number of positions (e.g., number of days/hours)
            d_model: Dimension of the embedding
        """
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        """
        Args:
            x: Long tensor of indices representing positions.
        Returns:
            Detached embeddings.
        """
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        """
        Temporal embedding based on different frequencies.

        Args:
            d_model: Dimension of the temporal embedding
            embed_type: Type of embedding ('fixed' or standard learnable)
            freq: Frequency of the input data ('t': minute, 'h': hour, etc.)
        """
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        """
        Args:
            x: Tensor containing time-related information [B, T, C]
        Returns:
            Temporal embeddings summed together
        """
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        """
        Linear projection of time features.

        Args:
            d_model: Target embedding dimension
            embed_type: Must be 'timeF'
            freq: Frequency of the input data ('h': hourly, 't': minute-level, etc.)
        """
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        """
        Args:
            x: Time features [B, T, d_inp]
        Returns:
            Embedded time features [B, T, d_model]
        """
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        """
        Combined embedding for time series data including value, position, and temporal embeddings.

        Args:
            c_in: Input feature dimension
            d_model: Model dimension
            embed_type: Type of temporal embedding
            freq: Time frequency
            dropout: Dropout rate
        """
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                  freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        Forward pass of combined data embedding.

        Args:
            x: Input time series data [B, T, N]
            x_mark: Optional temporal features [B, T, C]
        Returns:
            Embedded tensor after value, temporal, and positional embeddings
        """
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        """
        Inverted data embedding where time steps are treated as features.

        Args:
            c_in: Number of input features
            d_model: Model dimension
            dropout: Dropout rate
        """
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        Forward pass of inverted data embedding.

        Args:
            x: Input tensor [Batch, Time, Features]
            x_mark: Optional time features [Batch, Time, Temporal_Features]
        Returns:
            Embedded tensor [Batch, Features, d_model]
        """
        x = x.permute(0, 2, 1)  # Convert to [Batch, Features, Time]

        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))

        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        """
        Same as DataEmbedding but without positional encoding.

        Args:
            c_in: Input feature dimension
            d_model: Model dimension
            embed_type: Type of temporal embedding
            freq: Time frequency
            dropout: Dropout rate
        """
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        Forward pass without positional encoding.

        Args:
            x: Input tensor [Batch, Time, Features]
            x_mark: Temporal features [Batch, Time, C]
        Returns:
            Embedded tensor with temporal features
        """
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        """
        Patch embedding layer for long-term time series.

        Args:
            d_model: Dimension of the output
            patch_len: Length of each patch
            stride: Stride between patches
            padding: Padding applied before unfolding
            dropout: Dropout rate
        """
        super(PatchEmbedding, self).__init__()
        # Patching parameters
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Linear embedding for patched data
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for patch embedding.

        Args:
            x: Input tensor [B, N, T]
        Returns:
            Patch-embedded tensor and number of variables N
        """
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars
