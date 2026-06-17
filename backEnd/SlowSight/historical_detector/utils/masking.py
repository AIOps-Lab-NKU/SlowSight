import torch


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        """
        Create a causal mask for self-attention to prevent attending to future positions.

        Args:
            B (int): Batch size.
            L (int): Sequence length.
            device (str): Device on which the mask is created ('cpu' or 'cuda').
        """
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            # Upper triangular matrix where diagonal=1 keeps the lower triangle of shape [L, L]
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask:
    def __init__(self, B, H, L, index, scores, device="cpu"):
        """
        Generate a probability-based attention mask based on indices.

        This class creates a mask that excludes certain attention positions in the score matrix,
        typically used in sparse attention mechanisms.

        Args:
            B (int): Batch size.
            H (int): Number of attention heads.
            L (int): Sequence length.
            index (torch.Tensor): Indices indicating valid positions for each head and batch.
            scores (torch.Tensor): Attention scores tensor; mask will match its shape.
            device (str): Device for mask creation ('cpu' or 'cuda').
        """
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
