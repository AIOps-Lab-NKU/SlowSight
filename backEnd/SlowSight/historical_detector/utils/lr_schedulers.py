from typing import List

from math import cos
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_method: str = "linear",
        warmup_iters: int = 1000,
        warmup_ratio: float = 0.001,
        last_epoch: int = -1,
    ):
        """
        Base class for learning rate schedulers with warmup.

        Args:
            optimizer: Wrapped optimizer.
            warmup_method: Strategy of warmup, options: 'constant', 'linear', 'exp'.
            warmup_iters: Number of iteration steps to perform warmup.
            warmup_ratio: Learning rate ratio used at the beginning of warmup.
            last_epoch: The index of last epoch.
        """
        # Validate warmup method
        if warmup_method is not None:
            if warmup_method not in ["constant", "linear", "exp"]:
                raise ValueError(
                    f'"{warmup_method}" is not a supported type for warming up, valid types are "constant", "linear" and "exp"'
                )
        if warmup_method is not None:
            assert warmup_iters > 0, '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, '"warmup_ratio" must be in range (0,1]'

        self.warmup_ratio = warmup_ratio
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.by_iter = True
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Get current learning rates based on current step and warmup status.
        """
        if self.warmup_method is None or self.last_epoch >= self.warmup_iters:
            return self.get_regular_lr(self.last_epoch)
        else:
            return self.get_warmup_lr(self.last_epoch)

    def get_warmup_lr(self, iter: int):
        """
        Calculate learning rate during warmup phase.

        Args:
            iter: Current iteration number.

        Returns:
            List[float]: Learning rates adjusted by warmup strategy.
        """
        warmup_ratio = self._get_warmup_ratio_at_iter(
            self.warmup_method, iter, self.warmup_iters, self.warmup_ratio
        )
        return [_lr * warmup_ratio for _lr in self.base_lrs]

    def get_regular_lr(self, iter: int):
        raise NotImplemented

    @staticmethod
    def _get_warmup_ratio_at_iter(
        method: str, iter: int, warmup_iters: int, warmup_ratio: float
    ) -> float:
        """
        Compute warmup ratio using specified method.

        Args:
            method: Method string ('constant', 'linear', or 'exp').
            iter: Current iteration.
            warmup_iters: Total number of warmup iterations.
            warmup_ratio: Minimum multiplier applied to base LR.

        Returns:
            float: Scaling factor for learning rate.
        """
        if iter >= warmup_iters:
            return 1.0
        if method == "constant":
            return warmup_ratio
        elif method == "linear" and warmup_iters != 0:
            alpha = iter / warmup_iters
            return warmup_ratio * (1 - alpha) + alpha
        elif method == "exp" and warmup_iters != 0:
            return warmup_ratio ** (1 - iter / warmup_iters)
        else:
            raise ValueError("Unknown warmup method: {}".format(method))


class WarmupPolyLR(WarmupLR):
    def __init__(
        self,
        optimizer: Optimizer,
        max_iters: int,
        power: float = 0.9,
        warmup_method: str = "linear",
        warmup_iters: int = 1000,
        warmup_ratio: float = 0.001,
        constant_ending: float = 0.0,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        """
        A learning rate scheduler that combines warmup and polynomial decay.

        Args:
            optimizer: Wrapped optimizer.
            max_iters: Total number of training iterations.
            power: Exponent value for polynomial decay.
            warmup_method: Warmup strategy.
            warmup_iters: Number of warmup iterations.
            warmup_ratio: Ratio of the smallest learning rate during warmup.
            constant_ending: Final LR as a ratio of the base LR.
            min_lr: Minimum learning rate to prevent division-by-zero.
            last_epoch: Index of the last epoch used for resuming training.
        """
        super().__init__(
            optimizer, warmup_method, warmup_iters, warmup_ratio, last_epoch
        )
        self.max_iters = max_iters
        self.min_lr = min_lr
        self.power = power
        self.constant_ending = constant_ending

    def get_regular_lr(self, iter: int):
        """
        Calculate learning rate after warmup phase using polynomial decay.

        Args:
            iter: Current iteration number.

        Returns:
            List[float]: Polynomial-decayed learning rates.
        """
        if self.max_iters <= 0:
            return None
        coeff = (1.0 - iter / self.max_iters) ** self.power
        # Constant ending lr
        if coeff < self.constant_ending:
            lr = [_lr * self.constant_ending for _lr in self.base_lrs]
        else:
            lr = [(_lr - self.min_lr) * coeff + self.min_lr for _lr in self.base_lrs]

        return lr
