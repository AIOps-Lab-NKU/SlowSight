from typing import List

from math import cos
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupLR(_LRScheduler):
    """
    A learning rate scheduler with warm-up strategy.

    The warm-up phase gradually increases the learning rate from a small value to the base learning rate
    at the beginning of training, which helps stabilize training in early stages.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_method (str): Method of warm-up, options include 'constant', 'linear', 'exp'. Default is 'linear'.
        warmup_iters (int): Number of iterations for warm-up. Default is 1000.
        warmup_ratio (float): Ratio of the initial learning rate to the base learning rate. Default is 0.001.
        last_epoch (int): The index of the last epoch. Used for resuming training. Default is -1.
    """

    def __init__(
            self,
            optimizer: Optimizer,
            warmup_method: str = "linear",
            warmup_iters: int = 1000,
            warmup_ratio: float = 0.001,
            last_epoch: int = -1,
    ):
        # Validate the warmup method input
        if warmup_method is not None:
            if warmup_method not in ['constant', 'linear', 'exp']:
                raise ValueError(
                    f'"{warmup_method}" is not a supported type for warming up, valid'
                    ' types are "constant", "linear" and "exp"')
        if warmup_method is not None:
            assert warmup_iters > 0, \
                '"warmup_iters" must be a positive integer'
            assert 0 < warmup_ratio <= 1.0, \
                '"warmup_ratio" must be in range (0,1]'

        self.warmup_ratio = warmup_ratio
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.by_iter = True
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Get the current learning rates based on the current epoch or iteration.

        If warm-up phase is completed, use the regular LR schedule.
        Otherwise, apply the warm-up learning rate.
        """
        if self.warmup_method is None or self.last_epoch >= self.warmup_iters:
            return self.get_regular_lr(self.last_epoch)
        else:
            return self.get_warmup_lr(self.last_epoch)

    def get_warmup_lr(self, iter: int):
        """
        Compute the warm-up learning rate based on the current iteration.
        """
        warmup_ratio = self._get_warmup_ratio_at_iter(
            self.warmup_method, iter, self.warmup_iters, self.warmup_ratio
        )
        return [_lr * warmup_ratio for _lr in self.base_lrs]

    def get_regular_lr(self, iter: int):
        """
        Compute the regular learning rate after the warm-up phase.

        Must be implemented by subclasses.
        """
        raise NotImplemented

    @staticmethod
    def _get_warmup_ratio_at_iter(
            method: str, iter: int, warmup_iters: int, warmup_ratio: float
    ) -> float:
        """
        Calculate the ratio used to scale the learning rate during warm-up.

        Args:
            method: Type of warm-up ('constant', 'linear', 'exp').
            iter: Current iteration.
            warmup_iters: Total number of warm-up iterations.
            warmup_ratio: Starting ratio of the learning rate.

        Returns:
            Scaling factor for the learning rate.
        """
        if iter >= warmup_iters:
            return 1.0
        if method == "constant":
            return warmup_ratio
        elif method == "linear" and warmup_iters != 0:
            alpha = iter / warmup_iters
            return warmup_ratio * (1 - alpha) + alpha
        elif method == 'exp' and warmup_iters != 0:
            return warmup_ratio ** (1 - iter / warmup_iters)
        else:
            raise ValueError("Unknown warmup method: {}".format(method))


class WarmupPolyLR(WarmupLR):
    """
    A learning rate scheduler that combines warm-up and polynomial decay strategies.

    After the warm-up phase, the learning rate decays following a polynomial function.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_iters (int): Total number of training iterations.
        power (float): Exponent of the polynomial decay function. Default is 0.9.
        warmup_method (str): Warm-up method. Default is 'linear'.
        warmup_iters (int): Number of iterations for warm-up. Default is 1000.
        warmup_ratio (float): Initial LR ratio during warm-up. Default is 0.001.
        constant_ending (float): Minimum LR scaling factor when polynomial decay ends.
        min_lr (float): Minimum learning rate allowed. Default is 0.
        last_epoch (int): Index of the last epoch. Default is -1.
    """

    def __init__(
            self,
            optimizer: Optimizer,
            max_iters: int,
            power: float = 0.9,
            warmup_method: str = "linear",
            warmup_iters: int = 1000,
            warmup_ratio: float = 0.001,
            constant_ending: float = 0.0,
            min_lr: float = 0.,
            last_epoch: int = -1,
    ):
        super().__init__(optimizer, warmup_method, warmup_iters, warmup_ratio, last_epoch)
        self.max_iters = max_iters
        self.min_lr = min_lr
        self.power = power
        self.constant_ending = constant_ending

    def get_regular_lr(self, iter: int):
        """
        Compute the learning rate using polynomial decay after warm-up.

        Args:
            iter (int): Current iteration.

        Returns:
            List of learning rates for each parameter group.
        """
        if self.max_iters <= 0:
            return None
        coeff = (1.0 - iter / self.max_iters) ** self.power
        # Constant ending lr.
        if coeff < self.constant_ending:
            lr = [_lr * self.constant_ending for _lr in self.base_lrs]
        else:
            lr = [(_lr - self.min_lr) * coeff + self.min_lr for _lr in self.base_lrs]

        return lr
