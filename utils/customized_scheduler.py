import math
import warnings
from typing import List
import torch.optim as module_optim
import torch.optim.lr_scheduler as module_scheduler


class LinearWarmupCosineAnnealingLR(module_scheduler._LRScheduler):
    """
    Sets the learning rate of each parameter group to follow a linear warmup schedule
        between warmup_start_lr and base_lr followed by a cosine annealing schedule between
        base_lr and lr_eta_min.

    Inputs:
        optimizer <torch.optim.Optimizer>: Wrapped optimizer.
        warmup_epochs <python int>: Maximum number of iterations for linear warmup
        max_epochs <python int>: Maximum number of iterations
        warmup_start_lr <python float>: Learning rate to start the linear warmup. Default: 0.
        lr_eta_min <python float>: Minimum learning rate. Default: 0.
        last_epoch <python int>: The index of last epoch. Default: -1.

    Warning:
        1. It is recommended to call :func:`.step()` for :class:`LinearWarmupCosineAnnealingLR`
            after each iteration as calling it after each epoch will keep the starting lr at
            warmup_start_lr for the first epoch which is 0 in most cases.
        2. Passing epoch to :func:`.step()` is being deprecated and comes with an EPOCH_DEPRECATION_WARNING.
            It calls the :func:`_get_closed_form_lr()` method for this scheduler instead of
            :func:`get_lr()`. Though this does not change the behavior of the scheduler, when passing
            epoch param to :func:`.step()`, the user should call the :func:`.step()` function before calling
            train and validation methods.

    Usage:
        >>> layer = nn.Linear(10, 1)
        >>> optimizer = Adam(layer.parameters(), lr=0.02)
        >>> scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)
        >>> # the default case
        >>> for epoch in range(40):
        ...     # train(...)
        ...     # validate(...)
        ...     scheduler.step()
        >>> # passing epoch param case
        >>> for epoch in range(40):
        ...     scheduler.step(epoch)
        ...     # train(...)
        ...     # validate(...)
    """
    def __init__(
        self,
        optimizer: module_optim.Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        lr_eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:

        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.lr_eta_min = lr_eta_min

        super(LinearWarmupCosineAnnealingLR,
              self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) /
                (self.warmup_epochs - 1) for base_lr, group in zip(
                    self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (self.last_epoch - 1 -
              self.max_epochs) % (2 *
                                  (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"] + (base_lr - self.lr_eta_min) *
                (1 - math.cos(math.pi /
                              (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs,
                                          self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) /
                          (self.max_epochs - self.warmup_epochs))) /
            (1 + math.cos(math.pi *
                          (self.last_epoch - self.warmup_epochs - 1) /
                          (self.max_epochs - self.warmup_epochs))) *
            (group["lr"] - self.lr_eta_min) + self.lr_eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch *
                (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.lr_eta_min + 0.5 * (base_lr - self.lr_eta_min) *
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) /
                          (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]