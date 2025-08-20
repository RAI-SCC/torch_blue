from typing import OrderedDict, overload

import torch
from torch import Tensor
from torch.nn import Module, Sequential

from .base import VIModule


class VISequential(VIModule, Sequential):
    """
    Sequential container for :class:`~.VIModule`.

    Equivalent to :class:`nn.Sequential`, that manages :class:`~.VIModule` too. See its
    `documentation <https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html>`__
    for usage.

    Detects and aggregates prior_log_prob and variational_log_prob from submodules, if
    needed. Then passes on only the output to the next module. This makes mixed
    sequences of :class:`~.VIModule` and :class:`nn.Module` work with and without
    ``return_log_probs``.
    """

    @overload
    def __init__(self, *args: Module) -> None: ...

    @overload
    def __init__(self, arg: OrderedDict[str, Module]) -> None: ...

    def __init__(self, *args) -> None:  # type: ignore
        super().__init__()
        log_prob_set = False
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                if hasattr(module, "_return_log_probs"):
                    if not log_prob_set:
                        self._return_log_probs = module._return_log_probs
                        log_prob_set = True
                    else:
                        assert (
                            self._return_log_probs == module._return_log_probs
                        ), "return_log_probs has to be equal for all Modules"
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                if hasattr(module, "_return_log_probs"):
                    if not log_prob_set:
                        self._return_log_probs = module._return_log_probs
                        log_prob_set = True
                    else:
                        assert (
                            self._return_log_probs == module._return_log_probs
                        ), "return_log_probs has to be equal for all Modules"
                self.add_module(str(idx), module)

        # if no module is a VIModule return_log_probs is turned off
        if not log_prob_set:
            self._return_log_probs = False

    def forward(self, input_):  # type: ignore
        """
        Forward pass that manages log probs, if required.

        Parameters
        ----------
        input_ : Varies
            Input for the first module in the stack. Passed on to it unchanged.

        Returns
        -------
        output: Varies
            Output of the module stack.
        log_probs: Tensor
            Tensor of shape (2,) containing the total prior and variational log
            probability (in that order) of all sampled weights and biases.

            Only returned if ``return_log_probs``. Otherwise, only **output** is returned.
        """
        if self._return_log_probs:
            total_log_probs = torch.tensor([0.0, 0.0], device=input_.device)
            for module in self:
                if isinstance(module, VIModule):
                    input_, log_probs = module(input_)

                    total_log_probs = total_log_probs + log_probs
                else:
                    input_ = module(input_)
            return input_, total_log_probs
        else:
            for module in self:
                input_ = module(input_)
            return input_


class VIResidualConnection(VISequential):
    """
    A version of :class:`~.VISequential` that supports residual connections.

    This class is identical to :class:`~.VISequential`, but adds the input to the
    output. Importantly, it manages log prob tracking, if required. Note that a single
    module can also be wrapped to add a residual connection around it.

    Raises
    ------
    :exc:`RuntimeError`
        If the output shape does not match the input shape during the forward pass.
    """

    def forward(self, input_):  # type: ignore
        """
        Forward pass that manages log probs, if required, and adds the input to the output.

        Parameters
        ----------
        input_ : Varies
            Input for the first module in the stack. Passed on to it unchanged.

        Returns
        -------
        output: Varies
            Output of the module stack plus the input to the residual connection.
        log_probs: Tensor
            Tensor of shape (2,) containing the total prior and variational log
            probability (in that order) of all sampled weights and biases.

            Only returned if ``return_log_probs``. Otherwise, only **output** is returned.
        """
        if self._return_log_probs:
            output, log_probs = super().forward(input_)
            return self._safe_add(input_, output), log_probs
        else:
            output = super().forward(input_)
            return self._safe_add(input_, output)

    @staticmethod
    def _safe_add(input_: Tensor, output_: Tensor) -> Tensor:
        try:
            return output_ + input_
        except RuntimeError as e:
            if str(e).startswith("The size of tensor a"):
                raise RuntimeError(
                    f"Output shape ({output_.shape}) of residual connection must match input shape ({input_.shape})"
                )
            else:
                raise e  # pragma: no cover
