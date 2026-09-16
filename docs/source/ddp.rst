.. _ddp:

``torch_blue`` and PyTorch DDP
==============================

To accelerate training and evaluation of larger models ``torch_blue`` is compatible with
PyTorch's `DistributedDataParallel (DDP) wrapper <https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html>`_.
However, there is some specific sequencing to consider, when using ``torch_blue``'s
auto-conversion capability. The DDP wrapper always needs to be applied after the
conversion to a VIModule. Additionally, the optimizer needs to be initialized after both
of these steps are performed (this is a general interaction between optimizers and DDP):

.. code-block:: python3

    import torch.optim as optim
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch_blue.vi import convert_to_vimodule

    vi_model = convert_to_vi_module(model.to(rank))
    ddp_vi_model = DDP(vi_model, device_ids=[rank])

    optimizer = optim.SGD(ddp_vi_model.parameters(), lr=0.001)

For guidance on how to obtain ``rank`` in this example please refer to the PyTorch
`tutorial for DDP <https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html>`_.
