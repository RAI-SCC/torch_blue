torch_blue -- A PyTorch-based framework for Bayesian learning and uncertainty estimation
========================================================================================

Release: |release|

``torch_blue`` provides a simple way for non-expert users to implement and train
Bayesian Neural Networks (BNNs). Currently, it only supports Variational Inference (VI),
but will hopefully grow and expand in the future. To make the user experience as easy as
possible most components mirror components from `PyTorch <https://pytorch.org/>`_.

``torch_blue`` is available on `PyPI <https://pypi.org/project/torch-blue/>`_ under the
name ``torch-blue``. If you need further instructions, check the
:ref:`installation guide <installation>`.

To learn how you can use ``torch_blue`` you can read the
:ref:`quickstart guide <quickstart>`.
If you have implemented your model, but struggle to get it running check our
:ref:`debug guide <debug>` for some basic approaches to debugging a BNN.

If you find any bug, have a feature request, or want to contribute visit us on
`GitHub <https://github.com/RAI-SCC/torch_blue>`_.

And finally, if you are struggling to understand any part of this documentation let us
know. ``torch_blue`` is explicitly intended to be accessible for users without
experience with BNNs. If you are struggling someone else is probably struggling with
the same thing now or at least in the future. Make an issue, let us know what is
unclear or confusing so we can try to improve it and make BNNs accessible to everyone.

.. toctree::
   :hidden:

   Home <self>
   installation
   getting_started
   debug
   autoapi/index
