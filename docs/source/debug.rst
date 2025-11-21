.. _debug:

Debug Guide
===========

Since ``torch_blue`` is specifically aimed at users new to Bayesian neural networks
this page gathers some of the basic debugging steps we keep using for models that either
do not train or do not train properly. If you discover another method we would be happy
if shared it with the community either by making an issue or a pull request to have it
added here.

1. **Non-Bayesian Debugging**
    A very common and important sanity check is setting the variational distribution,
    prior and predictive distribution to ``NonBayesian()``. This should make your model
    equivalent to the non-Bayesian variant. If it does not something is wrong :). With
    that you can apply any approach you would use to debug your network, e.g. train on
    only one sample or batch to make it overfit (ensuring it can learn anything). It is
    typically easier to get the non-Bayesian version to train and the Bayesian is highly
    unlikely to work unless the non-Bayesian does.

.. _loss:

2. **Loss decomposition**
    An idiosyncrasy in VI is that the loss has two terms with distinct functions: the
    data-fitting term, which is closely related to non-Bayesian losses, and the
    prior-matching term, which most resembles a non-Bayesian regularisation term. All
    ``torch_blue`` losses have integrated loss tracking by setting ``track=True``. This
    will automatically track the two loss components separately. Plotting them
    separately can sometimes help to narrow down the issue. While this is more of an art
    than a science (just like understanding non-Bayesian loss curves) here are some
    pointers to get you started:

    - Keep in mind that the data fitting term is closely related to a non-Bayesian loss
      so many concepts are transferable. A common pattern looks much like overfitting
      in this term while the overall loss still drops. For that issue continue to
      :ref:`"Adjusting the heat" <heat>`.
    - The relative magnitude of the data fitting and prior term tend to be important
      quite often, particularly if the prior matching term is dominant. To adjust this
      balance keep in mind that the data fitting term becomes more important with more
      features per sample and larger datasets. The prior matching term becomes more
      important the more parameters your model has. This is not a bug but a feature!
      Occam's razor is build into the Bayesian paradigm and will punish you if you
      overparametrize your model too much. You can attempt to cheat this by
      :ref:`"Adjusting the heat" <heat>`.
    - Another reason for the prior matching term can be mistakes in
      :ref:`"Setting the prior" <prior>`.
    - It seems quite common that the data fitting term converges much faster and settle
      while the prior matching term is still improving. While it seems reasonable to
      assume that convergence of the prior matching term is important for calibration
      we still need to verify this.

.. _heat:

3. **Adjusting the heat**
    An important factor in the fitting of BNNs is the balance between the two loss terms
    (see :ref:`"Loss decomposition" <loss>`). Poor convergence often occurs when the
    prior matching term is too dominant. While this is quite heavy-handed and
    mathematically at least debatable, adjusting (typically lowering) the heat is the
    fastest way to fix this issue. There are two core reasons for this issue, which you
    should at least consider before taking the brute force option of adjusting the heat:

    - The model is overparametrized, i.e. either you model ist too big or your dataset
      too small (relative to each other).
    - You have chosen a prior that is too far away from the actual weight distribution.
      :ref:`"Setting the prior" <prior>` provides some guidance on this.

    If you do lower the heat, check the relative magnitude of the terms (via setting
    ``track=True`` in ``torch_blue`` losses) and aim for the prior matching term to be
    within an order of magnitude or two of the data fitting term. When you make this
    adjustment the predictive quality tends to be a poor indicator for the right value,
    since it tends to increase the lower the heat is set. However, the calibration of
    the uncertainty estimates also tends to deteriorate. Therefore, if you do this
    evaluate the calibration or make a conscious trade-off between calibration and
    predictive accuracy.

.. _prior:

4. **Setting the prior and initial standard deviation**
    There is a very extensive discussion on setting priors by the Stan development Team
    `here <https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations>`__.
    Most relevant is maybe the somewhat counterintuitive fact that a prior with large
    standard deviation is not necessarily non-informative if it gives undue preference
    to unrealistically large values. This is almost allways relevant for BNNs since -
    due to the normalization requirements that also form the basis of Kaiming
    initialization - the expected magnitude of the sampled weights should scale with
    one over the square root of the layer width to keep the network output from becoming
    unreasonably large. Therefore, standard deviations need to be scaled down by the
    same factor during initialization. This behavior is controlled by the
    ``kaiming_init`` argument from :class:`VIkwargs <torch_blue.vi.VIkwargs>` that can
    be provided to all layers and is ``True`` by default. However, you might want your
    priors to also be rescaled in this fashion to respect this effect using the
    ``rescale_prior`` argument, that is also in
    :class:`VIkwargs <torch_blue.vi.VIkwargs>`, but ``False`` by default (this might
    change in the future).
    Additionally, if you loss becomes infinite or your outputs unreasonably large during
    the first few epochs (or batches), try lowering the initial standard deviation
    specified by the variational distribution.
