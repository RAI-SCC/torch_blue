---
title: 'torch_bayesian: A Flexible Python Package for Bayesian Neural Networks in PyTorch'
tags:
    - Python
    - Bayesian Neural Networks
    - Variational Inference
authors:
    - name: Arvid Weyrauch
      affiliation: 1
    - name: Lars H. Heyen
      affiliation: 1
    - name: Peihsuan Hsia
      affiliation: 1
    - name: Asena K. Özdemir
      affiliation: 1
    - name: Markus Götz
      affiliation: 1
    - name: Charlotte Debus
      affiliation: 1
affiliations:
    - name: Karlsruhe Institute of Technology, Germany
      index: 1
      ror: 04t3en479
date: 10 June 2024
bibliography: paper.bib
---

# Summary

In the work towards reliable Neural Networks (NN), Bayesian Neural Networks (BNN) and
Variational Inference (VI) present an important approach enabling better decisions via
integrating uncertainty quantification in all steps of the training an predictions
process. They strike a balance between the ability to forecast a large variety of
distributions and compute requirements to potentially allow for larger models.

However, setting up and training BNNs is quite complicated and existing libraries all
either lack flexibility, scalability or tackle Bayesian computation in general, adding
even more complexity and therefore a huge entry barrier.

`torch_bayesian` provides an interface that is almost identical to the widely used
`pytorch` for basic use, providing low entry barrier, as well as an advanced interface
designed for exploration and research. It provides Bayesian versions of most standard
NN layer types, Bayesian optimization objectives, and a selection of relevant
distributions as well as instructions for advanced users to implement custom variants of
the same.

# Statement of need

While interest in uncertainty quantification and BNNs has been growing support for users
with little to no experience with Bayesian statistics is still limited. Bayesian
Programming Languages like Pyro and Stan are very powerful and versatile, but structured
around Bayesian concepts - like plate notation - which will be unfamiliar to many
primary machine learning users. `torch_bayesian` implements Bayes by Backprob and
approaches this by sacrificing some of the flexibility, which is usually not needed for
BNNs, to allow automatic handling. The ability to use multiple independent sampling
dimensions is sacrificed and sample vectorization is performed by the outermost instance
of the new base class `VIModule`, which captures the optional keyword argument `samples`
specifying the number of samples.
The log likelihoods typically needed for loss calculation can be returned by the layers
and for simple layer stacks handling and aggregation can be done via the `VISequential`
module allowing basic users to not be involved in this process.

For advanced users `torch_bayesian` provides a unified set of `VIkwargs`, that allows
control of the prior, variational distribution and several other advanced options.
Furthermore, it is set up to be easily extensible with custom layers and distributions.
All base classes feature post initialization checks that provide specific feedback on
missing or misspecified required attributes, methods, and sometimes signatures.

# Core design and features

`torch_bayesian` is designed around two core aims:
1. Ease of use, even for users with
little to no experience with Bayesian statistics
2. Flexibility and extensibility as required for research and exploration

| torch.nn          | Linear   | Conv[N]D   | Transformer   | Sequential   |
|-------------------|----------|------------|---------------|--------------|
| torch_bayesian.vi | VILinear | VIConv[N]D | VITransformer | VISequential |

While ease of use colors all design decisions it features most prominently in the
`pytorch`-like interface. While currently only the most common layer types provided by
`pytorch` are supported, corresponding Bayesian layers follow an analogous naming
pattern and accept the same arguments as their `pytorch` version. Additionally, while
there are minor differences the process of implementing custom layers is also very
similar to `pytorch`.

The additional arguments required to modify the Bayesian aspects of the layers are
collected on a common group of keyword arguments called `VIkwargs`. These all use
settings for mean field Gaussian variational inference with Gaussian prior as defaults
allowing beginner users to implement simple, unoptimized models without worrying about
Bayesian settings.

**include torch tutorial example here?**

While modular priors and predictive distributions are quite common even for packages
with a simpler interface flexible variational distributions are much more challenging
and are often restricted to mean field Gaussian. This is likely due to the fact that
a generic variational distribution might require any number of different parameters and
the number and shape of weight matrices can only be determined with knowledge of the
specific combination of layer and variational distribution. This is overcome in
`torch_bayesian` by having the layer provide the names and shapes of the required random
variables (e.g. mean and bias) and dynamically creating the associated class attributes
during initialization, when the variational distribution is known. The modules also
provide methods to either return samples of all random variable or the name of each
attribute for direct access.

Another challenge is introduced by the prior term of the ELBO loss. It can only be
calculated analytically for a very limited set of priors and variational distributions.
However, like the rest of the ELBO it can be estimated from the log probability of the
sampled weights under these two distributions. Therefore, `torch_bayesian` provides the
option to return these as part of the forward pass. As a result, the only requirement on
custom distributions is that there needs to be a method to differentiably sample from a
variational distribution and that for both priors and variational distributions the log
probability of a given sample can be computed.

Finally, in the age of large Neural Networks scalability and efficiency are always a
concern. While BNNs are currently scaled to very large models and this is not a primary
target of `torch_bayesian` it is kept in mind wherever possible. A core feature for this
purpose is GPU compatibility, which comes with the challenge of various backends and
device types. We address this by performing all core operations, in particular the layer
forward passes with the methods from `torch.nn.functional`. This outsources backend
maintenance to a large, community supported library.

Another, efficiency optimization is the automatic vectorization of the sampling process.
`torch_bayesian` adds an additional wrapper around the forward pass, which catches the
optional `samples` argument, creates the specified number of samples (default: 10), and
vectorizes the forward pass via `pytorch`s `vmap` method.
