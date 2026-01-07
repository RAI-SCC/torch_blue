---
title: 'torch_blue: A Flexible Python Package for Bayesian Neural Networks in PyTorch'
tags:
    - Python
    - Bayesian Neural Networks
    - Variational Inference
authors:
    - name: Arvid Weyrauch
      affiliation: 1
    - name: Lars H. Heyen
      affiliation: 1
    - name: Juan Pedro Gutiérrez Hermosillo Muriedas
      affiliation: 1
    - name: Peihsuan Hsia
      affiliation: 1
    - name: Asena K. Özdemir
      affiliation: 1
    - name: Achim Streit
      affiliation: 1
    - name: Markus Götz
      affiliation: 1, 2
    - name: Charlotte Debus
      affiliation: 1
affiliations:
    - name: Karlsruhe Institute of Technology, Germany
      index: 1
      ror: 04t3en479
    - name: Helmholtz AI
      index: 2
date: 31 October 2025
bibliography: paper.bib
---

# Summary

Bayesian Neural Networks (BNN) integrate uncertainty quantification in all steps of the
training and prediction process, thereby enabling better-informed decisions
[@arbel2023primer]. Among the different approaches to implementing BNNs, Variational
Inference (VI) [@hoffman2013svi] specifically strikes a balance between the ability to
consider a large variety of distributions while maintaining low enough compute
requirements to allow scaling to larger models.

However, setting up and training BNNs is quite complicated, and existing libraries all
either lack flexibility, lack scalability, or tackle Bayesian computation in general, adding
even more complexity and therefore a huge barrier to entry. Moreover, no existing framework
directly supports straightforward BNN model prototyping by offering pre-programmed Bayesian
network layer types, similar to PyTorch's `nn` module. This forces any BNNs to be
implemented from scratch, which can be challenging even for non-Bayesian networks.

`torch_blue` addresses this by providing an interface that is almost identical to
the widely used PyTorch [@ansel2024pytorch] for basic use, providing a low
barrier to entry, as well as an advanced interface designed for exploration and research.
Overall, this allows users to set up models and even custom layers without worrying
about the Bayesian intricacies under the hood.

# Statement of need

To represent uncertainty, BNNs do not consider their weights as point values, but
random variables, i.e., distributions. The optimization goal becomes adapting the weight
distributions to minimize their distance to the true distribution. This requires two
assumptions. For one, the distance between distributions needs to be defined, for
which the Kullback-Leibler divergence [@kullback1951information] is typically used.
Secondly, optimizing an object as complex as a distribution is a non-trivial task.
To overcome this, VI specifies a parametrized distribution and optimizes its parameters.
Thus, the Kullback-Leibler criterion can be simplified to the ELBO
(**E**vidence **L**ower **BO**und) loss [@jordan1999introduction]:
$$\mathrm{ELBO} = \mathbb{E}_{W\sim q}[\underbrace{\log p(Y|X, W)}_\mathrm{Data~fitting} - \underbrace{(\log q(W|\lambda) - \log p(W))}_\mathrm{Prior~matching}] \quad ,$$
where $(X, Y)$ are the training inputs and labels, $W$ the network weights, $q$ the
variational distribution and $\lambda$ its current best fit parameters.

While interest in uncertainty quantification and BNNs has been growing, support for
users with little to no experience in Bayesian statistics is still limited.
Probabilistic Programming Languages, such as Pyro [@bingham2019pyro] and
Stan [@stan2025stan], are very powerful and versatile, allowing the implementation of
many approaches beyond VI. However, their interfaces are structured around Bayesian
concepts - like plate notation - which will be unfamiliar to many primary machine
learning users.

![Code example of a three-layer Bayesian MLP with cross-entropy loss in
`torch_blue`. The highlight colors relate user-facing components to their position
in \autoref{design_graph}. \label{code}](code_example.png)

![Design graph of `torch_blue`. Colored highlights correspond to their practical
applications in the code example (\autoref{code}). \label{design_graph}](design_graph.png)

`torch_blue` sacrifices this extreme flexibility to allow nearly fully automatic
VI with reparametrization (Bayes by Backprop) [@blundell15bbb]. The ability to use
multiple independent sampling dimensions is removed, which allows to fully automate a
single sampling dimension in the outermost instance of the new base class `VIModule`,
which captures the optional keyword argument `samples` specifying the number of samples.
The log likelihoods typically needed for loss calculation are automatically calculated
whenever weights are sampled, aggregated, and returned once again by the outermost
`VIModule`.

# Core design and features

`torch_blue` is designed around two core aims:

1. Ease of use, even for users with little to no experience with Bayesian statistics
2. Flexibility and extensibility as required for research and exploration

While ease of use influences all design decisions, it features most prominently in the
PyTorch-like interface. While currently only the most common layer types provided by
PyTorch are supported, corresponding Bayesian layers follow an analogous naming
pattern and accept the same arguments as their PyTorch version. Additionally, while
there are minor differences, the process of implementing custom layers is also very
similar to PyTorch. To illustrate this \autoref{code} and \ref{design_graph} show an
application example and internal interactions of `torch_blue` with the colors
connecting the abstract and applied components.

The additional arguments required to modify the Bayesian aspects of the layers are
collected on a common group of keyword arguments called `VIkwargs`. The default settings use mean field Gaussian variational inference with a Gaussian prior,
allowing beginner users to implement simple, unoptimized models without worrying about
Bayesian settings.

An overview of the currently supported user-facing components is given in
\autoref{overview}. While modular priors and predictive distributions are quite common
even for packages with simpler interfaces, flexible variational distributions are much
more challenging and are often restricted to mean-field Gaussian. This is likely due to
the fact that a generic variational distribution might require any number of different
parameters, and the number and shape of weight matrices can only be determined with
knowledge of the specific combination of layer and variational distribution. This is
overcome in `torch_blue` by having the layer provide the names and shapes of the
required random variables (e.g., mean and bias) and dynamically creating the associated
class attributes during initialization, when the variational distribution is known. The
modules also provide methods to sample from the variational distribution and access its parameters.

![Overview of the major components of `torch_blue` and corresponding non-Bayesian
components of PyTorch. \label{overview}](content_overview.png "Content overview for
`torch_blue` and comparison with the interface of `torch.nn`")

Another challenge is introduced by the prior term of the ELBO loss. It can only be
calculated analytically for a very limited set of priors and variational distributions.
However, like the rest of the ELBO it can be estimated from the log probability of the
sampled weights under these two distributions. Therefore, `torch_blue` provides the
option to return these as part of the forward pass in the form of a `Tensor` containing
an additional `log_probs` attribute similar to gradient tracking. As a result, the only
requirement on custom distributions is that there needs to be a method to differentiably
sample from a variational distribution and, for both priors and variational
distributions, a method to compute the log probability of a given sample.

Finally, in the age of large Neural Networks, scalability and efficiency are always a
concern. While BNNs are not currently scaled to very large models and this is not a
primary target of `torch_blue`, it is kept in mind wherever possible. A core feature
for this purpose is GPU compatibility, which comes with the challenge of various
backends and device types. We address this by performing all core operations, in
particular the layer forward passes, with the methods from `torch.nn.functional`. This
outsources backend maintenance to a large, community-supported library.

Another efficiency optimization is the automatic vectorization of the sampling process.
`torch_blue` adds an additional wrapper around the forward pass, which catches the
optional `samples` argument, creates the specified number of samples (default: 10), and
vectorizes the forward pass via `PyTorch`s `vmap` method.

# Acknowledgements

This work is supported by the German Federal Ministry of Research, Technology and Space
under the 01IS22068 - EQUIPE grant. The authors gratefully acknowledge the computing
time made available to them through the HAICORE@KIT partition and on the
high-performance computer HoreKa at the NHR Center KIT. This center is jointly supported
by the Federal Ministry of Education and Research and the state governments
participating in the NHR ([www.nhr-verein.de/en/our-partners](www.nhr-verein.de/en/our-partners)).
