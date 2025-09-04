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
    - name: Juan Pedro Gutiérrez Hermosillo Muriedas
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
date: 08 September 2025
bibliography: paper.bib
---

# Summary

Bayesian Neural Networks (BNN) integrate uncertainty quantification in all steps of the
training and prediction process, thus enabling better-informed decisions
[@arbel2023primer]. Variational Inference (VI) [@hoffmann2013svi] specifically strikes a
balance between the ability to consider a large variety of distributions while
maintaining low enough compute requirements to potentially allow scaling to larger
models.

However, setting up and training BNNs is quite complicated and existing libraries all
either lack flexibility, scalability, or tackle Bayesian computation in general, adding
even more complexity and therefore a huge entry barrier. The most popular options - Pyro
[@bingham2019pyro] and Stan [@stan2025stan] - fall in the last category. While both are
quite powerful, their interfaces are designed for users experienced with Bayesian
statistics. Even more importantly, neither of these directly supports BNNs by providing
pre-programmed layers. This forces any BNNs to be implemented from scratch, which can be
challenging even for non-Bayesian networks.

`torch_bayesian` addresses this by providing an interface that is almost identical to
the widely used `pytorch` [@ansel2024pytorch] for basic use, providing a low entry
barrier, as well as an advanced interface designed for exploration and research.
Overall, this allows users to set up models and even custom layers without worrying
about the Bayesian intricacies under the hood.

# Statement of need

To represent uncertainty BNNs do not consider their weights as point values, but
random variables, i.e. distributions. The optimization goal becomes adapting the weight
distributions to minimize their distance to the perfect distribution. This requires two
assumptions. Firstly, the Kullback-Leibler divergence [@kullback1951information] is
typically used to define the distance between distributions. Secondly, it is non-trivial
to optimize an object as complex as a distributions. VI answers this by selecting a
parametrized distribution and optimizing its parameters. Thus, the Kullback-Leibler
criterion can be simplified to the ELBO (**E**vidence **L**ower **BO**und) loss
[@jordan1999introduction]:
$$\mathrm{ELBO} = \mathbb{E}_{W\sim q}[\underbrace{\log p(Y|X, W)}_\mathrm{Data~fitting} \underbrace{(\log q(W|\lambda) - \log p(W))}_\mathrm{Prior~matching}] \quad ,$$
where $(X, Y)$ are the training inputs and labels, $W$ are the network weights, $q$ the
variational distribution and $\lambda$ its current best fit parameters.

While interest in uncertainty quantification and BNNs has been growing, support for
users with little to no experience with Bayesian statistics is still limited.
Probabilistic Programming Languages like Pyro and Stan are very powerful and versatile
allowing the implementation of many approaches beyond VI. But they are structured
around Bayesian concepts - like plate notation - which will be unfamiliar to many
primary machine learning users.

`torch_bayesian` sacrifices this extreme flexibility to allow nearly fully automating
VI with reparametrization (Bayes by Backprop) [@blundell15bbb]. The ability to use
multiple independent sampling dimensions is sacrificed to fully automate it in the
outermost instance of the new base class `VIModule`, which captures the optional keyword
argument `samples` specifying the number of samples. The log likelihoods typically
needed for loss calculation are automatically calculated whenever weights are sampled,
aggregated, and returned once again by the outermost `VIModule`.

# Core design and features

`torch_bayesian` is designed around two core aims:
1. Ease of use, even for users with little to no experience with Bayesian statistics
2. Flexibility and extensibility as required for research and exploration

![Content overview](content_overview.png "Content overview for `torch_bayesian` and comparison with the interface of `torch.nn`")

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

![Code example](code_example.png "Simple usage example for `torch_bayesian`. Colours related to the design graph below.")
![Design graph](design_graph.png "Interaction graph of core components. Colors relate to the code example above.")

While modular priors and predictive distributions are quite common even for packages
with a simpler interface, flexible variational distributions are much more challenging
and are often restricted to mean-field Gaussian. This is likely due to the fact that
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
option to return these as part of the forward pass in the form of a `Tensor` containing
an additional `log_probs` attribute similar to gradient tracking. As a result, the only
requirement on custom distributions is that there needs to be a method to differentiably
sample from a variational distribution and that for both priors and variational
distributions the log probability of a given sample can be computed.

Finally, in the age of large Neural Networks, scalability and efficiency are always a
concern. While BNNs are not currently scaled to very large models and this is not a
primary target of `torch_bayesian` it is kept in mind wherever possible. A core feature
for this purpose is GPU compatibility, which comes with the challenge of various
backends and device types. We address this by performing all core operations, in
particular the layer forward passes with the methods from `torch.nn.functional`. This
outsources backend maintenance to a large, community supported library.

Another, efficiency optimization is the automatic vectorization of the sampling process.
`torch_bayesian` adds an additional wrapper around the forward pass, which catches the
optional `samples` argument, creates the specified number of samples (default: 10), and
vectorizes the forward pass via `pytorch`s `vmap` method.
