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
