# torch_blue - A PyTorch-like library for Bayesian learning and uncertainty estimation

[![status](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![status](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![status](https://img.shields.io/pypi/v/torch-blue)](https://pypi.org/project/torch-blue/)
[![codecov](https://codecov.io/gh/RAI-SCC/torch_blue/graph/badge.svg?token=0CD3FTVKRC)](https://codecov.io/gh/RAI-SCC/torch_blue)
[![status](https://readthedocs.org/projects/torch-blue/badge/?version=latest)](https://torch-blue.readthedocs.io/en/latest/?badge=latest)
[![status](https://joss.theoj.org/papers/68b05d930d43e44aac0675c5bb3aade2/status.svg)](https://joss.theoj.org/papers/68b05d930d43e44aac0675c5bb3aade2)

----------------------------------------------------------------------------------------

`torch_blue` provides a simple way for non-expert users to implement and train Bayesian
Neural Networks (BNNs). Currently, it only supports Variational Inference (VI), but will
hopefully grow and expand in the future. To make the user experience as easy as possible
most components mirror components from [PyTorch](https://pytorch.org/docs/stable/index.html).

- [Installation](#installation)
- [Documentation](#documentation)
- [Quickstart](#quickstart)
  - [Level 1](#level-1)
  - [Level 2](#level-2)
  - [Level 3](#level-3)
  - [Level 4](#level-4)

## Installation

We heavily recommend installing `torch_blue` in a dedicated `Python3.9+`
[virtual environment](https://docs.python.org/3/library/venv.html). You can install
`torch_blue` from PyPI:

```console
$ pip install torch-blue
```

Alternatively, you can install `torch_blue` locally. To achieve this, there
are two steps you need to follow:

1. Clone the repository

```console
$ git clone https://github.com/RAI-SCC/torch_blue
```

2. Install the code locally

```console
$ pip install -e .
```

To get the development dependencies, run:

```console
$ pip install -e .[dev]
```

For additional dependencies required if you want to run scripts from the scripts
directory, run:

```console
$ pip install -e .[scripts]
```


## Documentation

Documentation is available online at [readthedocs](https://torch-blue.readthedocs.io).

## Quickstart

This Quickstart guide assumes basic familiarity with [PyTorch](https://pytorch.org/docs/stable/index.html)
and knowledge of how to implement the intended model in it. For a (potentially familiar)
example see `scripts/mnist_tutorial` (as jupyter notebook with comments, or pure python
script), which contains a copy of the PyTorch [Quickstart tutorial](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) modified to
train a BNN with variational inference.

Three levels are introduced in this guide:
- [Level 1](#level-1): Simple sequential layer stacks
- [Level 2](#level-2): Customizing Bayesian assumptions and VI kwargs
- [Level 3](#level-3): Non-sequential models and log probabilities
- [Level 4](#level-4): Custom modules with weights

### Level 1

Many parts of a neural network remain completely unchanged when turning it into a BNN.
Indeed, only `Module`s containing `nn.Parameter`s, need to be changed. Therefore, if a
PyTorch model fulfills two requirements it can be transferred almost unchanged:

1. All PyTorch `Module`s containing parameters have equivalents in this package (table below).
2. The model can be expressed purely as a sequential application of a list of layers,
i.e. with `nn.Sequential`.

| PyTorch          | vi replacement  |
|------------------|-----------------|
| `nn.Linear`      | `VILinear`      |
| `nn.Conv1d`      | `VIConv1d`      |
| `nn.Conv2d`      | `VIConv2d`      |
| `nn.Conv3d`      | `VIConv3d`      |
| `nn.Transformer` | `VITransformer` |

Given these two conditions, inherit the module from `vi.VIModule` instead of `nn.Module`
and use `vi.VISequential` instead of `nn.Sequential`. Then replace all layers
containing parameters as shown in the table above. For basic usage initialize these
modules with the same arguments as their PyTorch equivalent. For advanced usage see
[Quickstart: Level 2](#level-2). Many other layers can be included as-is. In particular
activation functions, pooling, and padding (even dropout, though they
should not be necessary since the prior acts as regularization). Currently not supported
are recurrent and transposed convolution layers. Normalization layers may
have parameters depending on their setting, but can likely be left non-Bayesian.

Additionally, the loss must be replaced. To start out use `vi.KullbackLeiblerLoss`,
which requires a `Distribution` with `self.is_predictive_distribution=True` and the size
of the training dataset (this is important for balancing of assumptions and data. Choose
your `Distribution` from the table below based on the loss you would use in PyTorch.

> [!IMPORTANT]
> `KullbackLeiblerLoss` requires the length of the dataset, not the dataloader, which is
> just the number of batches.

| PyTorch               | vi replacement <br/> from `vi.distributions` |
|-----------------------|----------------------------------------------|
| `nn.MSELoss`          | `MeanFieldNormal`                            |
| `nn.CrossEntropyLoss` | `Categorical`                                |

> [!NOTE]
> Reasons for the requirement to use `VISequential` (and how to overcome it)
> are described in [Quickstart: Level 3](#level-3). However, adding residual connections
> from the start to the end of a block of layers can also be achieved using
> `VIResidualConnection`, which acts the same as `VISequential`, but adds the input to
> the output.

### Level 2

While the interface of `VIModule`s is kept intentionally similar to PyTorch, there are
additional arguments that customize the Bayesian assumptions that all provided layers
accept and custom modules should generally accept and pass on to submodules:
- variational_distribution (`Distribution`): defines the weight distribution and
variational parameters. The default `MeanFieldNormal` assumes normal distributed,
uncorrelated weights described by a mean and a standard deviation. While there are
currently no alternatives the initial value of the standard deviation can be customized
here.
- prior (`Distribution`): defines the assumptions on the weight distribution and acts as
regularizer. The default `MeanFieldNormal` assumes normal distributed, uncorrelated
weights with mean 0 and standard deviation 1 (also known as a standard normal prior).
Mean and standard deviation can be adapted here. Particularly reducing the standard
deviation may help convergence at the risk of an overconfident model. Other available
priors:
  - `BasicQuietPrior`: an experimental prior that correlates mean and standard deviation
  to disincentivize noisy weights
- rescale_prior (`bool`): Experimental. Scales the prior similar to Kaiming-initialization.
May help with convergence, but may lead to overconfidence. Current research.
- prior_initialization (`bool`): Experimental. Initialize parameters from the prior
instead of according to standard non-Bayesian methods. May lead to much faster
convergence, but can cause the issues Kaiming-initialization counteracts unless
rescale_prior is also set to True. Current research.
- return_log_probs (`bool`): This is the topic of [Quickstart: Level 3](#level-3).

### Level 3

For more advanced models one feature of Variational Inference (VI) needs to be taken
into account. Generally, a loss for VI will require the log probability of the actually
used weights (which are sampled on each forward pass) in the variational and prior
distribution. Since it is quite inefficient to save the samples these log probabilities
are evaluated during the forward pass and returned by the model. Since this is only
necessary for training it can be controlled with the argument `return_log_probs`. Once
the model is initialized this flag can be changed by setting `VIModule.return_log_probs`,
which either enables (`True`) or disables (`False`) the returning of the log
probabilities for all submodules.

While `torch_blue` calculates and aggregates log probs internally, this is handled
by the outermost `VIModule`. This module will not have the expected output signature
when returning log probs, but instead return a `VIReturn` object. This class is PyTorch
`Tensor` that also contains log prob information in its additional `log_probs`
attribute. This is the format `torch_blue` losses expect. Therefore, if you feed the
output directly into a loss there should be no issues. While all PyTorch tensor
operations can be performed on `VIReturns` many will delete the log prob information and
transform the object back into a `Tensor`. This needs to be considered when performing
further operations on the model output. The simplest way to avoid issues is to wrap all
operations - except the loss - in a `VIModule` since log prob aggregation is only
performed by the outermost module. For deployment `return_log_probs` should be set to
`False`. If multiple `Tensor`s are returned by the model, each will carry all log probs.

> [!NOTE]
> Always make sure your outermost module is a VIModule and keep in mind that the output
> of that module will be a `VIReturn` object, which behaves like a `Tensor`, carries
> weight log probabilities, if `return_log_probs == True`. Losses in `torch_blue`
> expect this format.

> [!NOTE]
> Due to Autosampling all output Tensors, i.e. each `VIReturn`
> in the model output and the `Tensor` containing the log probs has an additional
> dimension at the beginning representing the multiple samples necessary to properly
> evaluate the stochastic forward pass. This is only relevant for VIModules that are not
> contained within other VIModules. Loss functions are designed to expect and handle
> this output format, i.e. you can simply feed the model output into the loss and
> everything will work.

### Level 4

Creating `VIModule`s with Bayesian weights - which are typically called random
variables in documentation and code - is arguably simpler than in PyTorch. Since a
different number of weight matrices needs to be created based on the variational
distribution, the process is completely automated. For `VIModules` without weights
`super().__init__` is called without arguments. Modules with random variables
expect `VIkwargs` (which you should be familiar with from [Level 2](#level-2)), but
defaults are used if non are passed. More importantly, `VIModules` with weights call
`super().__init__` with the argument `variable_shapes`. The keys of this dictionary are
the names of the random variables and the values the shapes of the weight matrices as
tuple or list. The value may also be set to `None`, which will always be the value
returned for that variable.

The insertion order of this dictionary matters, as it becomes the order of the names
in the module attribute `random_variables`. `random_variables`, the shapes, and a similar
attribute of the variational distribution call `distribution_parameters` are used to
dynamically create the weight matrices. The weight matrices can be accesses as
attributes of the module, which will cause a sample to be drawn and its log prob to be
stored if needed.

Should you need to access the weight tensors directly you can use `getattr` and derive
the name using the method `variational_parameter_name`.

> [!IMPORTANT]
> Every access of the weights will yield a new sample and log probability to be stored.
> Aggregation of multiple log probs is handled internally, but unnecessary calls will
> distort the result.
