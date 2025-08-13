# torch_bayesian - Easy Variational Inference

This package provides a simple way for non-expert users to implement and train Bayesian
Neural Networks (BNNs) with Variational Inference (VI). To make this as easy as possible
most components mirror components from [pytorch](https://pytorch.org/docs/stable/index.html).

- [Installation](#installation)
- [Documentation](#documentation)
- [Quickstart](#quickstart)
  - [Level 1](#level-1)
  - [Level 2](#level-2)
  - [Level 3](#level-3)
  - [Level 4](#level-4)

## Installation

We heavily recommend installing ``torch_bayesian`` in a dedicated `Python3.8+`
[virtual environment](https://docs.python.org/3/library/venv.html). You can install
``torch_bayesian`` directly from the GitHub repository via:

```console
$ pip install git+https://github.com/RAI-SCC/torch_bayesian
```

Alternatively, you can install ``torch_bayesian`` locally. To achieve this, there are two steps you
need to follow:

1. Clone the repository

```console
$ git clone https://github.com/RAI-SCC/torch_bayesian
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

Documentation is not yet hosted online, but can easily be build locally with sphinx.
Run

```console
$ pip install -e .[docs]
```

Then navigate to ``torch_bayesian/docs/`` and run:

```console
$ sphinx-build -T -E -b html ./source ./build
```

You can then open ``torch_bayesian/docs/build/index.html`` to start browsing.


## Quickstart

This Quickstart guide assumes basic familiarity with [pytorch](https://pytorch.org/docs/stable/index.html)
and knowledge of how to implement the intended model in it. For a (potentially familiar)
example see `scripts/pytorch_tutorial.py`, which contains a copy of the pytorch
[Quickstart tutorial](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
modified to train a BNN with variational inference.
Three levels are introduced:
- [Level 1](#level-1): Simple sequential layer stacks
- [Level 2](#level-2): Customizing Bayesian assumptions and VI kwargs
- [Level 3](#level-3): Non-sequential models and log probabilities
- [Level 4](#level-4): Custom modules with weights

### Level 1

Many parts of a neural network remain completely unchanged when turning it into a BNN.
Indeed, only `Module`s containing `nn.Parameter`s, need to be changed. Therefore, if a
pytorch model fulfills two requirements it can be transferred almost unchanged:

1. All pytorch `Module`s containing parameters have equivalents in this package (table below).
2. The model can be expressed purely as a sequential application of a list of layers,
i.e. with `nn.Sequential`.

| pytorch          | vi replacement  |
|------------------|-----------------|
| `nn.Linear`      | `VILinear`      |
| `nn.Conv1d`      | `VIConv1d`      |
| `nn.Conv2d`      | `VIConv2d`      |
| `nn.Conv3d`      | `VIConv3d`      |
| `nn.Transformer` | `VITransformer` |

Given these two conditions, inherit the module from `vi.VIModule` instead of `nn.Module`
and use `vi.VISequential` instead of `nn.Sequential`. Then replace all layers
containing parameters as shown in the table above. For basic usage initialize these
modules with the same arguments as their pytorch equivalent. For advanced usage see
[Quickstart: Level 2](#level-2). Many other layers can be included as-is. In particular
activation functions, pooling, and padding (even dropout, though they
should not be necessary since the prior acts as regularization). Currently not supported
are recurrent and transposed convolution layers. Normalization layers may
have parameters depending on their setting, but can likely be left non-Bayesian.

Additionally, the loss must be replaced. To start out use `vi.KullbackLeiblerLoss`,
which requires a `PredictiveDistribution` and the size of the training dataset (this is
important for balancing of assumptions and data, more details
[here](#variational-inference)). Choose your `PredictiveDistribution`
from the table below based on the loss you would use in pytorch (more details
[here](#the-predictive-distribution)).

> [!IMPORTANT]
> `KullbackLeiblerLoss` requires the length of the dataset, not the dataloader, which is
> just the number of batches.

| pytorch               | vi replacement (import from `vi.predictive_distributions`) |
|-----------------------|------------------------------------------------------------|
| `nn.MSELoss`          | `MeanFieldNormalPredictiveDistribution`                    |
| `nn.CrossEntropyLoss` | `CategoricalPredictiveDistribution`                        |

> [!NOTE]
> Reasons for the requirement to use `VISequential` (and how to overcome it)
> are described in [Quickstart: Level 3](#level-3). However, adding residual connections
> from the start to the end of a block of layers can also be achieved using
> `VIResidualConnection`, which acts the same as `VISequential`, but adds the input to
> the output.

### Level 2

While the interface of `VIModule`s is kept intentionally similar to pytorch, there are
additional arguments that customize the Bayesian assumptions that all provided layers
accept and custom modules should generally accept and pass on to submodules:
- variational_distribution (`VariationalDistribution`): defines the weight distribution
and variational parameters (more details [here](#the-variational-distribution)). The
default `MeanFieldNormalVarDist` assumes normal distributed, uncorrelated weights
described by a mean and a standard deviation. While there are currently no alternatives
the initial value of the standard deviation can be customized here.
- prior (`Prior`): defines the assumptions on the weight distribution and acts as
regularizer (more details [here](#the-prior)). The default `MeanFieldNormalPrior`
assumes normal distributed, uncorrelated weights with mean 0 and standard deviation 1
  (also known as an uninformative or standard normal prior). Mean and standard deviation
can be adapted here. Particularly reducing the standard deviation may help convergence
at the risk of an overconfident model. Other available priors:
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

For more advanced models one feature of [Variational Inference](#variational-inference)
(VI) needs to be taken into account. Generally, a loss for VI will require the log
probability of the actually used weights (which are sampled on each forward pass) in the
variational and prior distribution. Since it is quite inefficient to save the samples
these log probabilities are evaluated during the forward pass and returned by the model.
Since this is only necessary for training it can be controlled with the argument
return_log_probs. Once the model is initialized this flag can be changed by setting
`VIModule.return_log_probs`, which either enables (`True`) or disables (`False`) the
returning of the log probabilities for all submodules.

While `torch_bayesian` calculates and aggregates log probs internally, this is handled
by the outermost `VIModule`. This module will not have the expected output signature
when returning log probs, but instead return a tuple of the normal output and the log
probs. This is the format `torch_bayesian` losses expect. Therefore, if you feed the
output directly into a loss there should be no issues. For deployment `return_log_probs`
should be set to `False`. However, if your outermost module is not a `VIModule`, but a
pytorch `Module`, or you are working with multiple models like an encoder and decoder
this feature needs to be handled. The easiest way to do this is to make sure that your
outermost module is always a `VIModule` even if it is only a wrapper that calls you
model and returns its output, since this will make sure log probs are only introduced at
the very end.

> [!NOTE]
> Always make sure your outermost module is a VIModule and keep in mind that the output
> of that module will be a tuple of the expected output and a tensor containing the
> weight log probabilities, if `return_log_probs == True`. Losses in `torch_baysian`
> expect this format.

`VIReturn` is a type alias that encapsulates this shifting return type. Just provide the
type of the layer output to it.

> [!NOTE]
> Due to [Autosampling](#autosampling) all output Tensors, i.e. each Tensor
> in the model output and the Tensor containing the log probs has an additional
> dimension at the beginning representing the multiple samples necessary to properly
> evaluate the stochastic forward pass. This is only relevant for VIModules that are not
> contained within other VIModules. Loss functions are designed to expect and handle
> this output format, i.e. you can simply feed the model output into the loss and
> everything will work.

### Level 4

Arguably, creating `VIModule`s with Bayesian weights - which are typically called random
variables in documentation and code - is arguably simpler than in pytorch. Since a
different number of weight matrices needs to be created based on the variational
distribution, the process is completely automated. For `VIModules` without weights
`super().__init__` is called without arguments. Modules with random variables
expect `VIkwargs` (which you should be familiar with from [Level 2](#level-2)), but
defaults are used if non are passed. More importantly, `VIModules` with weights call
`super().__init__` with the argument `variable_shapes`. The keys of this dictionary are
the names of the random variables and the values the shapes of the weight matrices as
tuple or list.

The insertion order of this dictionary matters, as it becomes the order of the names
in the module attribute `random_variables`. `random_variables`, the shapes, and a similar
attribute of the variational distribution call `variational_parameters` are used to
dynamically create the weight matrices. To get weight during the forward pass call the
`sample_variables` method, which returns a tuple of sampled weight matrices. One for
each entry in `random_variables` and in the same order. This call also internally
calculates and stores the log probabilities, if required.

Should you need to access the weight tensors directly you can use `getattr` and derive
the name using the method `variational_parameter_name`.

## Variational Inference

### The Prior

### The Variational Distribution

### The Predictive Distribution

### Autosampling
