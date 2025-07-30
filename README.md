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
are recurrent, Transformer and transposed convolution layers. Normalization layers may
have parameters depending on their setting, but can likely be left non-Bayesian.

Additionally, the loss must be replaced. To start out use `vi.KullbackLeiblerLoss`,
which requires a `PredictiveDistribution` and the size of the training dataset (this is
important for balancing of assumptions and data, more details
[here](#variational-inference)). Choose your `PredictiveDistribution`
from the table below based on the loss you would use in pytorch (more details
[here](#the-predictive-distribution)).

| pytorch               | vi replacement (import from `vi.predictive_distributions`) |
|-----------------------|------------------------------------------------------------|
| `nn.MSELoss`          | `MeanFieldNormalPredictiveDistribution`                    |
| `nn.CrossEntropyLoss` | `CategoricalPredictiveDistribution`                        |

> **Note:** Reasons for the requirement to use `VISequential` (and how to overcome it)
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
  - `BasicQuietPrior`: a prior that correlates mean and standard deviation to
  disincentivise noisy weights
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

When creating advance `VIModule`s you will need to consider, that provided modules
return a tuple during training. The first element of this tuple is the usual model
output. The second element is a Tensor containing two values: prior_log_prob and
variational_log_prob. Your modules must be able to handle both cases (by checking
`_return_log_probs`) and return log probs accordingly. If you have multiple submodels
returning log probs you can just add them. Generally, this will follow the pattern:

```python
from torch import Tensor

from torch_bayesian.vi import VIModule
from torch_bayesian.vi.utils.common_types import VIReturn


class VINetwork(VIModule):
  def __init__(self) -> None:
    super().__init__()
    self.module1 = ... # some VIModule
    self.module2 = ... # another VIModule

  def foward(self, input_: Tensor) -> VIReturn[Tensor]:
      if self._return_log_probs:
          input_, log_probs1 = self.module1(input_)
          output, log_probs2 = self.module2(input_)
          log_probs = log_probs1 + log_probs2
          return output, log_probs
      else:
          input_ = self.module1(input_)
          output = self.module2(input_)
          return output
```

`VIReturn` is a type alias that encapsulates this shifting return type. Just provide the
type of the layer output to it.

Creating custom `VIModules` with parameters goes beyond the scope of this guide.

> **Note:** Due to [Autosampling](#autosampling) all output Tensors, i.e. each Tensor
> in the model output and the Tensor containing the log probs has an additional
> dimension at the beginning representing the multiple samples necessary to properly
> evaluate the stochastic forward pass. This is only relevant for VIModules that are not
> contained within other VIModules. Loss functions are designed to expect and handle
> this output format, i.e. you can simply feed the model output into the loss and
> everything will work.

## Variational Inference

### The Prior

### The Variational Distribution

### The Predictive Distribution

### Autosampling
