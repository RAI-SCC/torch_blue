from typing import Callable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from torchbuq import vi
from torchbuq.vi import VIModule
from torchbuq.vi.distributions import Categorical


def torch_tutorial() -> None:
    """Reimplement the pytorch quickstart tutorial with BNNs."""
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    batch_size = 64

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for x, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {x.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Define model
    class NeuralNetwork(vi.VIModule):
        def __init__(self) -> None:
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = vi.VISequential(
                vi.VILinear(28 * 28, 512),
                nn.ReLU(),
                vi.VILinear(512, 512),
                nn.ReLU(),
                vi.VILinear(512, 10),
            )

        def forward(self, x_: Tensor) -> Tensor:
            x_ = self.flatten(x_)
            logits = self.linear_relu_stack(x_)
            return logits

    model = NeuralNetwork().to(device)
    model.return_log_probs = True
    print(model)

    predictive_distribution = Categorical()
    loss_fn = vi.KullbackLeiblerLoss(
        predictive_distribution, dataset_size=len(training_data)
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    def train(
        dataloader: DataLoader,
        model: VIModule,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        size = len(dataloader.dataset)
        model.train()
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            # Compute prediction error
            pred = model(x)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(dataloader: DataLoader, model: VIModule, loss_fn: Callable) -> None:
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0.0, 0.0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                samples = model(x)
                test_loss += loss_fn(samples, y).item()

                pred = predictive_distribution.predictive_parameters_from_samples(
                    samples[0]
                )
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(
            f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )

    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")


if __name__ == "__main__":
    torch_tutorial()
