import torch
from matplotlib import pyplot as plt
from torch.optim import Adam

from torchbuq.vi import KullbackLeiblerLoss, VILinear
from torchbuq.vi.distributions import MeanFieldNormal


def test_overfitting() -> None:
    """
    Manual test of full pipline.

    Trains a single VILinear layer to return the input.
    Plots loss and calibration.
    """
    batch_size = 10
    n_batch = 1
    samples = 10
    data_shape = (10,)
    dataset_size = batch_size * n_batch

    data = torch.randn(n_batch, batch_size, *data_shape)
    model = VILinear(
        in_features=data_shape[-1],
        out_features=data_shape[-1],
        prior_initialization=True,
        return_log_probs=True,
    )
    predictive_distribution = MeanFieldNormal()
    criterion = KullbackLeiblerLoss(predictive_distribution, dataset_size=dataset_size)
    optimizer = Adam(model.parameters(), lr=1e-2)

    epochs = 1000
    all_losses = []
    for epoch in range(epochs):
        for batch in data:
            loss = criterion(model(batch, samples=samples), batch)
            all_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 100 == 99:
            print(f"Epoch {epoch+1}/{epochs}, Training loss: {all_losses[-1]}")

    plt.plot(torch.arange(len(all_losses)) / batch_size, all_losses)
    # plt.yscale('log')
    plt.show()

    test_batch = data[0]  # torch.randn(5, *data_shape)
    model.return_log_probs = False
    pred_samples = model(test_batch, samples=samples)
    prediction = predictive_distribution.predictive_parameters_from_samples(
        pred_samples
    )
    sigma_deviation = (test_batch - prediction[0]) / prediction[1]
    percentile = 0.5 + 0.5 * torch.erf(sigma_deviation / torch.sqrt(torch.tensor(2.0)))
    bins = 10
    binned = torch.histogram(percentile.cpu(), bins=bins, range=(0, 1)).hist
    binned2 = binned.cumsum(0) / binned.sum()
    plt.plot(torch.arange(bins) / bins, binned2.detach().numpy())
    plt.show()


if __name__ == "__main__":
    test_overfitting()
