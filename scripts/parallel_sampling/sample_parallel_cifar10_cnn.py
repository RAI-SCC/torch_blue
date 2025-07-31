import torch
from torch import Tensor, nn
import torch_bayesian.vi as vi
from torch_bayesian.vi.variational_distributions import MeanFieldNormalVarDist
from torch.utils.data import DataLoader
from typing import Callable
import torch.distributed as dist
import torch.nn.functional as F
import os
import torch.multiprocessing as mp
from torchvision.transforms import ToTensor
from torchvision import datasets
import numpy as np
import random
sampling_state = None
from timing_utils import cuda_time_function, print_cuda_timing_summary
train_loss_list = []
test_loss_list = []


class CIFAR10CNN(vi.VIModule):
    def __init__(self, variational_distribution=MeanFieldNormalVarDist()):
        super().__init__()

        # Convolutional Block 1
        self.conv1 = vi.VIConv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1,
                        variational_distribution=variational_distribution)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Block 2
        self.conv2 = vi.VIConv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1,
                        variational_distribution=variational_distribution)

        # Convolutional Block 3
        self.conv3 = vi.VIConv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1,
                        variational_distribution=variational_distribution)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = vi.VILinear(128 * 4 * 4, 256, variational_distribution=variational_distribution)
        self.fc2 = vi.VILinear(256, 10, variational_distribution=variational_distribution)

    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.conv1(x)))  # (32, 16, 16)

        # Block 2
        x = self.pool(F.relu(self.conv2(x)))  # (64, 8, 8)

        # Block 3
        x = self.pool(F.relu(self.conv3(x)))  # (128, 4, 4)

        # Flatten
        x = self.flatten(x)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x



def setup(rank, world_size):
    # Initialize distributed backend
    dist.init_process_group(
        backend="nccl",  # Use NCCL for CUDA
        init_method="env://",
        rank=rank,
        world_size=world_size
    )

def cleanup():
    """
    Clean up the process group.
    """
    dist.destroy_process_group()


def get_all_rng_state(device):
    device = torch.device(device)
    if device.type == "cpu":
        return torch.get_rng_state()
    else:
        # Use device.index or default to 0
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        return torch.cuda.get_rng_state(device_index)

def set_all_rng_state(device, state):
    device = torch.device(device)
    if device.type == "cpu":
        torch.set_rng_state(state)
    else:
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.set_rng_state(state, device_index)


def train(
        dataloader: DataLoader,
        model: vi.VIModule,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        sample_num,
        train_loss_list,
        rank,
        world_size,
        device,
):
    # Communication variables
    global sampling_state  # Randomness switch
    model.train()

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        # Switch to process specific randomness
        regular_state = get_all_rng_state(device)
        if sampling_state == None:
            torch.manual_seed(rank)
            torch.cuda.manual_seed(rank)
        else:
            set_all_rng_state(device, sampling_state)

        # Get predictions
        pred = model(x, samples=sample_num)

        # Switch to general randomness
        sampling_state = get_all_rng_state(device)
        set_all_rng_state(device, regular_state)

        mean_model_output = pred.mean(dim=0)
        probs = F.softmax(mean_model_output, dim=1)
        loss = loss_fn(probs, y)
        # Backpropagation
        loss.backward()

        for param in model.parameters():
            if param.grad is not None:
                grad_global = param.grad.data
                dist.all_reduce(grad_global, op=dist.ReduceOp.SUM)

                # Average the gradients
                grad_global /= world_size

                # Copy the averaged gradients back to the parameter
                param.grad.data = torch.tensor(grad_global, dtype=param.grad.data.dtype)

        optimizer.step()
        optimizer.zero_grad()

    train_loss_list.append(loss.item())
    return model

def test(dataloader: DataLoader,
         model: vi.VIModule,
         loss_fn: Callable,
         sample_num,
         test_loss_list,
         rank,
         world_size,
         device
         ):

    num_batches = len(dataloader)
    model.eval()
    test_loss = 0.0
    correct = 0.0
    global sampling_state  # Randomness switch

    with torch.no_grad():
        for x, y in dataloader:

            x, y = x.to(device), y.to(device)

            regular_state = get_all_rng_state(device)
            if sampling_state == None:
                torch.manual_seed(rank)
                torch.cuda.manual_seed(rank)
            else:
                set_all_rng_state(device, sampling_state)

            samples = model(x, samples=sample_num)


            sampling_state = get_all_rng_state(device)
            set_all_rng_state(device, regular_state)


            samples_global = samples
            dist.all_reduce(samples_global, op=dist.ReduceOp.SUM)

            if rank == 0:
                samples_global /= world_size
                mean_model_output = torch.tensor(samples_global, dtype=samples.dtype).mean(dim=0)
                samples = F.softmax(mean_model_output, dim=1)
                correct += (samples.argmax(1) == y).type(torch.float).sum().item()
                test_loss += loss_fn(samples, y).item()

    if rank == 0:
        test_loss /= num_batches
        correct /= len(dataloader.dataset)
        test_loss_list.append(test_loss)

        print(
            f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )

    return
   

if __name__ == "__main__":
    # Hyper-parameters
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    #torch.cuda.set_device(local_rank)
    set_device = "cuda:" + str(local_rank)
    torch.device(set_device)
    
    batch_size = 64
    epochs = 5
    random_seed = 42
    all_sample_num = 32
    print(all_sample_num)
    lr = 1e-3

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    training_data = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=ToTensor())

    test_data = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=ToTensor())

    # Create data loaders.
    train_dataloader = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = CIFAR10CNN(variational_distribution=MeanFieldNormalVarDist(initial_std=1.)).to(device)
    model.return_log_probs(False)

    print(f"Using {device} device")
    print(model)
    loss_fn = F.cross_entropy
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=0)

    sample_num = int(all_sample_num / world_size)
    print(sample_num)
    
    setup(rank, world_size)
    
    # Do stuff here
    for t in range(epochs):
        if rank == 0:
            print(f"Epoch {t + 1}\n-------------------------------")
        model = train(train_dataloader, model, loss_fn, optimizer, sample_num, train_loss_list, rank, world_size,device)
        test(test_dataloader, model, loss_fn, sample_num, test_loss_list,rank, world_size, device)

    #print_cuda_timing_summary()
    cleanup()
    

