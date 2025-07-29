# Sample parallel implementation of a training loop with the entso_e dataset and a fully connected model architecture.
import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
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
sampling_state = None
train_loss_list = []
test_loss_list = []
class MNISTCNN(vi.VIModule):
    def __init__(self, variational_distribution=MeanFieldNormalVarDist()):
        super().__init__()

        # Convolutional Block 1
        self.conv1 = vi.VIConv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1,
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
        self.fc1 = vi.VILinear(128 * 7 * 7, 256, variational_distribution=variational_distribution)
        self.fc2 = vi.VILinear(256, 10, variational_distribution=variational_distribution)

    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.conv1(x)))  # (32, 14, 14)

        # Block 2
        x = self.pool(F.relu(self.conv2(x)))  # (64, 7, 7)

        # Block 3
        x = F.relu(self.conv3(x))  # (128, 7, 7)

        # Flatten
        x = self.flatten(x)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
'''

class NeuralNetwork(vi.VIModule):
    def __init__(self, in_channel, hidden1, hidden2, output_length,
                 variational_distribution=MeanFieldNormalVarDist()) -> None:
        super().__init__()
        self.conv_stack = vi.VISequential(
            vi.VIConv2d(in_channels=in_channel, out_channels=hidden1, kernel_size=3, stride=1, padding=1,
                        variational_distribution=variational_distribution),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            vi.VIConv2d(in_channels=hidden1, out_channels=hidden2, kernel_size=3, stride=1, padding=1,
                        variational_distribution=variational_distribution),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            vi.VILinear(hidden2*7*7, output_length, variational_distribution=variational_distribution),
        )

    def forward(self, x_: Tensor) -> Tensor:
        logits = self.conv_stack(x_)
        return logits
'''
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
        regular_state = torch.get_rng_state()
        if sampling_state == None:
            torch.manual_seed(rank)
        else:
            torch.set_rng_state(sampling_state)

        # Get predictions
        pred = model(x, samples=sample_num)
        # Switch to general randomness
        sampling_state = torch.get_rng_state()
        torch.set_rng_state(regular_state)

        mean_model_output = pred.mean(dim=0)
        probs = F.softmax(mean_model_output, dim=1)
        loss = loss_fn(probs, y)
        # Backpropagation
        loss.backward()

        #for param in model.parameters():
        #    if param.grad is not None:
        #        grad_global = param.grad.data
        #        dist.all_reduce(grad_global, op=dist.ReduceOp.SUM)
#
        #        # Average the gradients
        #        grad_global /= world_size
#
        #        # Copy the averaged gradients back to the parameter
        #        param.grad.data = torch.tensor(grad_global, dtype=param.grad.data.dtype)

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

            regular_state = torch.get_rng_state()
            if sampling_state == None:
                torch.manual_seed(rank)
            else:
                torch.set_rng_state(sampling_state)

            samples = model(x, samples=sample_num)


            sampling_state = torch.get_rng_state()
            torch.set_rng_state(regular_state)


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
    torch.cuda.set_device(local_rank)
    
    batch_size = 64
    epochs = 5
    random_seed = 42
    all_sample_num = 32
    print(all_sample_num)
    lr = 1e-3
    #mp.set_start_method("fork", force=True)
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    # Create data loaders.
    # set each process to have same random seed 
    torch.manual_seed(random_seed) # Ensure random seed is set before Dataloader is initialized
    train_dataloader = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    # Get cpu, gpu or mps device for training.
    setup(rank, world_size)
    device = (
        f"cuda:{local_rank}"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = MNISTCNN(variational_distribution=MeanFieldNormalVarDist(initial_std=1.)).to(device)
    model.return_log_probs(False)

    print(model)
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank
    )
    loss_fn = F.cross_entropy
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=0)

    sample_num = int(all_sample_num / world_size)
    print(sample_num)
    
    
    # Do stuff here
    for t in range(epochs):
        if rank == 0:
            print(f"Epoch {t + 1}\n-------------------------------")
        model = train(train_dataloader, model, loss_fn, optimizer, sample_num, train_loss_list, rank, world_size,device)
        test(test_dataloader, model, loss_fn, sample_num, test_loss_list,rank, world_size, device)

    cleanup()
    

