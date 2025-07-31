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
train_loss_list = []
test_loss_list = []

class BasicBlock(vi.VIModule):
    def __init__(self, inchannels, outchannels, stride=1, variational_distribution=MeanFieldNormalVarDist()):
        super(BasicBlock, self).__init__()
        self.conv1 = vi.VIConv2d(inchannels, outchannels, kernel_size=3, stride=stride, padding=1, bias=False, variational_distribution=variational_distribution)
        self.bn1 = nn.BatchNorm2d(outchannels, track_running_stats=False)
        self.conv2 = vi.VIConv2d(outchannels, outchannels, 3, 1, 1, bias=False, variational_distribution=variational_distribution)
        self.bn2 = nn.BatchNorm2d(outchannels, track_running_stats=False)

        self.shortcut = vi.VISequential()
        if stride != 1 or inchannels != outchannels:
            self.shortcut = vi.VISequential(
                vi.VIConv2d(inchannels, outchannels, 1, stride, bias=False, variational_distribution=MeanFieldNormalVarDist()),
                nn.BatchNorm2d(outchannels, track_running_stats=False),
            )

    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet10(vi.VIModule):
    def __init__(self, num_classes=10, variational_distribution=MeanFieldNormalVarDist()):
        super().__init__()
        self.inchannels = 64

        self.conv = vi.VIConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False, variational_distribution=variational_distribution)
        self.bn = nn.BatchNorm2d(64, track_running_stats=False)

        # ResNet-10: one block per layer group
        self.layer1 = self._make_layer(64, 1, stride=1)
        self.layer2 = self._make_layer(128, 1, stride=2)
        self.layer3 = self._make_layer(256, 1, stride=2)
        self.layer4 = self._make_layer(512, 1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = vi.VILinear(512, num_classes, variational_distribution=variational_distribution)

    def _make_layer(self, outchannels, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.inchannels, outchannels, stride=s))
            self.inchannels = outchannels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

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
                mean_model_output = samples_global.mean(dim=0)
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
    model = ResNet10(variational_distribution=MeanFieldNormalVarDist(initial_std=1.)).to(device)
    model.return_log_probs(False)

    print(f"Using {device} device")
    print(model)
    #pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    loss_fn = F.cross_entropy
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=0)

    sample_num = int(all_sample_num / world_size)
    print(sample_num)
    
    setup(rank, world_size)

    for t in range(epochs):
        if rank == 0:
            print(f"Epoch {t + 1}\n-------------------------------")
        model = train(train_dataloader, model, loss_fn, optimizer, sample_num, train_loss_list, rank, world_size,device)
        test(test_dataloader, model, loss_fn, sample_num, test_loss_list,rank, world_size, device)

    cleanup()
    

