import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms, models
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
import time
import csv


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


def seed_all(seed, rank):
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def DDP_train(train_dataloader, test_dataloader, g, dataloader_seed, model, loss_fn, optimizer, epochs, sample_num, rank, world_size, device):
    times = []
    for t in range(epochs):
        if rank == 0:
            
            start = time.time()
            print(f"Epoch {t + 1}\n-------------------------------")
 
        g.manual_seed(dataloader_seed + t)
        model.train()

        for batch, (x, y) in enumerate(train_dataloader):
            # Load batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()


            # Get predictions
            pred = model(x, samples=sample_num)

            # Calculate loss
            mean_model_output = pred.mean(dim=0)
            loss = loss_fn(mean_model_output, y)
            # print(loss.item())

            # Backpropagation
            loss.backward()

            # Update Model
            optimizer.step()
        
        if rank==0:
            now = time.time()
            print("Elapsed Time for Epoch:")
            print(now-start)
            times.append(now-start)
            
    if rank==0:
        print("Times Averaged:")
        print(sum(times[2:])/(len(times)-2))

    test_loss = DDP_test(test_dataloader, model, loss_fn, sample_num,rank, world_size, device)


    return model, sum(times[2:])/(len(times)-2), test_loss


def DDP_test(dataloader: DataLoader,
         model: vi.VIModule,
         loss_fn: Callable,
         sample_num,
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

            samples = model(x, samples=sample_num)

            mean_model_output = samples.mean(dim=0)
            correct += (mean_model_output.argmax(1) == y).type(torch.float).sum().item()
            test_loss += loss_fn(mean_model_output, y).item()

    if rank == 0:
        test_loss /= num_batches
        correct /= len(dataloader.dataset)

        print(
            f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )

    return test_loss
   

def DDP_pipeline(seed, training_data, test_data, model, epochs, batch_size, global_sample_num, optimizer, loss_fn):
    seed_all(seed, 0)

    # Distributed Stuff
    rank = int(os.environ["SLURM_PROCID"])       # global rank
    world_size = int(os.environ["SLURM_NTASKS"])
    gpus_per_node = int(os.environ.get("GPUS_PER_NODE"))
    node_rank = rank // gpus_per_node
    num_nodes = world_size // gpus_per_node

    print(f"Number of GPUs: {world_size}")

    device = torch.device("cuda")

    setup(rank, world_size)
    
    dataloader_seed = 1998

    print(f"Using {device} device")

    model = model.to(device)

    ddp_model = DDP(model)

    seed_all(seed, rank)

    sample_num = int(global_sample_num / world_size)

    # Create data loaders.
    g = torch.Generator()
    g.manual_seed(dataloader_seed)

    train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True, generator=g)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True) 

    mddp_model, time_taken, test_loss = DDP_train(train_dataloader, test_dataloader, g, dataloader_seed, ddp_model, loss_fn, optimizer, epochs, sample_num, rank, world_size, device)

    if rank==0:
        summary = ['CIFAR10', 'CNN', 'SAMPLE-PARALLEL', global_sample_num, world_size, time_taken, test_loss]
        with open('/p/project1/hai_1044/oezdemir/sample_parallel/torch_bayesian/experiments/outputs/scaling.csv', 'a', newline='') as fd:
            writer = csv.writer(fd)
            writer.writerow(summary)


    cleanup()

    


