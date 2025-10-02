# Sample parallel implementation of a training loop with the entso_e dataset and a fully connected model architecture.
import torch
from torch import Tensor, nn
import torch_bayesian.vi as vi
from torch_bayesian.vi.variational_distributions import MeanFieldNormalVarDist
from torch.utils.data import DataLoader
from typing import Callable
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import os
import torch.multiprocessing as mp
import polars as pl
from entsoe_data_load import TimeseriesDatasetUnsqueeze
import time
import random
import numpy as np

HISTORY_WINDOW = 50
PREDICTION_WINDOW = 10


class TimeSeriesTransformer(vi.VIModule):
    def __init__(self, embed_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, variational_distribution):
        super(TimeSeriesTransformer, self).__init__()
        self.transformer = vi.VITransformer(
            d_model=embed_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            variational_distribution=variational_distribution
        )
        self.encoder_projection = vi.VILinear(1, embed_dim, variational_distribution=variational_distribution)
        self.decoder_projection = vi.VILinear(1, embed_dim, variational_distribution=variational_distribution)
        self.fc_out = vi.VILinear(embed_dim, 1, variational_distribution=variational_distribution)

    def forward(self, src, tgt):
        src_emb = self.encoder_projection(src)
        tgt_emb = self.decoder_projection(tgt)
        src_emb = src_emb.transpose(0, 1)  # (seq, batch, embed)
        tgt_emb = tgt_emb.transpose(0, 1)
        output = self.transformer(src_emb, tgt_emb)  
        output = output.transpose(0, 1)
        return self.fc_out(output)

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


def train(dataloader, model, loss_fn, optimizer, sample_num, epochs, g, dataloader_seed, rank, device):
    for t in range(epochs):
        if rank==0:
            if t !=0:
                now = time.time()
                print("Elapsed Time for Epoch:")
                print(now-start)
            start = time.time()

        g.manual_seed(dataloader_seed + t)
        
        if rank == 0:
            print(f"Epoch {t + 1}\n-------------------------------")
 
        model.train()

        i = 0
        for batch, (x, y) in enumerate(dataloader):
            if rank==0:
                print(i)
                i +=1
            x, y = x.to(device), y.to(device)

            # prepare teacher-forced input/output
            future_input = y[:, :-1, :]
            future_output = y[:, 1:, :]
            output = model(x, future_input, samples=sample_num)
            mean_model_output = output.mean(dim=0)
            loss = loss_fn(mean_model_output, future_output)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return model


def test(dataloader, model, loss_fn, sample_num, rank, device):
    if rank == 0:
        num_batches = len(dataloader)
        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)

                # start decoder with zeros
                tgt_input = torch.zeros((y.size(0), PREDICTION_WINDOW, 1), device=device)
                output = model(x, tgt_input, samples=sample_num)
                mean_model_output = output.mean(dim=0)
                test_loss += loss_fn(mean_model_output, y).item()

        test_loss /= num_batches
        print(f"Test Error: Avg loss: {test_loss:.6f}\n")


if __name__ == "__main__":
    # Distributed Stuff
    
    input_length = HISTORY_WINDOW
    output_length = PREDICTION_WINDOW

    rank = int(os.environ["SLURM_PROCID"])       # global rank
    world_size = int(os.environ["SLURM_NTASKS"])
    gpus_per_node = int(os.environ.get("GPUS_PER_NODE"))
    node_rank = rank // gpus_per_node
    num_nodes = world_size // gpus_per_node

    device = torch.device("cuda")

    setup(rank, world_size)
    
    dataloader_seed = 1998
    seed = 42

    seed_all(seed, 0)

    epochs = 10
    batch_size = 32
    all_sample_num = 128
    EMBED_DIM = 128
    NHEAD = 4
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    DIM_FEEDFORWARD = 256
    
    df = pl.read_csv("/p/scratch/hai_1044/oezdemir1/data/de.csv",
                     dtypes={"start": pl.Datetime, "end": pl.Datetime, "load": pl.Float32},
                     )
    x = df["load"]
    x = x.fill_null(strategy="backward")
    normalized_x = (x - x.mean()) / x.std()
    x_tensor = normalized_x.to_torch()
    data_train, data_test = x_tensor[: int(len(x) * 0.7)], x_tensor[int(len(x) * 0.7):]
    dataset_train = TimeseriesDatasetUnsqueeze(data_train, input_length, output_length)
    dataset_test = TimeseriesDatasetUnsqueeze(data_test, input_length, output_length)

    # Create data loaders.
    g = torch.Generator()
    g.manual_seed(dataloader_seed)
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, generator=g)
    test_dataloader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True)

    model = TimeSeriesTransformer(EMBED_DIM, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD,
                          variational_distribution=MeanFieldNormalVarDist(initial_std=1.)).to(device)
    model.return_log_probs(False)
    model = DDP(model)

    loss_fn = vi.MeanSquaredErrorLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=0)

    sample_num = int(all_sample_num / world_size)
    

    model = train(train_dataloader, model, loss_fn, optimizer, sample_num, epochs, g, dataloader_seed,rank,device)
    test(test_dataloader, model, loss_fn, sample_num, rank,device)

    cleanup()

