def write_slurm_script() -> None:
    sample_nums = [128,256,512,1024]
    gpu_nums = [1,2,4,8,16]
    script_paths = ["/p/project1/hai_1044/oezdemir/sample_parallel/torch_bayesian/scripts/new_par_sample/DDP_Parallelism/DDP_CNN_CIFAR10.py", 
        "/p/project1/hai_1044/oezdemir/sample_parallel/torch_bayesian/scripts/new_par_sample/Hybrid-Parallelism/hybrid_CNN_CIFAR10.py",
        "/p/project1/hai_1044/oezdemir/sample_parallel/torch_bayesian/scripts/new_par_sample/Sample_Parallelism/Sample_Parallel_CNN_CIFAR10.py"
    ]
    output_file = "/p/project1/hai_1044/oezdemir/sample_parallel/torch_bayesian/experiments/outputs/"
    job_names = ["CNN_CIFAR10_DDP" , "CNN_CIFAR10_HYBRID" , "CNN_CIFAR10_SAMPLE_PARALLEL"]
    for sample_num in sample_nums:
        for gpus in gpu_nums:
            for i in range(3):
                if i == 1 and gpus < 8:
                    continue
                script_path = script_paths[i]
                job_name = job_names[i]
                filename = f"/p/project1/hai_1044/oezdemir/sample_parallel/torch_bayesian/experiments/slurm_scripts/{job_name}_{gpus}_{sample_num}.slurm"
                write_slurm_script_juwels(filename, script_path, job_name, output_file, gpus, sample_num)
    return


def write_slurm_script_juwels(filename:str, script_path: str, job_name: str, output_file: str, gpus: int, sample_num: int) -> None:
    if gpus<4:
        n_nodes = 1
        gpus_per_node = gpus
    else:
        n_nodes = int(gpus/4)
        gpus_per_node = 4

    lines = [
        "#!/bin/bash",
        "#SBATCH --partition=booster",
        f"#SBATCH --gres=gpu:{gpus_per_node}",
        "#SBATCH --time=01:00:00",
        f"#SBATCH --nodes={n_nodes}",
        f"#SBATCH --ntasks={gpus}",
        "#SBATCH --account='hai_1044'",
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --mem=16G",
        f"#SBATCH --error='{output_file}error-%j.log'",
        f"#SBATCH --output='{output_file}slurm-%j.out'",
        f"#SBATCH --job-name='{job_name}'",
        "#SBATCH --exclusive",
        "#SBATCH --export=ALL",
        "",
        "master_addr=$(scontrol show hostnames \"$SLURM_JOB_NODELIST\" | head -n 1)",
        "export MASTER_ADDR=$master_addr",
        "export MASTER_PORT=12355",
        "export GPUS_PER_NODE=$SLURM_GPUS_ON_NODE",
        "echo \"MASTER_ADDR=\"$MASTER_ADDR",
        "echo $CUDA_VISIBLE_DEVICES",
        "",
        "export PYDIR=/p/project1/hai_1044/oezdemir/sample_parallel/torch_bayesian # Set path to your python scripts."
        "",
        "module purge",
        "module load Stages/2025",
        "module load GCCcore/.13.3.0",
        "module load Python/3.12.3",


        f"srun --ntasks={gpus_per_node} bash -c '",
        "  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID",
        "  export RANK=$SLURM_PROCID",
        "  export WORLD_SIZE=$SLURM_NTASKS",
        "  echo \"Rank $SLURM_PROCID using GPU $CUDA_VISIBLE_DEVICES on node $SLURM_NODEID\"",
        "  /p/project1/hai_1044/oezdemir/sample_parallel/torch_bayesian/venv/bin/python \\",
        f"    -u {script_path} {sample_num}'",
    ]

    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return


if __name__ == "__main__":
    write_slurm_script()
