import torch
import os
import sys
from datetime import datetime
import time
sys.path.append("..")

from train import train_pinn_pat
from models.activation import Activation
from train.pinn_recorder import PINN_Recorder
from models.arg import Arg, MEDIUM, WAVEFIELD_FIT_TYPE


def create_gaussian_p0(filepath, size):
    X = torch.linspace(0, 1, 2 * size + 1)[1::2].float()
    x, y = torch.meshgrid([X] * 2, indexing="ij")
    xy = torch.stack([x, y], dim=-1).flatten(0, 1)
    print(xy.shape)
    p0 = (-((xy[:, 0] - 0.5).square() + (xy[:, 1] - 0.5).square()) * 30).exp()
    torch.save(dict(
        xy=xy,
        p0=p0,
    ), filepath)
    
if __name__ == "__main__":
    time_str = datetime.now().strftime("%m%d_%H%M")

    multi_gpu = "LOCAL_RANK" in os.environ
    if multi_gpu:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = torch.distributed.get_world_size()
        print("\033[91mlocal_rank:", local_rank, " | device:", device, "/", n_gpu, "\033[0m")
        seed = int(time.time())
        torch.manual_seed(seed + local_rank)
        torch.cuda.manual_seed(seed + local_rank)
    else:
        CUDA_ID = 0
        n_gpu = 1
        device = torch.device(f"cuda:{CUDA_ID}")
        print("\033[91mdevice:", device, "\033[0m")
        
    save_filepath = os.path.join("../ckpt/gaussian", f"pinn_pat_gs_{time_str}.ckpt")
    p0_filepath = os.path.join("../phantom/gaussian/gaussian_p0_100.pts")
    if not os.path.exists(p0_filepath):
        create_gaussian_p0(p0_filepath, 100)
    
    arg = Arg(
        medium=MEDIUM.A,
        fit_type=WAVEFIELD_FIT_TYPE.FORWARD,
        in_channels=3,
        out_channels=1,
        n_hidden=180,
        l_layers=9,
        activation=Activation.SINUSOIDAL,
    
        t_min=0.00,
        t_max=1.05,
        n_t=700,
    
        pde_batch_size=10000,
        t0_batch_size=5000,
        bc_batch_size=None,
        pde_weight=1e-7,
        dpdt0_weight=1e-6,
    
        lr_start=2e-4,
        lr_end=1e-5,
        num_iteration=50000,
    
        save_filepath=save_filepath,
        load_filepath=None,
        sensor_filepath=None,
        p0_filepath=p0_filepath
    )
    print(arg.save_filepath)
    recorder = PINN_Recorder(num_iteration=arg.num_iteration, record_psnr=False)
    train_pinn_pat.train(arg,
         recorder=recorder,
         print_when_save=(not multi_gpu) or local_rank == 0,
         device=device
    )