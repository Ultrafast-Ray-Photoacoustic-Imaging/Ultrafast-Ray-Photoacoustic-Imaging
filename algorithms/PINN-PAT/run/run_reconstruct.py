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

    medium = MEDIUM.A
    medium_token = medium._value_[0]
    save_filepath = f"../ckpt/reconstruct/pinn_pat_{medium_token}_{time_str}.ckpt"
    target_filepath = r"../phantom/vessel/p0_vessel_r8_mask.pts"
    load_filepath = r"../ckpt/gaussian/pinn_pat_gs_init.ckpt"
    sensor_filepath = f"../sensor/vessel/{medium_token}_300_700_40dB.pts"
    p0_filepath = None
    # p0_filepath = r"../time_reversal/tr_vessel.pts"

    arg = Arg(
        medium=medium,
        fit_type=WAVEFIELD_FIT_TYPE.INVERSE,
        in_channels=3,
        out_channels=1,
        n_hidden=180,
        l_layers=9,
        activation=Activation.SINUSOIDAL,
    
        t_min=0.00,
        t_max=1.05,
        n_t=700,

        pde_batch_size=30000 // n_gpu,
        t0_batch_size=5000 // n_gpu,
        bc_batch_size=50,
        p0_batch_size=None,
    
        pde_weight=1e-7,
        dpdt0_weight=1e-6,
    
        lr_start=2e-4,
        lr_end=1e-5,
    
        num_iteration=700000,
    
        save_filepath=save_filepath,
        load_filepath=load_filepath,
        sensor_filepath=sensor_filepath,
        p0_filepath=p0_filepath,
    )
    print(arg.save_filepath)
    recorder = PINN_Recorder(
        num_iteration=arg.num_iteration,
        target_filepath=target_filepath, record_img=True, 
    )
    train_pinn_pat.train(arg,
        n_iter_per_save=100,
        recorder=recorder,
        print_when_save=(not multi_gpu) or local_rank == 0,
        device=device
    )