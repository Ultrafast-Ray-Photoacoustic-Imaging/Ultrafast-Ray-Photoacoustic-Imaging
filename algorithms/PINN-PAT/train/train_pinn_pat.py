from datetime import datetime
import torch
import os
import sys
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
sys.path.append("..")
from models.pinn_for_pat import PINNForPAT, LOSS
from models.arg import Arg, WAVEFIELD_FIT_TYPE
from train.sensor_manager import SensorManager
from train.p0_manager import P0Manager
from train.pinn_recorder import PINN_Recorder

torch.set_printoptions(sci_mode=True, precision=5)

def train(
    arg: Arg,
    n_iter_per_save: int=1000,
    recorder: PINN_Recorder=None,
    print_when_save: bool=False,
    device: torch.device=None
):
    pinn_pat = PINNForPAT(arg=arg, device=device).to(device)
    recorder.attach_model(pinn_pat, device)

    if "LOCAL_RANK" in os.environ:
        pinn_pat = torch.nn.parallel.DistributedDataParallel(pinn_pat, device_ids=[int(os.environ["LOCAL_RANK"])])
        pinn_pat_module = pinn_pat.module
    else:
        pinn_pat_module = pinn_pat
    
    p0_exist = arg.fit_type == WAVEFIELD_FIT_TYPE.FORWARD or arg.p0_filepath is not None
    if p0_exist:
        assert arg.p0_filepath is not None
        p0_type = P0Manager.P0_TYPE.FORWARD_PROPAGATE if arg.fit_type == WAVEFIELD_FIT_TYPE.FORWARD else P0Manager.P0_TYPE.TR_ASSIST
        p0_manager = P0Manager(arg.p0_filepath, p0_type, arg.p0_batch_size, device)

    bc_exist = arg.fit_type == WAVEFIELD_FIT_TYPE.INVERSE
    if bc_exist:
        assert arg.sensor_filepath is not None
        sensor_manager = SensorManager(arg.sensor_filepath, arg.bc_batch_size, device)

    optimizer = torch.optim.Adam(params=pinn_pat.parameters(), lr=arg.lr_start)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=arg.num_iteration, T_mult=1, eta_min=arg.lr_end)
        
    for i_iter in range(arg.num_iteration):
        optimizer.zero_grad()
        pde_loss = pinn_pat(LOSS.PDE)
        dpdt0_loss = pinn_pat(LOSS.DPDT0)

        if bc_exist:
            sensor_signal, sensor_xy = sensor_manager.next(i_iter)
            bc_loss = pinn_pat(LOSS.BC, xy=sensor_xy, p=sensor_signal)
        else:
            bc_loss = 0.

        if p0_exist:
            p0_weight = p0_manager.weight(i_iter)
            if p0_weight > 0:
                p0_xyt0, p0 = p0_manager.next(i_iter)
                p0_loss = pinn_pat(LOSS.P0, xyt0=p0_xyt0, p0=p0)
            else:
                p0_loss = 0.
        else:
            p0_loss, p0_weight = 0., 0.

        total_loss = bc_loss + pde_loss * arg.pde_weight + dpdt0_loss * arg.dpdt0_weight + p0_loss * p0_weight
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if print_when_save:
            psnr = recorder.record(i_iter, [total_loss, bc_loss, pde_loss, dpdt0_loss, p0_loss]) if (recorder is not None) else None
            if (i_iter + 1) % n_iter_per_save == 0:
                torch.save(pinn_pat_module.state_dict(), arg.save_filepath)
                print(f"\033[92m[{i_iter + 1}] | {arg.save_filepath} | {f'{psnr:.3f}' if psnr is not None else ''} | {datetime.now().strftime('%m/%d %H:%M:%S')}\033[0m")