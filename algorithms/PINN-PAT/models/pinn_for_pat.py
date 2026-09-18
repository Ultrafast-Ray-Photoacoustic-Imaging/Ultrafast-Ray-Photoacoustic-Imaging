import torch
from torch import nn
from models.mlp import MLP
from models.arg import Arg, MEDIUM
import json
import os
from enum import Enum
class LOSS(Enum):
    BC = "BC"
    PDE = "PDE"
    DPDT0 = "DPDT0"
    P0 = "P0"
class PINNForPAT(nn.Module):
    def __init__(self, arg: Arg, device: torch.device):
        super().__init__()
        self.device = device
        self._net = MLP(arg.in_channels, arg.n_hidden, arg.l_layers, arg.activation)
        self.medium = arg.medium

        self.mse = nn.MSELoss()
        t_axis = torch.linspace(arg.t_min, arg.t_max, arg.n_t + 1)[:-1]
        self.register_buffer("t_axis", t_axis, persistent=False)
        self.n_t = arg.n_t
        self.in_channels = arg.in_channels
        self.pde_batch_size = arg.pde_batch_size
        self.t0_batch_size = arg.t0_batch_size

        if arg.load_filepath is not None:
            print("Load:", arg.load_filepath, self._net.load_state_dict(
                {k.removeprefix("_net."):v for k,v in torch.load(arg.load_filepath, map_location=device).items() if k.startswith("_net.")}
            ))
        self.register_arg(arg)
        print("\033[91m", "[PINNForPAT]", "\033[94m", arg.to_dict(), "\033[0m")

    def forward(self, loss: LOSS, **kwargs):
        if loss == LOSS.BC:
            return self.bc_loss(**kwargs)
        elif loss == LOSS.PDE:
            return self.pde_loss(**kwargs)
        elif loss == LOSS.DPDT0:
            return self.dpdt0_loss(**kwargs)
        elif loss == LOSS.P0:
            return self.p0_loss(**kwargs)

    def bc_loss(self, xy: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        B = xy.shape[0]
        xyt = torch.zeros([B, self.n_t, self.in_channels], device=self.device)
        xyt[:, :, :2] = xy.unsqueeze(1)
        xyt[:, :, 2] = self.t_axis

        p_bc = self.p(xyt)
        return self.mse(p_bc, p)

    def pde_loss(self, xyt: torch.Tensor=None):
        if xyt is None:
            xyt = torch.rand(self.pde_batch_size, self.in_channels, device=self.device, requires_grad=True)
        p = self.p(xyt)
        dp = torch.autograd.grad(
            p, xyt,
            create_graph=True,
            grad_outputs=torch.ones_like(p)
        )[0]
        d2pdx2, d2pdy2, d2pdt2 = [
            torch.autograd.grad(
                dp[:, idim], xyt,
                grad_outputs=torch.ones_like(dp[:, idim]),
                create_graph=True,
            )[0][:, idim] for idim in range(3)
        ]

        if self.medium == MEDIUM.A:
            lhs_pde = d2pdx2 + d2pdy2
        else:
            if self.medium in [MEDIUM.B, MEDIUM.D]: # heterogeneous
                x = xyt[:, 0]
                v = (torch.floor(x.detach() / 0.2) * 0.030 + 1.440) / 1.500
            if self.medium in [MEDIUM.C, MEDIUM.D]: # lossy
                if self.medium == MEDIUM.C:
                    mu = 9.84e-5
                else:
                    mu = 9.84e-5 / ((torch.floor(x.detach() / 0.2) * 0.030 + 1.440) / 1.500).square()
                d2pdxy2 = d2pdx2 + d2pdy2
                damping_term = mu * torch.autograd.grad(
                    d2pdxy2, xyt,
                    grad_outputs=torch.ones_like(d2pdxy2),
                    create_graph=True,
                )[0][:, 2]

            if self.medium == MEDIUM.B:
                lhs_pde = v ** 2 * (d2pdx2 + d2pdy2)
            elif self.medium == MEDIUM.C:
                lhs_pde = d2pdx2 + d2pdy2 + damping_term
            elif self.medium == MEDIUM.D:
                lhs_pde = v ** 2 * (d2pdx2 + d2pdy2) + damping_term
            else:
                assert False

        rhs_pde = d2pdt2
        return self.mse(lhs_pde, rhs_pde)

    def dpdt0_loss(self, xyt0: torch.Tensor=None):
        if xyt0 is None:
            xyt0 = torch.rand(self.t0_batch_size, self.in_channels, device=self.device)
            xyt0[:, 2] = 0
        xyt0.requires_grad_(True)
        p_t0 = self.p(xyt0)
        dpdt0 = torch.autograd.grad(
            p_t0, xyt0,
            create_graph=True,
            grad_outputs=torch.ones_like(p_t0)
        )[0][:, 2]
        return torch.mean(dpdt0.square())

    def p0_loss(self, xyt0, p0):
        return self.mse(self.p(xyt0), p0)

    def p(self, xyt):
        return self._net(xyt)

    @property
    def net(self):
        return self._net

    def register_param(self, name: str, value):
        if not isinstance(value, torch.Tensor):
            if isinstance(value, str):
                value = value.encode('utf-8')
                value = torch.tensor(list(value), dtype=torch.uint8, device=self.device)
            else:
                value = torch.tensor(value, device=self.device)
        self.register_buffer(name, value)

    def register_arg(self, arg: Arg):
        self.register_param("arg", json.dumps(arg.to_dict()))

