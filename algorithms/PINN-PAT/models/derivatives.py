import torch

def first_derivative(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y).float(),
        create_graph=True,
    )[0]

def second_derivative(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    dydx = first_derivative(y, x).view(y.shape)
    d2ydx2 = torch.autograd.grad(
        dydx, x,
        grad_outputs=torch.ones_like(y).float(),
        create_graph=True,
    )[0]
    return d2ydx2


def third_derivative(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    dydx = second_derivative(y, x).view(y.shape)
    d2ydx2 = torch.autograd.grad(
        dydx, x,
        grad_outputs=torch.ones_like(y).float(),
        create_graph=True,
    )[0]
    return d2ydx2