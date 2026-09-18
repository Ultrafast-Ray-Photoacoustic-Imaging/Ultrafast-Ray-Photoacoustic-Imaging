import torch
import math
from enum import Enum
from torch.utils.data import Dataset, DataLoader

max_iter_with_tr = 10000
max_tr_assist_weight = 1e-1
def tr_assist_weight(i_iter):
    if i_iter >= max_iter_with_tr:
        return 0.
    return max_tr_assist_weight * 0.5 * (math.cos(i_iter / max_iter_with_tr * math.pi) + 1.)

class P0Dataset(Dataset):
    def __init__(self, xyt: torch.Tensor, p0: torch.Tensor):
        self.xyt = xyt
        self.p0 = p0
        self.n_p0 = self.p0.shape[0]

    def __getitem__(self, idx):
        return self.xyt[idx], self.p0[idx]

    def __len__(self):
        return self.n_p0
    

class P0Manager:
    class P0_TYPE(Enum):
        TR_ASSIST = 0
        FORWARD_PROPAGATE = 1
    def __init__(self, p0_filepath: str, p0_type: P0_TYPE, batch_size: int, device: torch.device):
        self.p0_type = p0_type
        self.device = device

        p0_data = torch.load(p0_filepath)
        xy, p0 = p0_data["xy"], p0_data["p0"]
        assert len(xy.shape) == len(p0.shape) + 1
        if len(xy.shape) > 2:
            xy = xy.flatten(0, 1)
            p0 = p0.flatten()
        xyt = torch.zeros([xy.shape[0], 3])
        xyt[:, :2] = xy
        n_p0 = xyt.shape[0]
        

        self.n_batch = n_p0 // batch_size if batch_size is not None else 1
        if self.n_batch > 1:
            dataset = P0Dataset(xyt, p0)
            self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,)
        else:
            self.xyt, self.p0 = xyt.to(device), p0.to(device)
        self.dataloader_iter = None

    def weight(self, i_iter: int):
        return tr_assist_weight(i_iter) if self.p0_type == self.P0_TYPE.TR_ASSIST else 1.

    def next(self, i_iter: int):
        if self.n_batch > 1:
            if i_iter % self.n_batch == 0 or self.dataloader_iter is None:
                self.dataloader_iter = iter(self.dataloader)
            xyt, p0 = next(self.dataloader_iter)
            return xyt.to(self.device), p0.to(self.device)
        else:
            return self.xyt, self.p0
               