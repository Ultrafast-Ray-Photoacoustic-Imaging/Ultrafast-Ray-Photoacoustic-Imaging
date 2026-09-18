import torch
from torch.utils.data import Dataset, DataLoader

class SensorDataset(Dataset):
    def __init__(self, sensor_signal: torch.Tensor, sensor_xy: torch.Tensor):
        self.sensor_signal = sensor_signal
        self.sensor_xy = sensor_xy
        self.n_sensor = self.sensor_signal.shape[0]

    def __getitem__(self, idx):
        return self.sensor_signal[idx], self.sensor_xy[idx]

    def __len__(self):
        return self.n_sensor


class SensorManager:
    def __init__(self, sensor_filepath: str, batch_size: int, device: torch.device):
        self.device = device
        sensor_data = torch.load(sensor_filepath)
        signal, xy = sensor_data["signal"], sensor_data["xy"]
        self.n_batch = signal.shape[0] // batch_size if (batch_size is not None) else 1

        if self.n_batch > 1:
            dataset = SensorDataset(signal, xy)
            self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        else:
            self.signal, self.xy = signal.to(device), xy.to(device)
        self.dataloader_iter = None

    
    def next(self, i_iter: int):
        if self.n_batch > 1:
            if i_iter % self.n_batch == 0 or self.dataloader_iter is None:
                self.dataloader_iter = iter(self.dataloader)
            p, xy = next(self.dataloader_iter)
            return p.to(self.device), xy.to(self.device)
        else:
            return self.signal, self.xy