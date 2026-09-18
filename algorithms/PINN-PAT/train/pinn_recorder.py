import torch
import sys
import os
sys.path.append("..")
from models.pinn_for_pat import PINNForPAT
import json

class PINN_Recorder:
    def __init__(self,
        num_iteration: int,
        target_filepath: str=None,
        record_loss: bool=True,
        record_psnr: bool=True,
        record_img: bool=False,
        n_iter_per_record: int=100,
        n_iter_per_img: int=100000
    ):
        self.target_filepath = target_filepath
        self.num_iteration = num_iteration
        self.record_loss = record_loss
        self.record_psnr = record_psnr
        self.record_img = record_img
        self.n_iter_per_record = n_iter_per_record
        self.n_iter_per_img = n_iter_per_img

    def attach_model(self, model: PINNForPAT, device: torch.device=None):
        self.model = model
        if self.record_loss:
            n_loss_term, len_loss_list = 5, self.num_iteration // self.n_iter_per_record
            model.register_buffer("loss_list", torch.zeros([len_loss_list, n_loss_term], device=device))
            self.temp_loss_list = torch.zeros([self.n_iter_per_record, n_loss_term])

        if self.record_psnr:
            target_data = torch.load(self.target_filepath)
            self.img_size = target_data["xy"].shape[0]

            self.xyt = torch.zeros([self.img_size] * 2 + [3], device=device)
            self.xyt[:, :, :2] = target_data["xy"]
            self.mask, self.target_p0 = target_data["mask"].to(device), target_data["p0"].to(device)
            
            model.register_buffer("psnr_list", torch.zeros([len_loss_list], device=device))
            if self.record_img:
                self.img_record_dict = {k:i for i, k in enumerate(list(range(0, self.num_iteration + 1, self.n_iter_per_img))[1:])} 
                model.register_buffer("img_list", torch.zeros([len(self.img_record_dict)] + [self.img_size] * 2, device=device))
                model.register_param("img_record_dict", json.dumps(self.img_record_dict))

    def record(self, i_iter: int, loss_item_list: list):
        with torch.no_grad():
            if self.record_loss:
                self.temp_loss_list[i_iter % self.n_iter_per_record] = torch.tensor(loss_item_list).detach().cpu()
            if (i_iter + 1) % self.n_iter_per_record == 0:
                record_idx = i_iter // self.n_iter_per_record
                if self.record_loss:
                    self.model.loss_list[record_idx] = self.temp_loss_list.mean(dim=0)
                if self.record_psnr:
                    p0 = self.model.p(self.xyt)
                    psnr = self.calculate_PSNR(p0)
                    self.model.psnr_list[record_idx] = psnr
                    if self.record_img and (i_iter + 1) in self.img_record_dict.keys():
                        img_idx = self.img_record_dict[(i_iter + 1)] 
                        self.model.img_list[img_idx] = p0
                    return psnr
        return None
        
    def rescale_recon(self, pred): # least-square
        k = torch.sum(
            pred * self.target_p0 * self.mask
        ) / (torch.sum((pred * self.mask) ** 2))
        return pred * k

    def calculate_PSNR(self, pred):
        rescale_pred = self.rescale_recon(pred)
        mse = torch.sum((
            rescale_pred - self.target_p0
        ) ** 2 * self.mask) / torch.sum(self.mask)
        psnr = 10 * torch.log10(1 / mse)
        return psnr    