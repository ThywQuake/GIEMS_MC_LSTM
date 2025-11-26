import xarray as xr
import os
import numpy as np
import torch

from utils.model import LSTMNetKAN
from utils.dataset import WetlandDataset
from utils.train import Train
from utils.eval import Eval

'''============================= Configs ======================================='''

TVARs = {
    "giems2": xr.open_dataset("data/clean/GIEMS-MC_fwet.nc")["fwet"],
    "era5": xr.open_dataset("data/clean/ERA5_tmp.nc")["tmp"],
    "mswep": xr.open_dataset("data/clean/MSWEP_pre.nc")["pre"],
    "gleam": xr.open_dataset("data/clean/GLEAM4a_sm.nc")["sm"],
    "grace": xr.open_dataset("data/clean/GRACE_lwe_thickness.nc")["lwe_thickness"]
} # TVARs with time series
CVARs = {
    "fcti": xr.open_dataset("data/clean/fcti.nc")["fcti"],
} # CVARs without time series
mask = xr.open_dataset("data/wetland_mask.nc")["mask"].values

device = torch.device("cpu")
