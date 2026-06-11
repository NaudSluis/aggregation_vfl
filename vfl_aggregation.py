import random
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import polars as pl
import glob
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import math
import pandas as pd
import numpy as np


class VFLClientAttentionWeight(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()

        self.in_dim = in_dim
        self.hidden = hidden

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden)
        )

    def forward(self, ClientEmb: torch.Tensor) -> torch.Tensor:

        W_ClientEmb = self.net(ClientEmb)
    
        return W_ClientEmb
    
class ICAFSSelector(nn.Module):
    def __init__(self, in_dim: int, hidden: int, batch_size: int):
        super().__init__()

        self.in_dim = in_dim

        self.selector = torch.zeros([batch_size, in_dim])

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden)
        )

    def forward(self, z, inital_weights, gamma, epoch, total_epochs):

        tau = gamma ** (epoch/total_epochs)

        w_z = self.net(z)
        
        selector = nn.Sigmoid(tau * w_z)/nn.Sigmoid(inital_weights)

        return selector
    
    def select(self, z, z_tilde):

        s_tilde_n = self.selector * z + (1 - self.selector) * z_tilde

        return s_tilde_n








    

        