"""
This file contains the centralised DeepAR implementation as well as the dataset class for loading
"""

import random
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import glob
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import math



def gaussian_nll_loss(mu, target, sigma):
    """
    Loss function for Gaussian negative log-likelihood.
    
    Sigma is already ensured to be positive via _sigma() in the model,
    so no additional clamping is needed here.

    Args:
        mu: predicted mean (batch_size x pred_len x 1)
        target: true values (batch_size x pred_len x 1)
        sigma: predicted std dev (batch_size x pred_len x 1), already positive

    Returns:
        mean negative log-likelihood loss over the batch
    """
    
    variance = sigma ** 2
    loss = torch.log(sigma) + 0.5 * math.log(2 * math.pi) + 0.5 * ((target - mu) ** 2) / variance
    
    return loss.mean()

class CentralEncoder(nn.Module):
    """Simple MLP encoder to transform raw features into embeddings  """

    def __init__(self, in_dim: int, hidden: int, emb_dim:int):
        super(CentralEncoder, self).__init__()

        self.in_dim = in_dim
        self.hidden = hidden
        self.emb_dim = emb_dim

        self.net = nn.Sequential(

            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, emb_dim)
        )

    def forward(self, x):
        """x: (batch_size x seq_len x in_dim)        """
        batch_size, seq_len, dim = x.shape
        x = x.contiguous() # Ensure memory layout is correct for reshaping
        h = self.net(x.reshape(batch_size * seq_len, dim)).reshape(batch_size, seq_len, -1) # Reshape back to (batch_size x seq_len x emb_dim)

        return h
    
class DeepARAutoreg(nn.Module):

    def __init__(self, emb_dim: int, hidden: int = 128, 
                 pred_len: int = 24, sigma_eps: float = 1e-3):
        super(DeepARAutoreg, self).__init__()

        self.emb_dim = emb_dim
        self.hidden = hidden
        self.pred_len = pred_len
        self.sigma_eps = sigma_eps
    

        self.lstm_0 = nn.LSTMCell(input_size=emb_dim + 1, hidden_size=hidden) # +1 for previous target value input
        self.lstm_1 = nn.LSTMCell(input_size=hidden, hidden_size=hidden)
        self.lstm_2 = nn.LSTMCell(input_size=hidden, hidden_size=hidden)

        # Output heads for distribution parameters
        self.mu_head = nn.Linear(hidden, 1)
        self.sg_head = nn.Linear(hidden, 1)

    def _sigma(self, raw: torch.Tensor):
        """Ensure positive std dev output with a lower bound for numerical stability."""
        return F.softplus(raw) + self.sigma_eps
    
    def _init_states(self, batch_size):
        """Initialize LSTM states to zeros, ensuring they are on the same device as the model parameters."""

        device = next(self.parameters()).device
        
        states = []
        for _ in range(3):  # 3 layers
            h = torch.zeros(batch_size, self.hidden, device=device)
            c = torch.zeros(batch_size, self.hidden, device=device)
            states.append((h, c))

        return states
    
    def encode_context(self, emb_context, y_context):

        """Encode the context window to initialize LSTM states. Returns the final states after processing the context."""

        batch_size, time, emb_dim = emb_context.shape
        device = emb_context.device

        states = self._init_states(batch_size)
        
        y_prev = torch.zeros(batch_size, 1, device=device)

        for t in range(time):
            x_t = emb_context[:, t, :] #shape (batch_size x emb_dim)

            y_prev_2d = y_prev.view(batch_size, 1) # Ensure y_prev is (batch_size x 1) for concatenation

            inp = torch.cat([x_t, y_prev_2d], dim=-1) #shape (batch_size x (emb_dim + 1))
            
            # Layer 0 takes 'inp', Layer 1 and 2 take the 'h' from the layer below
            states[0] = self.lstm_0(inp, states[0])
            states[1] = self.lstm_1(states[0][0], states[1]) 
            states[2] = self.lstm_2(states[1][0], states[2]) 

            y_prev = y_context[:, t] #shape (batch_size x 1) for next step teacher forcing during context encoding

        return states
    def forward(self, emb_context, y_context, emb_future, y_future, tf_ratio=0.0):
            
            states = self.encode_context(emb_context, y_context)

            mu_out = []
            sg_out = []

            y_prev = y_context[:, -1]

            for t in range(self.pred_len):

                current_fut_emb = emb_future[:, t, :]

                inp = torch.cat([current_fut_emb, y_prev], dim=-1)

                states[0] = self.lstm_0(inp, states[0])
                states[1] = self.lstm_1(states[0][0], states[1]) 
                states[2] = self.lstm_2(states[1][0], states[2]) 

                mu = self.mu_head(states[2][0])
                sg = self._sigma(self.sg_head(states[2][0]))

                mu_out.append(mu)
                sg_out.append(sg)

                if y_future is not None and random.random() < tf_ratio:
                    y_prev = y_future[:, t]
                else:
                    if self.training:
                        y_prev = mu + sg * torch.randn_like(sg) 
                    else:
                        y_prev = mu

            mu = torch.stack(mu_out, dim=1)
            sg = torch.stack(sg_out, dim=1)
                
            return mu, sg
    
class DeepARAutoregDrop(nn.Module):
    def __init__(self, total_emb_dim: int, hidden: int = 128, 
                 pred_len: int = 24, sigma_eps: float = 1e-3, 
                 dropout: float = 0.2):
        super(DeepARAutoregDrop, self).__init__()

        self.total_emb_dim = total_emb_dim
        self.hidden = hidden
        self.pred_len = pred_len
        self.sigma_eps = sigma_eps
    
        self.lstm_0 = nn.LSTMCell(input_size=total_emb_dim, hidden_size=hidden) 
        self.lstm_1 = nn.LSTMCell(input_size=hidden, hidden_size=hidden)
        self.lstm_2 = nn.LSTMCell(input_size=hidden, hidden_size=hidden)
        
        self.dropout = nn.Dropout(p=dropout)

        self.mu_head = nn.Linear(hidden, 1)
        self.sg_head = nn.Linear(hidden, 1)

    def _sigma(self, raw: torch.Tensor):
        return F.softplus(raw) + self.sigma_eps
    
    def _init_states(self, batch_size):
        device = next(self.parameters()).device
        states = []
        for _ in range(3): 
            h = torch.zeros(batch_size, self.hidden, device=device)
            c = torch.zeros(batch_size, self.hidden, device=device)
            states.append((h, c))
        return states
    
    def encode_context(self, emb_context):
        batch_size, time, _ = emb_context.shape
        states = self._init_states(batch_size)
        
        for t in range(time):
            inp = emb_context[:, t, :] 
            
            states[0] = self.lstm_0(inp, states[0])
            drop_h0 = self.dropout(states[0][0])
            states[1] = self.lstm_1(drop_h0, states[1]) 
            drop_h1 = self.dropout(states[1][0])
            states[2] = self.lstm_2(drop_h1, states[2]) 

        return states
        
    def forward(self, emb_context, y_context, emb_future, y_future, tf_ratio=0.0):

        states = self.encode_context(emb_context)

        mu_out = []
        sg_out = []

        for t in range(self.pred_len):
            current_fut_emb = emb_future[:, t, :]
            
            inp = current_fut_emb 

            states[0] = self.lstm_0(inp, states[0])
            drop_h0 = self.dropout(states[0][0])
            states[1] = self.lstm_1(drop_h0, states[1]) 
            drop_h1 = self.dropout(states[1][0])
            states[2] = self.lstm_2(drop_h1, states[2]) 

            mu = self.mu_head(states[2][0])
            sg = self._sigma(self.sg_head(states[2][0]))

            mu_out.append(mu)
            sg_out.append(sg)

        mu = torch.stack(mu_out, dim=1)
        sg = torch.stack(sg_out, dim=1)
            
        return mu, sg

class CentralDataset(Dataset):
    def __init__(
        self,
        data_path,
        num_buildings=None,
        context=168,
        pred_window=24,
        target_col='electricity',
        is_train=True,
        split_date='2017-06-01',
        building_ids=None,
        random_seed=42
    ):

        if data_path is None:
            raise ValueError("data_path must be provided for CentralDataset")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Merged parquet file not found: {data_path}")

        self.data_path = data_path
        self.context = context
        self.pred_window = pred_window
        self.target_col = target_col
        self.is_train = is_train
        self.building_ids = building_ids
        self.total_window  = context + pred_window
        self.building_data = []
        self.valid_starts  = []

        split_dt = datetime.strptime(split_date, "%Y-%m-%d")
        df = pd.read_parquet(data_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["building", "timestamp"])

        available_buildings = df["building"].unique().tolist()

        if building_ids is None:
            if num_buildings is not None and 0 < num_buildings < len(available_buildings):
                random.seed(random_seed)
                building_ids = random.sample(available_buildings, num_buildings)
            else:
                building_ids = available_buildings
        else:
            building_ids = [b for b in building_ids if b in available_buildings]

        if len(building_ids) == 0:
            raise ValueError("No buildings were selected for CentralDataset")

        # IMPORTANT: this will hold only buildings that survive all checks,
        # aligned with self.building_data.
        self.building_ids = []

        global_numeric_df = df.select_dtypes(
            include=["float64", "float32", "int64", "int32", "int16", "int8"]
        )
        self.feature_names = global_numeric_df.columns.tolist()

        self.target_col_idx = self.feature_names.index(self.target_col)
        self.mean_col_idx   = self.feature_names.index("target_mean")
        self.std_col_idx    = self.feature_names.index("target_std")

        print(f"[INFO] Loading {len(building_ids)} buildings from {data_path} (is_train={is_train})")

        for bldg in building_ids:
            bldg_df = df[df["building"] == bldg].copy()
            if is_train:
                bldg_df = bldg_df[bldg_df["timestamp"] < split_dt]
            else:
                bldg_df = bldg_df[bldg_df["timestamp"] >= split_dt]

            if len(bldg_df) == 0:
                print(f"[WARNING] Building {bldg} has no rows after date split; skipping.")
                continue

            bldg_numeric_df = bldg_df[self.feature_names].copy().fillna(0.0)
            data_tensor = torch.tensor(bldg_numeric_df.to_numpy(), dtype=torch.float32)
            data_tensor = torch.nan_to_num(data_tensor, nan=0.0, posinf=1e4, neginf=-1e4)

            if torch.isnan(data_tensor).any():
                print(f"[WARNING] Building {bldg} still contains NaNs after preprocessing; skipping.")
                continue

            n_rows = data_tensor.shape[0]
            if n_rows < self.total_window:
                print(f"[WARNING] Building {bldg} has {n_rows} rows, less than required {self.total_window}; skipping.")
                continue

            # Store tensor and *aligned* building ID
            self.building_data.append(data_tensor)
            self.building_ids.append(bldg)
            bldg_idx_in_list = len(self.building_data) - 1

            num_samples = n_rows - self.total_window + 1
            valid_sample_count = 0

            for start_row in range(num_samples):
                # optionally filter low-variance windows, as you already do in VFL:
                y_future_window = data_tensor[
                    start_row + self.context : start_row + self.context + self.pred_window,
                    self.target_col_idx,
                ]
                if torch.std(y_future_window).item() > 0.1:
                    self.valid_starts.append((bldg_idx_in_list, start_row))
                    valid_sample_count += 1

            print(
                f"[INFO] Building {bldg}: {n_rows} rows -> {num_samples} windows, "
                f"{valid_sample_count} valid"
            )

        if len(self.valid_starts) == 0:
            raise ValueError("No valid windows were generated for CentralDataset.")

    def __len__(self):
        return len(self.valid_starts)
    
    def __getitem__(self, idx):
        bldg_idx, start_row = self.valid_starts[idx]
        bldg_tensor = self.building_data[bldg_idx]
        bldg_name_str = self.building_ids[bldg_idx]

        # This dynamically drops the target column
        features = torch.cat([
            bldg_tensor[:, :self.target_col_idx], 
            bldg_tensor[:, self.target_col_idx+1:]
        ], dim=1)

        x_context = features[start_row : start_row + self.context]
        x_future = features[start_row + self.context : start_row + self.context + self.pred_window]
        
        y_context = bldg_tensor[start_row : start_row + self.context, self.target_col_idx:self.target_col_idx+1]
        y_future = bldg_tensor[start_row + self.context : start_row + self.context + self.pred_window, self.target_col_idx:self.target_col_idx+1]

        bldg_mean = bldg_tensor[start_row, self.mean_col_idx]
        bldg_std = bldg_tensor[start_row, self.std_col_idx]

        return x_context, x_future, y_context, y_future, bldg_mean, bldg_std, bldg_name_str


