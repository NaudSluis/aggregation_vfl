import os
import csv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from vfl_deepar import DeepARAutoregSplitDrop, SplitDataset, ClientAProjector, ClientEncoder, DeepARAutoregSplit, gaussian_nll_loss
from plots_metric_utils import plot_training_validation_loss, plot_two_buildings_with_ci, plot_validation_metrics
import matplotlib.pyplot as plt
import optuna
from optuna.trial import TrialState
from tqdm import tqdm
import random
import argparse
import pandas as pd
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # Set PyTorch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

class MetricsLogger:
    """Log training and validation metrics to a CSV file."""
    
    def __init__(self, log_file):
        self.log_file = log_file
        self.fieldnames = [
            'epoch', 'train_loss', 'val_loss', 'val_mae', 'val_rmse', 'val_crps', 'epoch_time_seconds'
        ]
        
        # Write header if file doesn't exist
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
    
    def log(self, metrics):
        """Append metrics dictionary to log file."""
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(metrics)

def train_one_epoch(epoch_index, tb_writer, train_dataloader, model_opt, model, clientA_encoder, clientB_encoder, clientC_encoder, 
                    clientA_opt, clientB_opt, clientC_opt, total_epochs, grad_clip):
    running_loss = 0.
    last_loss = 0.

    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), 
                desc=f'Epoch {epoch_index + 1}/{total_epochs} [TRAIN]', leave=False)

    for i, data in pbar:

        clientA_x_context, clientB_x_context, clientC_x_context, clientA_x_future_safe, clientB_x_future, clientC_x_future, y_context, y_future, *_ = data

        clientA_x_context = clientA_x_context.to(device)
        clientA_x_future_safe = clientA_x_future_safe.to(device)

        clientB_x_context = clientB_x_context.to(device)
        clientB_x_future = clientB_x_future.to(device)

        clientC_x_context = clientC_x_context.to(device)
        clientC_x_future = clientC_x_future.to(device)

        y_context = y_context.to(device)
        y_future = y_future.to(device)

        clientA_emb_context = clientA_encoder(y_context, clientA_x_context)
        
        last_known_y = y_context[:, -1:, :] 
        
        dummy_y_future = last_known_y.repeat(1, y_future.shape[1], 1)
        clientA_emb_future = clientA_encoder(dummy_y_future, clientA_x_future_safe)

        clientB_emb_context = clientB_encoder(clientB_x_context)
        clientB_emb_future = clientB_encoder(clientB_x_future)

        clientC_emb_context = clientC_encoder(clientC_x_context)
        clientC_emb_future = clientC_encoder(clientC_x_future)

        emb_context = torch.cat([clientA_emb_context, clientB_emb_context, clientC_emb_context], dim=-1)
        emb_future = torch.cat([clientA_emb_future, clientB_emb_future, clientC_emb_future], dim=-1)

        clientA_opt.zero_grad()
        clientB_opt.zero_grad()
        clientC_opt.zero_grad()
        model_opt.zero_grad()

        mu, sg = model(emb_context, emb_future)
        #Output is send to clientA for loss calculation to avoid sending raw y_future to the server.
        loss = gaussian_nll_loss(mu, y_future, sg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip) # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(clientA_encoder.parameters(), max_norm=grad_clip)
        torch.nn.utils.clip_grad_norm_(clientB_encoder.parameters(), max_norm=grad_clip)
        torch.nn.utils.clip_grad_norm_(clientC_encoder.parameters(), max_norm=grad_clip)
        if torch.isnan(loss):
            print(f"[WARNING] NaN loss detected at epoch {epoch_index}, batch {i}")
            break

        clientA_opt.step()
        clientB_opt.step()
        clientC_opt.step()
        model_opt.step()

        running_loss += loss.item()

        last_loss = running_loss / len(train_dataloader)
    
        # Optional: Still log the final epoch loss to TensorBoard
        tb_writer.add_scalar('Loss/train_epoch_avg', last_loss, epoch_index)
        
        # Update progress bar with current loss
        pbar.set_postfix({'loss': f'{running_loss / (i + 1):.4f}'})

    return last_loss

def compute_crps(mu, sigma, y_true):
    """
    Compute CRPS (Continuous Ranked Probability Score) for Gaussian distribution.
    
    Args:
        mu: (batch, time, 1) - predicted mean
        sigma: (batch, time, 1) - predicted std
        y_true: (batch, time, 1) - actual values
    
    Returns:
        Scalar CRPS value (average across all predictions)
    """
    mu = mu.to(device)
    sigma = sigma.to(device)
    y_true = y_true.to(device)
    
    z = (y_true - mu) / (sigma + 1e-8)  # Avoid division by zero
    normal_dist = torch.distributions.Normal(
        torch.tensor(0.0, device=device), 
        torch.tensor(1.0, device=device)
    )
    
    pdf_z = torch.exp(normal_dist.log_prob(z))
    cdf_z = normal_dist.cdf(z)
    
    crps = sigma * (z * (2 * cdf_z - 1) + 2 * pdf_z - 1 / torch.sqrt(torch.tensor(torch.pi, device=device)))
    
    return crps.mean().item()

def save_checkpoint_vfl(checkpoint_path, model, clientA_encoder, clientB_encoder, clientC_encoder, model_optimizer, 
                        clientA_optimizer, clientB_optimizer, clientC_optimizer, epoch, best_loss):
    """
    Save model, encoder, optimizer, and metadata to checkpoint.
    
    Args:
        checkpoint_path: Path to save checkpoint
        model: DeepARAutoregSplit model
        clientA_encoder: ClientAProjector
        clientB_encoder: ClientBEncoder
        clientC_encoder: ClientCEncoder
        model_optimizer: Optimizer for server model
        clientA_optimizer: Optimizer for Client A encoder
        clientB_optimizer: Optimizer for Client B encoder
        clientC_optimizer: Optimizer for Client C encoder
        epoch: Current epoch number
        best_loss: Best validation loss so far
    """
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'clientA_encoder_state': clientA_encoder.state_dict(),
        'clientB_encoder_state': clientB_encoder.state_dict(),
        'clientC_encoder_state': clientC_encoder.state_dict(),
        'model_optimizer_state': model_optimizer.state_dict(),
        'clientA_optimizer_state': clientA_optimizer.state_dict(),
        'clientB_optimizer_state': clientB_optimizer.state_dict(),
        'clientC_optimizer_state': clientC_optimizer.state_dict(),
        'epoch': epoch,
        'best_loss': best_loss
    }, checkpoint_path)

def load_checkpoint_vfl(checkpoint_path, model, clientA_encoder, clientB_encoder, clientC_encoder, model_optimizer,
                         clientA_optimizer, clientB_optimizer, clientC_optimizer):
    """
    Load model, encoder, and optimizer state from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: DeepARAutoregSplit model
        clientA_encoder: ClientAProjector
        clientB_encoder: ClientBEncoder
        clientC_encoder: ClientCEncoder
        model_optimizer: Optimizer for server model
        clientA_optimizer: Optimizer for Client A encoder
        clientB_optimizer: Optimizer for Client B encoder
        clientC_optimizer: Optimizer for Client C encoder

    Returns:
        Tuple of (epoch_number, best_vloss)
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} not found. Starting from scratch.")
        return 0, 1_000_000.0
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    clientA_encoder.load_state_dict(checkpoint['clientA_encoder_state'])
    clientB_encoder.load_state_dict(checkpoint['clientB_encoder_state'])
    clientC_encoder.load_state_dict(checkpoint['clientC_encoder_state'])
    model_optimizer.load_state_dict(checkpoint['model_optimizer_state'])
    clientA_optimizer.load_state_dict(checkpoint['clientA_optimizer_state'])
    clientB_optimizer.load_state_dict(checkpoint['clientB_optimizer_state'])
    clientC_optimizer.load_state_dict(checkpoint['clientC_optimizer_state'])
    resuming_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    
    print(f"Checkpoint loaded. Resuming from epoch {resuming_epoch}, best loss: {best_loss:.6f}")
    return resuming_epoch, best_loss

def sample_building_ids(data_file, num_buildings, random_seed=42):
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")
    df = pd.read_parquet(data_file, columns=['building'])
    building_ids = df['building'].unique().tolist()
    if len(building_ids) == 0:
        raise ValueError("No buildings found in merged parquet file")
    if num_buildings is not None and num_buildings > 0 and num_buildings < len(building_ids):
        random.seed(random_seed)
        return random.sample(building_ids, num_buildings)
    return building_ids

def build_dataloaders(
    data_path,
    num_buildings,
    context,
    pred_window,
    clientA_cols,
    clientB_cols,
    clientC_cols,
    target_col,
    batch_size,
    num_workers,
    split_date='2017-06-01'
):
    building_ids = sample_building_ids(data_path, num_buildings)
    print(f"[INFO] Using {len(building_ids)} buildings for train/validation split")

    # ===== DATASET INITIALIZATION =====
    train_dataset = SplitDataset(data_path,
                                num_buildings=len(building_ids), 
                                context=context, 
                                pred_window=pred_window, 
                                target_col=target_col, 
                                clientA_cols=clientA_cols, 
                                clientB_cols=clientB_cols, 
                                clientC_cols=clientC_cols,
                                split_date=split_date,
                                building_ids=building_ids,
                                is_train=True,
                                )

    val_dataset = SplitDataset(data_path,
                               num_buildings=len(building_ids), 
                                context=context, 
                                pred_window=pred_window, 
                                target_col=target_col, 
                                clientA_cols=clientA_cols, 
                                clientB_cols=clientB_cols, 
                                clientC_cols=clientC_cols,
                                split_date=split_date,
                                building_ids=building_ids,
                                is_train=False)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    g = torch.Generator()
    g.manual_seed(42)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'),
        prefetch_factor=2,
        generator=g
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )
    return train_dataset, val_dataset, train_dataloader, val_dataloader

def get_encoder_input_dim(dataset):
    if len(dataset) == 0:
        raise ValueError("Train dataset is empty; cannot infer encoder input dimension")
    x_context, _, _, _ = dataset[0]
    return x_context.shape[-1]



def run_many(
    context_length, pred_window, clientA_encoder_input_dim, clientB_encoder_input_dim, clientC_encoder_input_dim,
    encoder_hidden, encoder_emb_dim, model_hidden, learning_rate, batch_size, epochs, sigma_eps, grad_clip_norm,
    patience, min_delta, train_dataloader, val_dataloader, server, clientA_encoder, clientB_encoder,
    clientC_encoder, server_optimizer, clientA_optimizer, clientB_optimizer, clientC_optimizer, writer, 
    metrics_logger, checkpoint_dir, metrics_csv_path, plots_dir, clientA_cols, clientB_cols, clientC_cols, resume_checkpoint=None,
):

    # ===== HYPERPARAMETER LOGGING =====
    hparams = {
        'context_length': context_length,
        'pred_window': pred_window,
        'clientA_encoder_in_dim': clientA_encoder_input_dim,
        'clientB_encoder_in_dim': clientB_encoder_input_dim,
        'clientC_encoder_in_dim': clientC_encoder_input_dim,
        'encoder_hidden': encoder_hidden,
        'encoder_emb_dim': encoder_emb_dim,
        'model_hidden': model_hidden,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs,
        'sigma_eps': sigma_eps,
        'grad_clip_norm': grad_clip_norm,
        'optimizer': 'Adam',
        'early_stopping_patience': patience,
        'early_stopping_min_delta': min_delta,
        'clientA_cols': clientA_cols,
        'clientB_cols': clientB_cols,
        'clientC_cols': clientC_cols
    }

    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    for key, value in hparams.items():
        print(f"  {key:<25} : {value}")
    print("=" * 60)

    os.makedirs(checkpoint_dir, exist_ok=True)

    epoch_number = 0
    best_vloss = 1_000_000.
    epochs_no_improve = 0

    # Load checkpoint if resuming
    if resume_checkpoint:
        epoch_number, best_vloss = load_checkpoint_vfl(resume_checkpoint, server, clientA_encoder, clientB_encoder, 
                                                       clientC_encoder, server_optimizer, clientA_optimizer, clientB_optimizer, clientC_optimizer)

    EPOCHS = epochs

    train_loss_history = []
    val_loss_history = []

    for epoch in range(epoch_number, EPOCHS):
        epoch_start_time = time.time()
        
        # Training 
        server.train(True)
        clientA_encoder.train(True)
        clientB_encoder.train(True)
        clientC_encoder.train(True)
        avg_loss = train_one_epoch(epoch, writer, train_dataloader, server_optimizer, server, clientA_encoder, clientB_encoder, clientC_encoder, clientA_optimizer,
                                   clientB_optimizer, clientC_optimizer, EPOCHS, grad_clip_norm)
        train_loss_history.append(avg_loss)

        # Validation
        running_vloss = 0.0
        running_mae = 0.0
        running_rmse = 0.0
        running_crps = 0.0

        server.eval()
        clientA_encoder.eval()
        clientB_encoder.eval()
        clientC_encoder.eval()

        pbar_val = tqdm(enumerate(val_dataloader), total=len(val_dataloader), 
                        desc=f'Epoch {epoch + 1}/{EPOCHS} [VAL]', leave=False)


        last_val_mu = None
        last_val_true = None
        with torch.no_grad():
            for i, vdata in pbar_val:

                vclientA_x_context, vclientB_x_context, vclientC_x_context, vclientA_x_future_safe, vclientB_x_future, vclientC_x_future, vy_context, vy_future, bldg_mean, bldg_std, bldg_names = vdata

                vclientA_x_context = vclientA_x_context.to(device)
                vclientA_x_future_safe = vclientA_x_future_safe.to(device)

                vclientB_x_context = vclientB_x_context.to(device)
                vclientB_x_future = vclientB_x_future.to(device)

                vclientC_x_context = vclientC_x_context.to(device)
                vclientC_x_future = vclientC_x_future.to(device)

                vy_context = vy_context.to(device)
                vy_future = vy_future.to(device)

                bldg_mean = bldg_mean.to(device).view(-1, 1, 1)
                bldg_std = bldg_std.to(device).view(-1, 1, 1)

                vclientA_emb_context = clientA_encoder(vy_context, vclientA_x_context)

                # Pass a dummy y for the future
                last_known_vy = vy_context[:, -1:, :] 
                
                # Stretch that last value across the entire 24h future window
                dummy_vy_future = last_known_vy.repeat(1, vy_future.shape[1], 1)
                vclientA_emb_future = clientA_encoder(dummy_vy_future, vclientA_x_future_safe)

                vclientB_emb_context = clientB_encoder(vclientB_x_context)
                vclientB_emb_future = clientB_encoder(vclientB_x_future)

                vclientC_emb_context = clientC_encoder(vclientC_x_context)
                vclientC_emb_future = clientC_encoder(vclientC_x_future)

                vemb_context = torch.cat([vclientA_emb_context, vclientB_emb_context, vclientC_emb_context], dim=-1)
                vemb_future = torch.cat([vclientA_emb_future, vclientB_emb_future, vclientC_emb_future], dim=-1)

                vmu, vsg = server(vemb_context, vemb_future)
                
                # Calculate validation loss on the actual autoregressive rollout
                vloss = gaussian_nll_loss(vmu, vy_future, vsg)
                running_vloss += vloss.item()

                true_vy_future = (vy_future * bldg_std) + bldg_mean
                true_vmu = (vmu * bldg_std) + bldg_mean
                true_vsg = vsg * bldg_std

                mae = torch.abs(true_vmu - true_vy_future).mean().item()
                rmse = torch.sqrt(((true_vmu - true_vy_future)**2).mean()).item()
                crps = compute_crps(true_vmu, true_vsg, true_vy_future)

                running_mae += mae
                running_rmse += rmse
                running_crps += crps

                last_val_mu = true_vmu.cpu().numpy()
                last_val_true = true_vy_future.cpu().numpy()
                last_val_sg = true_vsg.cpu().numpy()
                last_val_bldg_names = bldg_names


                h_sin_idx = clientA_cols.index('hour_sin')
                h_cos_idx = clientA_cols.index('hour_cos')
                d_sin_idx = clientA_cols.index('day_of_week_sin')
                d_cos_idx = clientA_cols.index('day_of_week_cos')
                time_indices = [h_sin_idx, h_cos_idx, d_sin_idx, d_cos_idx]
                
                last_val_time_features = vclientA_x_future_safe[:, :, time_indices].cpu().numpy()
            
                num_batches = i + 1

        avg_vloss = running_vloss / num_batches
        avg_mae = running_mae / num_batches
        avg_rmse = running_rmse / num_batches
        avg_crps = running_crps / num_batches

        val_loss_history.append(avg_vloss)

        # Log metrics to TensorBoard
        writer.add_scalars('Validation Metrics',
            {
                'MAE': avg_mae,
                'RMSE': avg_rmse,
                'CRPS': avg_crps
            },
            epoch)

        writer.add_scalars('Training vs. Validation Loss',
            {
                'Training': avg_loss,
                'Validation': avg_vloss
            },
            epoch)

        writer.flush()

        # Calculate epoch runtime
        epoch_elapsed = time.time() - epoch_start_time
        
        # Log metrics to CSV file
        metrics_logger.log({
            'epoch': epoch + 1,
            'train_loss': f'{avg_loss:.6f}',
            'val_loss': f'{avg_vloss:.6f}',
            'val_mae': f'{avg_mae:.6f}',
            'val_rmse': f'{avg_rmse:.6f}',
            'val_crps': f'{avg_crps:.6f}',
            'epoch_time_seconds': f'{epoch_elapsed:.2f}'
        })

        print(f"Loss: {avg_loss:.4f} | Val Loss: {avg_vloss:.4f} | MAE: {avg_mae:.4f} | RMSE: {avg_rmse:.4f}")

        # Save periodic checkpoint
        periodic_checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
        save_checkpoint_vfl(periodic_checkpoint_path, server, clientA_encoder, clientB_encoder,
                            clientC_encoder, server_optimizer, clientA_optimizer, clientB_optimizer,
                            clientC_optimizer, epoch, best_vloss)

        timestamp = datetime.now()

        # Early stopping logic
        if avg_vloss < best_vloss - min_delta:
            best_vloss = avg_vloss
            epochs_no_improve = 0
            best_model_path = os.path.join(checkpoint_dir, f'best_model_{timestamp}.pth')
            save_checkpoint_vfl(best_model_path, server, clientA_encoder, clientB_encoder,
                            clientC_encoder, server_optimizer, clientA_optimizer, clientB_optimizer,
                            clientC_optimizer, epoch, best_vloss)
            print(f"  → Best val loss: {best_vloss:.6f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break # Exit the training loop
    
    print("\n[SUMMARY] Training complete! Metrics saved to: " + metrics_csv_path)
    writer.close()

    # Visualize and save training / validation metrics plots
    os.makedirs(plots_dir, exist_ok=True)
    loss_plot_path = os.path.join(plots_dir, 'training_validation_loss.png')
    val_plot_path = os.path.join(plots_dir, 'validation_metrics.png')
    pred_plot_path = os.path.join(plots_dir, 'predicted_vs_actual_24h.png')

    print(f"\n[SUMMARY] Generating plots:\n  - {loss_plot_path}\n  - {val_plot_path}\n  - {pred_plot_path}\n")
    plot_training_validation_loss(metrics_csv_path, save_path=loss_plot_path)
    plot_validation_metrics(metrics_csv_path, save_path=val_plot_path)
    if last_val_mu is not None and last_val_true is not None and last_val_sg is not None:
        usages = [name.split('_')[1] if '_' in name else "Unknown" for name in last_val_bldg_names]
        
        plot_two_buildings_with_ci(
            actuals=last_val_true,           
            predicted_mus=last_val_mu,       
            predicted_sigmas=last_val_sg,    
            building_names=last_val_bldg_names,
            building_usages=usages,
            time_features=last_val_time_features, # <-- Pass the time features here
            hours=24, 
            save_path=pred_plot_path
        )

def hyperparameter_tune(train_dataset, val_dataset, output_dir, n_trials=20, n_epochs=3):
    """
    Hyperparameter tuning using Optuna for VFL DeepAR model.
    """
    # Extract input dimensions for the encoders
    sample_data = train_dataset[0]
    clientA_dim = sample_data[0].shape[-1]
    clientB_dim = sample_data[1].shape[-1]
    clientC_dim = sample_data[2].shape[-1]

    def objective(trial):
        # Suggest hyperparameters (matching central ranges)
        encoder_hidden = trial.suggest_int('encoder_hidden', 64, 128, step=32)
        encoder_emb_dim = trial.suggest_int('encoder_emb_dim', 16, 64, step=8)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        sigma_eps = trial.suggest_float('sigma_eps', 1e-8, 1e-2, log=True)
        grad_clip = trial.suggest_float('grad_clip', 0.5, 5.0)
        batch_size_trial = trial.suggest_categorical('batch_size', [256, 512, 1024])
        model_hidden = trial.suggest_int('model_hidden', 64, 256, step=64)

        g = torch.Generator()
        g.manual_seed(42)
        train_dl = DataLoader(train_dataset, batch_size=batch_size_trial, shuffle=True, num_workers=4, generator=g)
        val_dl = DataLoader(val_dataset, batch_size=batch_size_trial, shuffle=False, num_workers=4)
        
        # Create fresh encoders and server model for this trial
        trial_cA = ClientAProjector(clientA_dim, encoder_hidden, encoder_emb_dim).to(device)
        trial_cB = ClientEncoder(clientB_dim, encoder_hidden, encoder_emb_dim).to(device)
        trial_cC = ClientEncoder(clientC_dim, encoder_hidden, encoder_emb_dim).to(device)
        
        trial_server = DeepARAutoregSplitDrop(
            encoder_emb_dim * 3, 
            model_hidden, 
            pred_len=24, 
            sigma_eps=sigma_eps
        ).to(device)
        
        optimizer = torch.optim.Adam(
            list(trial_server.parameters()) + list(trial_cA.parameters()) + 
            list(trial_cB.parameters()) + list(trial_cC.parameters()), 
            lr=learning_rate
        )
        
        best_val_loss = float('inf')
        
        with tqdm(range(n_epochs), desc=f'Trial {trial.number + 1}/{n_trials}', leave=False) as epoch_pbar:
            for epoch in epoch_pbar:
                
                # Training phase
                trial_server.train()
                trial_cA.train(); trial_cB.train(); trial_cC.train()
                running_loss = 0.0
                
                for data in train_dl:
                    cA_xc, cB_xc, cC_xc, cA_xf_safe, cB_xf, cC_xf, y_ctx, y_fut, *_ = data

                    cA_xc = cA_xc.to(device)
                    cB_xc = cB_xc.to(device)
                    cC_xc = cC_xc.to(device)
                    cA_xf_safe = cA_xf_safe.to(device)
                    cB_xf = cB_xf.to(device)
                    cC_xf = cC_xf.to(device)
                    y_ctx = y_ctx.to(device)
                    y_fut = y_fut.to(device)
                    
                    embA_c = trial_cA(y_ctx, cA_xc)
                    embB_c = trial_cB(cB_xc)
                    embC_c = trial_cC(cC_xc)
                    
                    dummy_y_fut = y_ctx[:, -1:, :].repeat(1, y_fut.shape[1], 1)
                    embA_f = trial_cA(dummy_y_fut, cA_xf_safe)
                    embB_f = trial_cB(cB_xf)
                    embC_f = trial_cC(cC_xf)

                    emb_c = torch.cat([embA_c, embB_c, embC_c], dim=-1)
                    emb_f = torch.cat([embA_f, embB_f, embC_f], dim=-1)
                    
                    optimizer.zero_grad()
                    mu, sg = trial_server(emb_c, emb_f)
                    loss = gaussian_nll_loss(mu, y_fut, sg)
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(trial_server.parameters(), max_norm=grad_clip)
                    torch.nn.utils.clip_grad_norm_(trial_cA.parameters(), max_norm=grad_clip)
                    torch.nn.utils.clip_grad_norm_(trial_cB.parameters(), max_norm=grad_clip)
                    torch.nn.utils.clip_grad_norm_(trial_cC.parameters(), max_norm=grad_clip)
                    optimizer.step()
                    running_loss += loss.item()
                
                # Validation phase
                trial_server.eval()
                trial_cA.eval(); trial_cB.eval(); trial_cC.eval()
                running_vloss = 0.0
                
                with torch.no_grad():
                    for vdata in val_dl:
                        vcA_xc, vcB_xc, vcC_xc, vcA_xf_safe, vcB_xf, vcC_xf, vy_ctx, vy_fut, *_ = vdata

                        vcA_xc = vcA_xc.to(device)
                        vcB_xc = vcB_xc.to(device)
                        vcC_xc = vcC_xc.to(device)
                        vcA_xf_safe = vcA_xf_safe.to(device)
                        vcB_xf = vcB_xf.to(device)
                        vcC_xf = vcC_xf.to(device)
                        vy_ctx = vy_ctx.to(device)
                        vy_fut = vy_fut.to(device)
                        
                        vembA_c = trial_cA(vy_ctx, vcA_xc)
                        vembB_c = trial_cB(vcB_xc)
                        vembC_c = trial_cC(vcC_xc)
                        
                        dummy_vy_fut = vy_ctx[:, -1:, :].repeat(1, vy_fut.shape[1], 1)
                        vembA_f = trial_cA(dummy_vy_fut, vcA_xf_safe)
                        vembB_f = trial_cB(vcB_xf)
                        vembC_f = trial_cC(vcC_xf)

                        vemb_c = torch.cat([vembA_c, vembB_c, vembC_c], dim=-1)
                        vemb_f = torch.cat([vembA_f, vembB_f, vembC_f], dim=-1)
                        
                        vmu, vsg = trial_server(vemb_c, vemb_f)
                        vloss = gaussian_nll_loss(vmu, vy_fut, vsg)
                        running_vloss += vloss.item()
                
                avg_vloss = running_vloss / len(val_dl)
                best_val_loss = min(best_val_loss, avg_vloss)
                
                trial.report(avg_vloss, epoch)
                epoch_pbar.set_postfix({'val_loss': f'{avg_vloss:.4f}', 'best': f'{best_val_loss:.4f}'})
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
            del train_dl, val_dl

        return best_val_loss
    
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler, pruner=optuna.pruners.MedianPruner())
    
    print(f"\n{'='*60}")
    print(f"VFL HYPERPARAMETER TUNING WITH OPTUNA")
    print(f"Trials: {n_trials} | Epochs per trial: {n_epochs}")
    print(f"{'='*60}")
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_trial = study.best_trial
    print(f"\n{'='*60}")
    print(f"TUNING COMPLETE")
    print(f"Best validation loss: {best_trial.value:.6f}")
    print(f"\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key:<20}: {value}")
    print(f"{'='*60}\n")
    
    trials_df = study.trials_dataframe()
    csv_path = os.path.join(output_dir, 'optuna_trials_history.csv')
    trials_df.to_csv(csv_path, index=False)
    
    return best_trial.params
    

def main():
    parser = argparse.ArgumentParser(description='Train DeepAR on a merged parquet dataset.')
    parser.add_argument('--tune', action='store_true',
                        help='Trigger Optuna hyperparameter tuning before main training loop')
    
    parser.add_argument('--data-file', type=str, default='final_lstm_dataset_merged.parquet',
                        help='Path to the merged parquet dataset')
    
    parser.add_argument('--num-buildings', type=int, default=100,
                        help='Number of buildings to sample from the merged dataset')
    
    parser.add_argument('--context', type=int, default=168,
                        help='Context window length')
    
    parser.add_argument('--pred-window', type=int, default=24,
                        help='Prediction window length')
    
    parser.add_argument('--encoder-hidden', type=int, default=64,
                        help='Hidden dimention of encoders')
    
    parser.add_argument('--emb-dim', type=int, default=16,
                        help='Output dimension of encoder')
    
    parser.add_argument('--model-hidden', type=int, default=128,
                        help='Hidden dimension of model')
    
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning Rate')
    
    parser.add_argument('--batch-size', type=int, default=512,
                        help='PyTorch batch size')

    parser.add_argument('--target-col', type=str, default='electricity',
                        help='Name of target column in numeric feature matrix')
    
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    
    parser.add_argument('--sigma-eps', type=float, default=1e-4,
                        help='Sigma Eps')
    
    parser.add_argument('--grad-clip', type=float, default=5,
                        help='GRadient Clipping norm')
    
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for early stopping')
    
    parser.add_argument('--min-delta', type=float, default=0.001,
                        help='Minimum delta for early stopping')
    
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of DataLoader worker processes')
    
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    
    parser.add_argument('--output-dir', type=str, default='runs',
                        help='Directory for TensorBoard logs and metrics')
    
    parser.add_argument('--resume-checkpoint', type=str, default=None,
                        help='Path to checkpoint file to resume training')
    
    parser.add_argument('--split-date', type=str, default='2017-06-01',
                        help='Train/validation date split')
    
    parser.add_argument('--clientA-cols', nargs='+', default=['water', 'steam', 'irrigation', 'gas', 
                                                              'hotwater', 'chilledwater', 'solar', 
                                                              'hour_sin', 'hour_cos', 'day_of_week_sin', 
                                                              'day_of_week_cos', 'month_sin', 'month_cos', 
                                                              'days_from_start'],
                        help='List of column names for Client A features')
    parser.add_argument('--clientB-cols', nargs='+', default=['airTemperature', 'cloudCoverage', 'dewTemperature', 
                                                              'precipDepth1HR', 'precipDepth6HR', 'seaLvlPressure', 
                                                              'windDirection', 'windSpeed'],
                        help='List of column names for Client B features')
    parser.add_argument('--clientC-cols', nargs='+', default=['site_id_meta', 'primaryspaceusage', 'sub_primaryspaceusage', 
                                                              'sqm', 'timezone', 'industry', 'subindustry', 'heatingtype', 'yearbuilt',
                                                              'numberoffloors', 'occupants', 'energystarscore', 'eui', 
                                                              'site_eui', 'source_eui', 'leed_level', 'rating'],
                        help='List of column names for Client C features')
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_dataset, val_dataset, train_dataloader, val_dataloader = build_dataloaders(
        data_path=args.data_file,
        num_buildings=args.num_buildings,
        context=args.context,
        pred_window=args.pred_window,
        clientA_cols=args.clientA_cols,
        clientB_cols=args.clientB_cols,
        clientC_cols=args.clientC_cols,
        target_col=args.target_col,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_date=args.split_date
    )

    if args.tune:
            best_params = hyperparameter_tune(train_dataset, val_dataset, args.output_dir)
            if best_params:
                # Sync hyperparameters found by Optuna
                args.batch_size = best_params.get('batch_size', args.batch_size)
                args.learning_rate = best_params.get('learning_rate', args.learning_rate)
                args.encoder_hidden = best_params.get('encoder_hidden', args.encoder_hidden)
                args.emb_dim = best_params.get('encoder_emb_dim', args.emb_dim)
                args.model_hidden = best_params.get('model_hidden', args.model_hidden)
                args.sigma_eps = best_params.get('sigma_eps', args.sigma_eps)
                args.grad_clip = best_params.get('grad_clip', args.grad_clip)

                print(f"\n[INFO] Re-initializing dataloaders with optimized batch size: {args.batch_size}")
                g = torch.Generator()
                g.manual_seed(42)
                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, generator=g)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    sample_data = train_dataset[0]
    clientA_encoder_input_dim = sample_data[0].shape[-1]
    clientB_encoder_input_dim = sample_data[1].shape[-1]
    clientC_encoder_input_dim = sample_data[2].shape[-1]

    clientA_encoder = ClientAProjector(clientA_encoder_input_dim, args.encoder_hidden, args.emb_dim)    
    clientB_encoder = ClientEncoder(clientB_encoder_input_dim, args.encoder_hidden, args.emb_dim)
    clientC_encoder = ClientEncoder(clientC_encoder_input_dim, args.encoder_hidden, args.emb_dim)

    server = DeepARAutoregSplitDrop((args.emb_dim * 3), args.model_hidden, pred_len=args.pred_window, sigma_eps=args.sigma_eps, dropout=args.dropout)

    clientA_encoder = clientA_encoder.to(device)
    clientB_encoder = clientB_encoder.to(device)
    clientC_encoder = clientC_encoder.to(device)
    server = server.to(device)

    clientA_optimizer = torch.optim.Adam(clientA_encoder.parameters(), lr=args.learning_rate)
    clientB_optimizer = torch.optim.Adam(clientB_encoder.parameters(), lr=args.learning_rate)
    clientC_optimizer = torch.optim.Adam(clientC_encoder.parameters(), lr=args.learning_rate)
    server_optimizer = torch.optim.Adam(server.parameters(), lr=args.learning_rate)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    writer = SummaryWriter(os.path.join(args.output_dir, f'deepar_trainer_{timestamp}'))

    metrics_log_file = os.path.join(args.output_dir, f'metrics_{timestamp}.csv')
    metrics_logger = MetricsLogger(metrics_log_file)

    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    run_many(
    args.context, args.pred_window, clientA_encoder_input_dim, clientB_encoder_input_dim, clientC_encoder_input_dim,
    args.encoder_hidden, args.emb_dim, args.model_hidden, args.learning_rate, args.batch_size, args.epochs, args.sigma_eps, args.grad_clip,
    args.patience, args.min_delta, train_dataloader, val_dataloader, server, clientA_encoder, clientB_encoder,
    clientC_encoder, server_optimizer, clientA_optimizer, clientB_optimizer, clientC_optimizer, writer, 
    metrics_logger, args.checkpoint_dir, metrics_log_file, plots_dir, args.clientA_cols, args.clientB_cols, args.clientC_cols, resume_checkpoint=None,
)


if __name__ == '__main__':
    main()