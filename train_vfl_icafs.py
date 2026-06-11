import os
import csv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from vfl_deepar import SplitDataset, ClientAProjector, ClientEncoder, DeepARAutoregSplit, gaussian_nll_loss
from vfl_aggregation import ICAFSSelector
from plots_metric_utils import plot_predicted_vs_actual, plot_training_validation_loss, plot_validation_metrics
import matplotlib.pyplot as plt
import optuna
from optuna.trial import TrialState
from tqdm import tqdm
import random
import argparse
import pandas as pd
import numpy as np
import math

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

def train_one_epoch(epoch_index, tb_writer, train_dataloader, models_opts_selectors, clientA_encoder, clientB_encoder, clientC_encoder, 
                    clientA_opt, clientB_opt, clientC_opt, total_epochs, selectors, selector_optimizers, initial_selector_weights,
                    beta, gamma):
    running_loss = 0.
    last_loss = 0.

    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), 
                desc=f'Epoch {epoch_index + 1}/{total_epochs} [TRAIN]', leave=False)

    for i, data in pbar:

        clientA_x_context, clientB_x_context, clientC_x_context, clientA_x_future, clientB_x_future, clientC_x_future, y_context, y_future = data

        clientA_x_context = clientA_x_context.to(device)
        clientA_x_future = clientA_x_future.to(device)

        clientB_x_context = clientB_x_context.to(device)
        clientB_x_future = clientB_x_future.to(device)

        clientC_x_context = clientC_x_context.to(device)
        clientC_x_future = clientC_x_future.to(device)

        y_context = y_context.to(device)
        y_future = y_future.to(device)

        clientA_emb_context = clientA_encoder(y_context, clientA_x_context)
        clientA_emb_future = clientA_encoder(y_future, clientA_x_future)

        clientB_emb_context = clientB_encoder(clientB_x_context)
        clientB_emb_future = clientB_encoder(clientB_x_future)

        clientC_emb_context = clientC_encoder(clientC_x_context)
        clientC_emb_future = clientC_encoder(clientC_x_future)

        z_context = torch.cat([clientA_emb_context, clientB_emb_context, clientC_emb_context], dim=-1)
        z_future = torch.cat([clientA_emb_future, clientB_emb_future, clientC_emb_future], dim=-1)

        # Vector with mean values to replace 0
        z_tilde_context = torch.full([z_context.shape], z_context.mean(dim=1))
        z_tilde_future = torch.full([z_future.shape], z_future.mean(dim=1))

            # Update selectors with current weights and annealing schedule
        for selector_key, selector in selectors.items():
            selector(z_context, initial_selector_weights[selector_key], gamma, epoch_index, total_epochs)
        
        mus = []
        sigmas = []

        clientA_opt.zero_grad()
        clientB_opt.zero_grad()
        clientC_opt.zero_grad()
        for model_key, (model, optimizer, _) in models_opts_selectors.items():
            optimizer.zero_grad()
        for selector_key, (selector, optimizer) in selector_optimizers.items():
            optimizer.zero_grad()

        # Forward pass through all models with their corresponding selectors
        for model_key, (model, _, model_selector) in models_opts_selectors.items():
            s_tilde_n_context = model_selector.select(z_context, z_tilde_context)
            s_tilde_n_future = model_selector.select(z_future, z_tilde_future)

            mu, sg = model(s_tilde_n_context, y_context, s_tilde_n_future, y_future)
            mus.append(mu)
            sigmas.append(sg)

        ensembled_mu = np.mean(mus)
        ensembled_mu_squared = ensembled_mu**2
        mu_squared = [i**2 for i in mus]

        variance = [i**2 for i in sigmas]
        ensembled_sg = np.mean(variance) + (np.mean(np.array(mu_squared) - ensembled_mu_squared))

        gaus_loss = gaussian_nll_loss(ensembled_mu, y_future, ensembled_sg)
        l2_penalty = torch.norm(selector, p=2)
        
        loss = gaus_loss + (beta * l2_penalty)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clip gradients to prevent explosion
        if torch.isnan(loss):
            print(f"[WARNING] NaN loss detected at epoch {epoch_index}, batch {i}")
            break

        clientA_opt.step()
        clientB_opt.step()
        clientC_opt.step()
        for model_key, (model, optimizer, _) in models_opts_selectors.items():
            optimizer.step()
        for selector_key, (selector, optimizer) in selector_optimizers.items():
            optimizer.step()

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
    target_col_idx,
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
                                target_col_idx=target_col_idx, 
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
                                target_col_idx=target_col_idx, 
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

clientA_cols = ['water', 'steam', 'irrigation', 'gas', 'hotwater', 'chilledwater', 'solar', 'hour', 'day_of_week', 'month', 'days_from_start']
clientB_cols = ['airTemperature', 'cloudCoverage', 'dewTemperature', 'precipDepth1HR', 'precipDepth6HR', 'seaLvlPressure', 'windDirection', 'windSpeed']
clientC_cols = ['site_id_meta', 'primaryspaceusage', 'sub_primaryspaceusage', 'sqm', 'timezone', 'industry', 'subindustry', 'heatingtype', 'yearbuilt',
                'numberoffloors', 'occupants', 'energystarscore', 'eui', 'site_eui', 'source_eui', 'leed_level', 'rating']

def run_many(
    context_length, pred_window, clientA_encoder_input_dim, clientB_encoder_input_dim, clientC_encoder_input_dim,
    encoder_hidden, encoder_emb_dim, model_hidden, learning_rate, batch_size, epochs, sigma_eps, grad_clip_norm,
    patience, min_delta, train_dataloader, val_dataloader, server, clientA_encoder, clientB_encoder,
    clientC_encoder, server_optimizer, clientA_optimizer, clientB_optimizer, clientC_optimizer, writer, 
    metrics_logger, checkpoint_dir, metrics_csv_path, plots_dir, models_opts_selectors, selectors, selector_optimizers, 
    initial_selector_weights, beta, gamma, resume_checkpoint=None
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
        'beta': beta,
        'gamma': gamma
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
        avg_loss = train_one_epoch(epoch, writer, train_dataloader, models_opts_selectors, clientA_encoder, clientB_encoder, clientC_encoder, 
                    clientA_optimizer, clientB_optimizer, clientC_optimizer, EPOCHS, selectors, selector_optimizers, initial_selector_weights,
                    beta, gamma)
        train_loss_history.append(avg_loss)

        # Validation
        running_vloss = 0.0
        running_mae = 0.0
        running_rmse = 0.0
        running_crps = 0.0

        server.eval()

        pbar_val = tqdm(enumerate(val_dataloader), total=len(val_dataloader), 
                        desc=f'Epoch {epoch + 1}/{EPOCHS} [VAL]', leave=False)
        
        if epoch < 20:
            val_tf_ratio = 0.2  # 20% TF early on
        else:
            val_tf_ratio = max(0.0, 0.05 * (1.0 - (epoch - 20) / (EPOCHS - 20)))

        last_val_mu = None
        last_val_true = None
        with torch.no_grad():
            for i, vdata in pbar_val:

                vclientA_x_context, vclientB_x_context, vclientC_x_context, vclientA_x_future, vclientB_x_future, vclientC_x_future, vy_context, vy_future = vdata

                vclientA_x_context = vclientA_x_context.to(device)
                vclientA_x_future = vclientA_x_future.to(device)

                vclientB_x_context = vclientB_x_context.to(device)
                vclientB_x_future = vclientB_x_future.to(device)

                vclientC_x_context = vclientC_x_context.to(device)
                vclientC_x_future = vclientC_x_future.to(device)

                vy_context = vy_context.to(device)
                vy_future = vy_future.to(device)

                # Generate embeddings from all three clients
                vclientA_emb_context = clientA_encoder(vy_context, vclientA_x_context)
                vclientA_emb_future = clientA_encoder(vy_future, vclientA_x_future)

                vclientB_emb_context = clientB_encoder(vclientB_x_context)
                vclientB_emb_future = clientB_encoder(vclientB_x_future)

                vclientC_emb_context = clientC_encoder(vclientC_x_context)
                vclientC_emb_future = clientC_encoder(vclientC_x_future)

                # Concatenate embeddings
                vz_context = torch.cat([vclientA_emb_context, vclientB_emb_context, vclientC_emb_context], dim=-1)
                vz_future = torch.cat([vclientA_emb_future, vclientB_emb_future, vclientC_emb_future], dim=-1)

                # Create mean-replacement tensors
                vz_tilde_context = torch.full([vz_context.shape], vz_context.mean(dim=1), device=device)
                vz_tilde_future = torch.full([vz_future.shape], vz_future.mean(dim=1), device=device)

                # Apply selectors and collect predictions from all models
                vmus = []
                vsigmas = []
                
                for model_key, (model, _, val_selector) in models_opts_selectors.items():
                    # Select features using selector
                    vs_tilde_n_context = val_selector.select(vz_context, vz_tilde_context)
                    vs_tilde_n_future = val_selector.select(vz_future, vz_tilde_future)

                    # Get predictions from model
                    vmu, vsg = model(vs_tilde_n_context, vy_context, vs_tilde_n_future, vy_future)
                    vmus.append(vmu)
                    vsigmas.append(vsg)

                # Ensemble predictions
                vensembled_mu = np.mean([m.detach().cpu().numpy() for m in vmus], axis=0)
                vensembled_mu_squared = vensembled_mu**2
                vmu_squared = [m.detach().cpu().numpy()**2 for m in vmus]
                
                vvariance = [s.detach().cpu().numpy()**2 for s in vsigmas]
                vensembled_sg = np.mean(vvariance, axis=0) + (np.mean(np.array(vmu_squared) - vensembled_mu_squared, axis=0))
                
                # Convert back to tensors
                vensembled_mu_tensor = torch.tensor(vensembled_mu, device=device, dtype=torch.float32)
                vensembled_sg_tensor = torch.tensor(vensembled_sg, device=device, dtype=torch.float32)
                
                vloss = gaussian_nll_loss(vensembled_mu_tensor, vy_future, vensembled_sg_tensor)
                running_vloss += vloss.item()

                mae = torch.abs(vensembled_mu_tensor - vy_future).mean().item()
                rmse = torch.sqrt(((vensembled_mu_tensor - vy_future)**2).mean()).item()
                crps = compute_crps(vensembled_mu_tensor, vensembled_sg_tensor, vy_future)

                running_mae += mae
                running_rmse += rmse
                running_crps += crps

                last_val_mu = vensembled_mu_tensor.cpu()
                last_val_true = vy_future.cpu()
            
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
            epoch + 1)

        writer.add_scalars('Training vs. Validation Loss',
            {
                'Training': avg_loss,
                'Validation': avg_vloss
            },
            epoch + 1)

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
    if last_val_mu is not None and last_val_true is not None:
        plot_predicted_vs_actual(last_val_true[0], last_val_mu[0], hours=24, save_path=pred_plot_path)

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
        
        trial_server = DeepARAutoregSplit(
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
                tf_ratio = max(0.0, 1.0 - (epoch / n_epochs))
                
                # Training phase
                trial_server.train()
                trial_cA.train(); trial_cB.train(); trial_cC.train()
                running_loss = 0.0
                
                for data in train_dl:
                    cA_xc, cB_xc, cC_xc, cA_xf, cB_xf, cC_xf, y_ctx, y_fut = [d.to(device) for d in data]
                    
                    embA_c = trial_cA(y_ctx, cA_xc)
                    embB_c = trial_cB(cB_xc)
                    embC_c = trial_cC(cC_xc)
                    
                    embA_f = trial_cA(y_fut, cA_xf)
                    embB_f = trial_cB(cB_xf)
                    embC_f = trial_cC(cC_xf)

                    emb_c = torch.cat([embA_c, embB_c, embC_c], dim=-1)
                    emb_f = torch.cat([embA_f, embB_f, embC_f], dim=-1)
                    
                    optimizer.zero_grad()
                    mu, sg = trial_server(emb_c, y_ctx, emb_f, y_fut, tf_ratio)
                    loss = gaussian_nll_loss(mu, y_fut, sg)
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(trial_server.parameters(), max_norm=grad_clip)
                    optimizer.step()
                    running_loss += loss.item()
                
                # Validation phase
                trial_server.eval()
                trial_cA.eval(); trial_cB.eval(); trial_cC.eval()
                running_vloss = 0.0
                
                with torch.no_grad():
                    for vdata in val_dl:
                        vcA_xc, vcB_xc, vcC_xc, vcA_xf, vcB_xf, vcC_xf, vy_ctx, vy_fut = [d.to(device) for d in vdata]
                        
                        vembA_c = trial_cA(vy_ctx, vcA_xc)
                        vembB_c = trial_cB(vcB_xc)
                        vembC_c = trial_cC(vcC_xc)
                        
                        vembA_f = trial_cA(vy_fut, vcA_xf)
                        vembB_f = trial_cB(vcB_xf)
                        vembC_f = trial_cC(vcC_xf)

                        vemb_c = torch.cat([vembA_c, vembB_c, vembC_c], dim=-1)
                        vemb_f = torch.cat([vembA_f, vembB_f, vembC_f], dim=-1)
                        
                        vmu, vsg = trial_server(vemb_c, vy_ctx, vemb_f, y_future=None)
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

    parser.add_argument('--target-col-idx', type=int, default=0,
                        help='Index of target column in numeric feature matrix')
    
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    
    parser.add_argument('--sigma-eps', type=float, default=1e-4,
                        help='Sigma Eps')
    
    parser.add_argument('--grad-clip', type=float, default=5,
                        help='GRadient Clipping norm')
    
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for early stopping')
    
    parser.add_argument('--min-delta', type=float, default=20,
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
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_dataset, val_dataset, train_dataloader, val_dataloader = build_dataloaders(
        data_path=args.data_file,
        num_buildings=args.num_buildings,
        context=args.context,
        pred_window=args.pred_window,
        target_col_idx=args.target_col_idx,
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

    server = DeepARAutoregSplit((args.emb_dim * 3), args.model_hidden, pred_len=args.pred_window, sigma_eps=args.sigma_eps)

    clientA_encoder = clientA_encoder.to(device)
    clientB_encoder = clientB_encoder.to(device)
    clientC_encoder = clientC_encoder.to(device)
    server = server.to(device)

    clientA_optimizer = torch.optim.Adam(clientA_encoder.parameters(), lr=args.learning_rate)
    clientB_optimizer = torch.optim.Adam(clientB_encoder.parameters(), lr=args.learning_rate)
    clientC_optimizer = torch.optim.Adam(clientC_encoder.parameters(), lr=args.learning_rate)
    server_optimizer = torch.optim.Adam(server.parameters(), lr=args.learning_rate)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/deepar_trainer_{}'.format(timestamp))

    # Initialize metrics logger
    metrics_log_file = f'runs/metrics_{timestamp}.csv'
    metrics_logger = MetricsLogger(metrics_log_file)

    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Initialize models, optimizers, and selectors for VFL-ICAFS
    num_models = 1  # Number of ensemble models (can be tuned)
    models_opts_selectors = {}  # Dict: model_key -> (model, model_opt, selector)
    selectors = {}  # Dict: selector_key -> selector
    selector_optimizers = {}  # Dict: selector_key -> (selector, selector_opt)
    initial_selector_weights = {}  # Dict: selector_key -> initialized weights
    beta = 0.01  # L2 regularization for selectors
    gamma = 1.0  # Temperature for selector annealing

    for i in range(num_models):
        model_key = f'model_{i}'
        selector_key = f'selector_{i}'
        
        model = DeepARAutoregSplit(args.emb_dim * 3, args.model_hidden, pred_len=args.pred_window, sigma_eps=args.sigma_eps)
        model = model.to(device)
        model_opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        
        selector = ICAFSSelector(args.emb_dim * 3)
        selector = selector.to(device)
        selector_opt = torch.optim.Adam(selector.parameters(), lr=args.learning_rate)
        
        # Initialize selector weights (e.g., uniform)
        init_weights = selector.net[0].weight
        
        models_opts_selectors[model_key] = (model, model_opt, selector)
        selectors[selector_key] = selector
        selector_optimizers[selector_key] = (selector, selector_opt)
        initial_selector_weights[selector] = init_weights

    run_many(
        args.context, args.pred_window, clientA_encoder_input_dim, clientB_encoder_input_dim, clientC_encoder_input_dim,
        args.encoder_hidden, args.emb_dim, args.model_hidden, args.learning_rate, args.batch_size, args.epochs, args.sigma_eps, args.grad_clip,
        args.patience, args.min_delta, train_dataloader, val_dataloader, server, clientA_encoder, clientB_encoder,
        clientC_encoder, server_optimizer, clientA_optimizer, clientB_optimizer, clientC_optimizer, writer, 
        metrics_logger, args.checkpoint_dir, metrics_log_file, plots_dir, models_opts_selectors, selectors, selector_optimizers,
        initial_selector_weights, beta, gamma, resume_checkpoint=args.resume_checkpoint
    )


if __name__ == '__main__':
    main()