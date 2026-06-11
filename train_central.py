import os
import csv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from central_deepar import CentralDataset, CentralEncoder, DeepARAutoreg, gaussian_nll_loss
from plots_metric_utils import plot_training_validation_loss, plot_two_buildings_with_ci, plot_validation_metrics
import argparse
import matplotlib.pyplot as plt
import optuna
from tqdm import tqdm
import random
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

import numpy as np # Add this to your imports

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

def train_one_epoch(epoch_index, tb_writer, train_dataloader, optimizer, model, encoder, total_epochs):
    running_loss = 0.
    total_loss = 0.
    batch_count = 0

    # Linearly decay Teacher Forcing from 1.0 (start) down to 0.0 (end)
    tf_ratio = max(0.0, 1.0 - (epoch_index / total_epochs))

    pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                desc=f'Epoch {epoch_index + 1}/{total_epochs} [TRAIN]', leave=False)

    for i, data in pbar:

        x_context, x_future, y_context, y_future, *_ = data
        x_context = x_context.to(device)
        x_future = x_future.to(device)
        y_context = y_context.to(device)
        y_future = y_future.to(device)

        emb_context = encoder(x_context)
        emb_future = encoder(x_future)

        optimizer.zero_grad()

        mu, sg = model(emb_context, y_context, emb_future, y_future, tf_ratio)

        loss = gaussian_nll_loss(mu, y_future, sg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(encoder.parameters()), max_norm=1.0) # Clip gradients to prevent explosion
        
        # Check for NaN/inf before stepping
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[ERROR] NaN/Inf loss at epoch {epoch_index}, batch {i}: {loss.item()}")
            print(f"  mu stats: min={mu.min().item():.4f}, max={mu.max().item():.4f}, mean={mu.mean().item():.4f}")
            print(f"  sg stats: min={sg.min().item():.4f}, max={sg.max().item():.4f}, mean={sg.mean().item():.4f}")
            break

        optimizer.step()

        loss_val = loss.item()
        running_loss += loss_val
        total_loss += loss_val
        batch_count += 1

        # Log every 100 batches to TensorBoard (not reset, just for monitoring)
        if (i + 1) % 100 == 0:
            tb_x = epoch_index * len(train_dataloader) + i + 1
            tb_writer.add_scalar('Loss/train_batch', loss_val, tb_x)

        # Update progress bar with running average
        pbar.set_postfix({'loss': f'{total_loss / batch_count:.6f}'})

    # Return proper epoch average
    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    return avg_loss

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

def save_checkpoint(checkpoint_path, model, encoder, optimizer, epoch, best_loss):
    """
    Save model, encoder, optimizer, and metadata to checkpoint.

    Args:
        checkpoint_path: Path to save checkpoint
        model: DeepARAutoreg model
        encoder: CentralEncoder
        optimizer: Optimizer
        epoch: Current epoch number
        best_loss: Best validation loss so far
    """
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'encoder_state': encoder.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch,
        'best_loss': best_loss
    }, checkpoint_path)

def load_checkpoint(checkpoint_path, model, encoder, optimizer):
    """
    Load model, encoder, and optimizer state from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: DeepARAutoreg model
        encoder: CentralEncoder
        optimizer: Optimizer

    Returns:
        Tuple of (epoch_number, best_vloss)
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} not found. Starting from scratch.")
        return 0, 1_000_000.0

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    encoder.load_state_dict(checkpoint['encoder_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    resuming_epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']

    print(f"Checkpoint loaded. Resuming from epoch {resuming_epoch}, best loss: {best_loss:.6f}")
    return resuming_epoch, best_loss

random.seed(42)  # For reproducibility

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
    data_file,
    num_buildings,
    context,
    pred_window,
    target_col,
    batch_size,
    num_workers,
    seed=42,
    split_date='2017-06-01'
):
    building_ids = sample_building_ids(data_file, num_buildings)
    print(f"[INFO] Using {len(building_ids)} buildings for train/validation split")

    train_dataset = CentralDataset(
        data_path=data_file,
        num_buildings=len(building_ids),
        context=context,
        pred_window=pred_window,
        target_col=target_col,
        is_train=True,
        split_date=split_date,
        building_ids=building_ids
    )
    val_dataset = CentralDataset(
        data_path=data_file,
        num_buildings=len(building_ids),
        context=context,
        pred_window=pred_window,
        target_col=target_col,
        is_train=False,
        split_date=split_date,
        building_ids=building_ids
    )

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    g = torch.Generator()
    g.manual_seed(seed)

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


def run_many(
    context_length, pred_window, encoder_hidden, encoder_in_dim, encoder_emb_dim, model_hidden, learning_rate, batch_size, epochs, 
    sigma_eps, grad_clip_norm, patience, min_delta, train_dataloader, val_dataloader, encoder, model, optimizer, writer, 
    metrics_logger, checkpoint_dir, metrics_csv_path, plots_dir, feature_names, resume_checkpoint=None,
    ):

    # ===== HYPERPARAMETER LOGGING =====
    hparams = {
        'context_length': context_length,
        'pred_window': pred_window,
        'encoder_hidden': encoder_hidden,
        'encoder_in_dim': encoder_in_dim,
        'encoder_emb_dim': encoder_emb_dim,
        'model_hidden': model_hidden,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs,
        'sigma_eps': sigma_eps,
        'grad_clip_norm': grad_clip_norm,
        'optimizer': optimizer,
        'early_stopping_patience': patience,
        'early_stopping_min_delta': min_delta
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
        epoch_number, best_vloss = load_checkpoint(resume_checkpoint, model, encoder, optimizer)

    try:
        time_indices = [
            feature_names.index('hour_sin'),
            feature_names.index('hour_cos'),
            feature_names.index('day_of_week_sin'),
            feature_names.index('day_of_week_cos')
        ]
    except ValueError:
        print("[WARNING] Time features missing from dataset. X-axis will default to integers.")
        time_indices = None

    EPOCHS = epochs

    train_loss_history = []
    val_loss_history = []

    for epoch in range(epoch_number, EPOCHS):
        epoch_start_time = time.time()

        # Training
        model.train(True)
        avg_loss = train_one_epoch(epoch, writer, train_dataloader, optimizer, model, encoder, EPOCHS)
        train_loss_history.append(avg_loss)

        # Validation
        running_vloss = 0.0
        running_mae = 0.0
        running_rmse = 0.0
        running_crps = 0.0

        model.eval()

        last_val_mu = None
        last_val_true = None

        pbar_val = tqdm(enumerate(val_dataloader), total=len(val_dataloader),
                        desc=f'Epoch {epoch + 1}/{EPOCHS} [VAL]', leave=False)

        with torch.no_grad():
            for i, vdata in pbar_val:
                vx_context, vx_future, vy_context, vy_future, bldg_mean, bldg_std, bldg_names = vdata
                vx_context = vx_context.to(device)
                vx_future = vx_future.to(device)
                vy_context = vy_context.to(device)
                vy_future = vy_future.to(device)

                bldg_mean = bldg_mean.to(device).view(-1, 1, 1)
                bldg_std = bldg_std.to(device).view(-1, 1, 1)

                vemb_context = encoder(vx_context)
                vemb_future = encoder(vx_future)

                vmu, vsg = model(vemb_context, vy_context, vemb_future, y_future=None)
                
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
                
                # Save last batch for plotting
                last_val_mu = true_vmu.cpu().numpy()
                last_val_true = true_vy_future.cpu().numpy()
                last_val_sg = true_vsg.cpu().numpy() 

                if time_indices is not None:
                    last_val_time_features = vx_future[:, :, time_indices].cpu().numpy()

                num_batches = i + 1

                last_val_bldg_names = bldg_names
                
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

        print(f"Loss: {avg_loss:.4f} | Val Loss: {avg_vloss:.4f} | MAE: {avg_mae:.4f} | RMSE: {avg_rmse:.4f} | CRPS: {avg_crps:.4f}")

        # Save periodic checkpoint
        periodic_checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
        save_checkpoint(periodic_checkpoint_path, model, encoder, optimizer, epoch, best_vloss)

        timestamp = datetime.now()

        # Early stopping logic
        if avg_vloss < best_vloss - min_delta:
            best_vloss = avg_vloss
            epochs_no_improve = 0
            best_model_path = os.path.join(checkpoint_dir, f'best_model_{timestamp}.pth')
            save_checkpoint(best_model_path, model, encoder, optimizer, epoch, best_vloss)
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
            time_features=last_val_time_features, # <-- NEW
            hours=24, 
            save_path=pred_plot_path
        )
    

def get_encoder_input_dim(dataset):
    if len(dataset) == 0:
        raise ValueError("Train dataset is empty; cannot infer encoder input dimension")
    x_context, *_= dataset[0]
    return x_context.shape[-1]

def hyperparameter_tune(train_dataset, val_dataset, input_dim=45, n_trials=20, n_epochs=3):
    """
    Hyperparameter tuning using Optuna for DeepAR model.
    
    Args:
        train_dl: Training dataloader
        val_dl: Validation dataloader
        n_trials: Number of Optuna trials (default 20)
        n_epochs: Number of epochs per trial (default 3)
    
    Returns:
        Dictionary with best hyperparameters
    """
    
    def objective(trial):
        """Objective function for Optuna optimization."""
        
        # Suggest hyperparameters
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

        sample_batch = next(iter(train_dl))
        x_context_sample = sample_batch[0] 
        input_dim = x_context_sample.shape[-1]
        
        # Create fresh encoder and model for this trial
        trial_encoder = CentralEncoder(input_dim, encoder_hidden, encoder_emb_dim)
        trial_encoder = trial_encoder.to(device)
        
        trial_model = DeepARAutoreg(
            encoder_emb_dim,
            model_hidden,
            pred_len=24,
            sigma_eps=sigma_eps,
        )
        trial_model = trial_model.to(device)
        
        optimizer = torch.optim.Adam(list(trial_model.parameters()) + list(trial_encoder.parameters()), lr=learning_rate)
        
        # Train for limited epochs
        best_val_loss = float('inf')
        
        with tqdm(range(n_epochs), desc=f'Trial {trial.number + 1}/{n_trials}', leave=False) as epoch_pbar:
            for epoch in epoch_pbar:
                tf_ratio = max(0.0, 1.0 - (epoch / n_epochs))
                # Training phase
                trial_model.train(True)
                running_loss = 0.0
                
                for i, data in enumerate(train_dl):
                    x_context, x_future, y_context, y_future, _, _ = data
                    x_context = x_context.to(device)
                    x_future = x_future.to(device)
                    y_context = y_context.to(device)
                    y_future = y_future.to(device)
                    
                    emb_context = trial_encoder(x_context)
                    emb_future = trial_encoder(x_future)
                    
                    optimizer.zero_grad()
                    mu, sg = trial_model(emb_context, y_context, emb_future, y_future, tf_ratio)
                    loss = gaussian_nll_loss(mu, y_future, sg)
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(list(trial_model.parameters()) + list(trial_encoder.parameters()), max_norm=grad_clip)
                    optimizer.step()
                    running_loss += loss.item()
                
                # Validation phase
                trial_model.eval()
                running_vloss = 0.0
                
                with torch.no_grad():
                    for i, vdata in enumerate(val_dl):
                        vx_context, vx_future, vy_context, vy_future, _, _ = vdata
                        vx_context = vx_context.to(device)
                        vx_future = vx_future.to(device)
                        vy_context = vy_context.to(device)
                        vy_future = vy_future.to(device)
                        
                        vemb_context = trial_encoder(vx_context)
                        vemb_future = trial_encoder(vx_future)
                        
                        vmu, vsg = trial_model(vemb_context, vy_context, vemb_future, y_future=None)
                        vloss = gaussian_nll_loss(vmu, vy_future, vsg)
                        running_vloss += vloss.item()
                
                avg_vloss = running_vloss / (i + 1)
                best_val_loss = min(best_val_loss, avg_vloss)
                
                # Report intermediate value for pruning
                trial.report(avg_vloss, epoch)
                
                epoch_pbar.set_postfix({'val_loss': f'{avg_vloss:.4f}', 'best': f'{best_val_loss:.4f}'})
                
                # Prune if not promising
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
            del train_dl, val_dl

        return best_val_loss
    
    # Create study and optimize
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner()
    )
    
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER TUNING WITH OPTUNA")
    print(f"Trials: {n_trials} | Epochs per trial: {n_epochs}")
    print(f"{'='*60}")
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Get best trial
    best_trial = study.best_trial
    
    print(f"\n{'='*60}")
    print(f"TUNING COMPLETE")
    print(f"Best validation loss: {best_trial.value:.6f}")
    print(f"\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key:<20}: {value}")
    print(f"{'='*60}\n")
    
    trials_df = study.trials_dataframe()
    trials_df.to_csv('optuna_trials_history.csv', index=False)
    print(f"[INFO] Saved complete Optuna trial history to 'optuna_trials_history.csv'")

    return best_trial.params


def main(tune=False):
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
    
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer used for gradient optimization')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed to set randomness')
    
    args = parser.parse_args()

    seed_everything(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_dataset, val_dataset, train_dataloader, val_dataloader = build_dataloaders(
        data_file=args.data_file,
        num_buildings=args.num_buildings,
        context=args.context,
        pred_window=args.pred_window,
        target_col=args.target_col,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_date=args.split_date
    )

    if args.tune == True: # (Consider changing this to use an argparse flag like args.tune)
            best_params = hyperparameter_tune(train_dataset, val_dataset)
            if best_params:
                # Sync hyperparameters
                args.batch_size = best_params.get('batch_size', args.batch_size)
                args.learning_rate = best_params.get('learning_rate', args.learning_rate)
                args.encoder_hidden = best_params.get('encoder_hidden', args.encoder_hidden)
                args.emb_dim = best_params.get('encoder_emb_dim', args.emb_dim)
                args.sigma_eps = best_params.get('sigma_eps', args.sigma_eps)
                args.grad_clip = best_params.get('grad_clip', args.grad_clip)

                print(f"\n[INFO] Using best hyperparameters from tuning...")
                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
                val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    
    encoder_input_dim = get_encoder_input_dim(train_dataset)
    encoder = CentralEncoder(encoder_input_dim, args.encoder_hidden, args.emb_dim)
    model = DeepARAutoreg(args.emb_dim, args.model_hidden, pred_len=args.pred_window, sigma_eps=args.sigma_eps)
    encoder = encoder.to(device)
    model = model.to(device)
    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(encoder.parameters()), lr=args.learning_rate)
    else:
        optimizer = torch.optim.Adam(list(model.parameters()) + list(encoder.parameters()), lr=args.learning_rate)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(os.path.join(args.output_dir, f'deepar_trainer_{timestamp}'))
    metrics_log_file = os.path.join(args.output_dir, f'metrics_{timestamp}.csv')
    metrics_logger = MetricsLogger(metrics_log_file)

    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    feature_names = train_dataset.feature_names

    run_many(
        args.context, args.pred_window, args.encoder_hidden, encoder_input_dim, args.emb_dim, args.model_hidden,
        args.learning_rate, args.batch_size, args.epochs, args.sigma_eps, args.grad_clip, args.patience, args.min_delta, 
        train_dataloader, val_dataloader, encoder, model, optimizer, writer, metrics_logger, args.checkpoint_dir, metrics_log_file,
        plots_dir, feature_names, resume_checkpoint=None,
    )


if __name__ == '__main__':
    main()
    