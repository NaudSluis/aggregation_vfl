import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def CRPS():
    # Placeholder for CRPS calculation if needed later.
    # For DLinear baseline we avoid computing CRPS by default.
    return None

def compute_metrics(y_true, y_pred, include_nd=False, include_crps=False):

    """
    Compute basic error metrics between `y_true` and `y_pred`.

    By default this returns only `mae` and `rmse`. Set `include_nd=True`
    to also compute normalized deviation (ND), and `include_crps=True` to
    compute CRPS (not computed for DLinear by default).
    """

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    metrics = {}

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true has {len(y_true)} samples, y_pred has {len(y_pred)}")

    # Basic metrics
    metrics['mae'] = mae(y_true, y_pred)
    metrics['rmse'] = rmse(y_true, y_pred)

    # Optional metrics
    if include_nd:
        denom = np.sum(np.abs(y_true)) + 1e-9
        metrics['nd'] = float(np.sum(np.abs(y_true - y_pred)) / denom)

    if include_crps:
        # CRPS requires probabilistic forecasts; keep as None by default
        metrics['crps'] = CRPS()
    # if 'crps' in metric_names:
    #     residual_std = np.std(residuals)
    #     if residual_std > 0:
    #         z_scores = residuals / residual_std
    #         crps_vals = residual_std * (z_scores * (2 * norm.cdf(z_scores) - 1) + 
    #                                         2 * norm.pdf(z_scores) - 1/np.sqrt(np.pi))
    #         crps = np.mean(np.abs(crps_vals))
    #     else:
    #         crps = 0.0
    #     metrics['crps'] = crps
    
    return metrics

def plot_losses(version):
    # Load the metrics from the specific version
    metrics_df = pd.read_csv(f"/content/nf_logs/DLinear_Model/version_{version}/metrics.csv")

    epoch_metrics = metrics_df.groupby('epoch').mean()

    plt.figure(figsize=(10, 6))

    if 'train_loss_step' in epoch_metrics.columns:
        plt.plot(epoch_metrics.index, epoch_metrics['train_loss_step'],
                 label='Train Loss', color='tab:blue', alpha=0.8)

    if 'valid_loss' in epoch_metrics.columns:
        valid_data = epoch_metrics['valid_loss'].dropna()
        plt.plot(valid_data.index, valid_data,
                 label='Validation Loss', marker='o', color='tab:orange', linewidth=2)

    plt.title('Training and Validation MAE (Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MAE)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()

    plt.show()
    
def plot_daily_forecast(df, title="Daily Energy Forecast: Actual vs Predicted"):

    x = df['ds']
    y_true = df['y']
    y_pred = df['DLinear']

    plt.figure(figsize=(10, 5))

    plt.plot(x, y_true, label='Actual (kWh)',
             color='blue', linewidth=2, marker='o', markersize=4)

    plt.plot(x, y_pred, label='Predicted (kWh)',
             color='orange', linewidth=2, linestyle='--', marker='x', markersize=4)

    plt.title(title, fontsize=14)
    plt.ylabel("Energy (kWh)", fontsize=12)
    plt.xlabel("Time of Day", fontsize=12)

    plt.xticks(rotation=45)

    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()

    plt.show()


def plot_training_validation_loss(metrics_csv_path, save_path=None):
    """
    Plot training and validation loss over epochs from metrics CSV file.
    
    Args:
        metrics_csv_path: Path to the metrics CSV file generated during training
        save_path: Optional path where to save the generated figure
    """
    metrics_df = pd.read_csv(metrics_csv_path)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(metrics_df['epoch'], metrics_df['train_loss'].astype(float),
             label='Training Loss', color='tab:blue', linewidth=2, marker='o', markersize=4)
    
    plt.plot(metrics_df['epoch'], metrics_df['val_loss'].astype(float),
             label='Validation Loss', color='tab:orange', linewidth=2, marker='o', markersize=4)
    
    plt.title('Training and Validation Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"[INFO] Saved loss plot to {save_path}")

    plt.show()
    plt.close()

def plot_validation_metrics(metrics_csv_path, save_path=None):
    """
    Plot validation metrics (MAE, RMSE, CRPS) over epochs from metrics CSV file.
    
    Args:
        metrics_csv_path: Path to the metrics CSV file generated during training
        save_path: Optional path where to save the generated figure
    """
    metrics_df = pd.read_csv(metrics_csv_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # MAE
    axes[0, 0].plot(metrics_df['epoch'], metrics_df['val_mae'].astype(float),
                    color='tab:blue', linewidth=2, marker='o', markersize=4)
    axes[0, 0].set_title('Mean Absolute Error (MAE)', fontsize=12)
    axes[0, 0].set_xlabel('Epoch', fontsize=10)
    axes[0, 0].set_ylabel('MAE', fontsize=10)
    axes[0, 0].grid(True, linestyle=':', alpha=0.7)
    
    # RMSE
    axes[0, 1].plot(metrics_df['epoch'], metrics_df['val_rmse'].astype(float),
                    color='tab:orange', linewidth=2, marker='o', markersize=4)
    axes[0, 1].set_title('Root Mean Squared Error (RMSE)', fontsize=12)
    axes[0, 1].set_xlabel('Epoch', fontsize=10)
    axes[0, 1].set_ylabel('RMSE', fontsize=10)
    axes[0, 1].grid(True, linestyle=':', alpha=0.7)
    
    # CRPS
    axes[1, 1].plot(metrics_df['epoch'], metrics_df['val_crps'].astype(float),
                    color='tab:red', linewidth=2, marker='o', markersize=4)
    axes[1, 1].set_title('Continuous Ranked Probability Score (CRPS)', fontsize=12)
    axes[1, 1].set_xlabel('Epoch', fontsize=10)
    axes[1, 1].set_ylabel('CRPS', fontsize=10)
    axes[1, 1].grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"[INFO] Saved validation metrics plot to {save_path}")

    plt.show()
    plt.close()


import matplotlib.pyplot as plt
import numpy as np

def _decode_time_features(time_features, hours_to_plot):
    """
    Reverse-engineers the hour and day of the week from sine/cosine features.
    Expects time_features shape: (time_steps, 4) -> [hour_sin, hour_cos, day_sin, day_cos]
    """
    hour_sin, hour_cos = time_features[:hours_to_plot, 0], time_features[:hours_to_plot, 1]
    day_sin, day_cos = time_features[:hours_to_plot, 2], time_features[:hours_to_plot, 3]

    # Convert sin/cos back to 0-23 (hours) and 0-6 (days)
    hours_decoded = np.round((np.arctan2(hour_sin, hour_cos) / (2 * np.pi)) * 24) % 24
    days_decoded = np.round((np.arctan2(day_sin, day_cos) / (2 * np.pi)) * 7) % 7

    # Pandas default dt.dayofweek: 0=Monday, 6=Sunday
    day_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}

    labels = []
    for d, h in zip(days_decoded, hours_decoded):
        labels.append(f"{day_names[int(d)]} {int(h):02d}:00")

    return labels

def plot_two_buildings_with_ci(actuals, predicted_mus, predicted_sigmas, building_names, building_usages, 
                               time_features=None, hours=24, save_path=None):
    """
    Plots actual vs predicted values. Dynamically scales to 1 or 2 distinct buildings.
    If time_features is provided, replaces the generic x-axis with Day/Hour labels.
    """
    time_steps = np.arange(hours)
    z_score = 1.96

    actuals_np = np.asarray(actuals)
    mus_np = np.asarray(predicted_mus)
    sigmas_np = np.asarray(predicted_sigmas)

    # 1. Search the batch for up to two DISTINCT building indices
    indices_to_plot = [0] 
    for i in range(1, len(building_names)):
        if building_names[i] != building_names[0]:
            indices_to_plot.append(i)
            break 
            
    num_plots = len(indices_to_plot)
    fig, axes = plt.subplots(1, num_plots, figsize=(10 * num_plots, 7), dpi=300)
    
    if num_plots == 1:
        axes = [axes]

    for i, batch_idx in enumerate(indices_to_plot): 
        actual = actuals_np[batch_idx, :hours].squeeze()
        mu = mus_np[batch_idx, :hours].squeeze()
        sigma = sigmas_np[batch_idx, :hours].squeeze()

        lower_bound = mu - (z_score * sigma)
        upper_bound = mu + (z_score * sigma)

        ax = axes[i]
        ax.plot(time_steps, actual, label='Actual', marker='o', markersize=4, linestyle='-', color='#1f77b4')
        ax.plot(time_steps, mu, label='Predicted Mean', marker='x', markersize=4, linestyle='--', color='#ff7f0e')

        ax.fill_between(
            time_steps, lower_bound, upper_bound,
            color='#ff7f0e', alpha=0.2, label='95% CI', edgecolor='none'
        )

        ax.set_title(f"Building: {building_names[batch_idx]}", fontsize=13, fontweight='bold', pad=10)
        
        usage = building_usages[batch_idx] if batch_idx < len(building_usages) else "Unknown"
        ax.set_xlabel(f"Primary Usage: {usage}", fontsize=11, labelpad=8)
        
        # --- NEW: Time Decoding and Custom X-Axis ---
        ax.set_xticks(time_steps[::4]) # Place a tick every 4 hours to avoid overlapping text
        
        if time_features is not None:
            # Decode time for this specific batch item
            x_labels = _decode_time_features(time_features[batch_idx], hours)
            ax.set_xticklabels([x_labels[t] for t in range(0, hours, 4)], rotation=45, ha='right')
        else:
            ax.set_xticklabels(time_steps[::4])
        # --------------------------------------------

        ax.set_ylabel('Electricity Consumption (kWh)', fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right')

    fig.suptitle("Vertical Federated Learning: 24-Hour Forecast Horizon Comparison", 
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.15) # Adjusted bottom to fit angled text

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"[INFO] Saved subplot to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_four_models_comparison(actual, models_predictions, model_names, building_name, building_usage, hours=24, save_path=None):
    """
    Plots the actual values against 4 different model predictions.
    models_predictions: List of 4 1D arrays [pred_model1, pred_model2, pred_model3, pred_model4]
    model_names: List of 4 strings for the legend
    """
    plt.figure(figsize=(14, 7), dpi=300)
    time_steps = np.arange(hours)

    # Plot actual (Thick black line)
    plt.plot(time_steps, actual[:hours], label='Actual', marker='o', markersize=6, linestyle='-', color='black', linewidth=2)

    # Distinct colors and markers for 4 models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['x', '^', 's', 'd']

    for i in range(len(models_predictions)):
        pred = np.asarray(models_predictions[i]).flatten()[:hours]
        plt.plot(time_steps, pred, label=model_names[i],
                 marker=markers[i], markersize=5, linestyle='--', color=colors[i])

    # Title and Subtitle
    plt.title(f"Model Comparison - Building: {building_name}", fontsize=15, fontweight='bold')
    plt.text(0.5, 1.01, f"Primary Usage: {building_usage}", 
            horizontalalignment='center', verticalalignment='bottom', 
            transform=plt.gca().transAxes, fontsize=12, color='gray')

    plt.xlabel('Time step (Hours)')
    plt.ylabel('Electricity Consumption (kWh)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
