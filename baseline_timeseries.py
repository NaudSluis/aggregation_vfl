import argparse
import os
import random
import warnings
from datetime import datetime
import traceback
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import DLinear
from pytorch_lightning.loggers import CSVLogger
import glob

from plots_metric_utils import compute_metrics, plot_training_validation_loss, plot_validation_metrics, plot_daily_forecast, plot_predicted_vs_actual

warnings.filterwarnings('ignore')

DYNAMIC_FEATURES = [
    'water', 'steam', 'irrigation', 'gas', 'hotwater', 'chilledwater', 'solar',
    'airTemperature', 'cloudCoverage', 'dewTemperature', 'precipDepth1HR',
    'precipDepth6HR', 'seaLvlPressure', 'windDirection', 'windSpeed',
    'hour', 'day_of_week', 'month', 'days_from_start'
]

STATIC_CATEGORICAL = [
    'primaryspaceusage', 'sub_primaryspaceusage', 'timezone',
    'industry', 'subindustry', 'heatingtype', 'leed_level', 'rating',
    'electricity_meta', 'hotwater_meta', 'chilledwater_meta', 'steam_meta',
    'water_meta', 'irrigation_meta', 'solar_meta', 'gas_meta'
]

STATIC_NUMERIC = [
    'sqm', 'yearbuilt', 'numberoffloors', 'occupants',
    'energystarscore', 'eui', 'site_eui', 'source_eui'
]

def load_merged_dataset(
    dataset_path,
    target_col='electricity',
    include_static_features=True,
    num_buildings=None,
    random_seed=42
):
    """Load a single merged parquet table and prepare it for NeuralForecast."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Merged parquet file not found: {dataset_path}")

    df = pd.read_parquet(dataset_path)

    if 'timestamp' not in df.columns and 'ds' not in df.columns:
        raise ValueError("Merged parquet must contain a timestamp column named 'timestamp' or 'ds'.")

    if 'timestamp' in df.columns:
        df['ds'] = pd.to_datetime(df['timestamp'])
    else:
        df['ds'] = pd.to_datetime(df['ds'])

    if 'building' in df.columns:
        df['unique_id'] = df['building'].astype(str)
    elif 'unique_id' not in df.columns:
        raise ValueError("Merged parquet must contain either a 'building' or 'unique_id' column.")

    df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)

    if num_buildings is not None and num_buildings > 0:
        unique_ids = df['unique_id'].unique().tolist()
        if num_buildings < len(unique_ids):
            random.seed(random_seed)
            selected_ids = random.sample(unique_ids, num_buildings)
            df = df[df['unique_id'].isin(selected_ids)].copy()

    # Keep only requested features and the target
    keep_cols = ['unique_id', 'ds', target_col]
    keep_cols += [col for col in DYNAMIC_FEATURES if col in df.columns and col != target_col]
    if include_static_features:
        keep_cols += [col for col in STATIC_CATEGORICAL + STATIC_NUMERIC if col in df.columns]
    keep_cols = list(dict.fromkeys(keep_cols))

    missing = [c for c in ['unique_id', 'ds', target_col] if c not in keep_cols]
    if missing:
        raise ValueError(f"Required columns missing from merged dataset: {missing}")

    df = df[keep_cols].copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'unique_id']

    df[numeric_cols] = df.groupby('unique_id')[numeric_cols].transform(lambda group: group.ffill().bfill())
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0.0)
    df[numeric_cols] = df[numeric_cols].fillna(0.0)
    df = df[df[target_col].notna()].copy()

    print(f"[INFO] Loaded merged dataset: {dataset_path}")
    print(f"[INFO] Shape after feature selection: {df.shape}")
    print(f"[INFO] Buildings: {df['unique_id'].nunique()}")
    print(f"[INFO] Time range: {df['ds'].min()} to {df['ds'].max()}")

    return df


def split_train_test(df, split_date='2017-06-01'):
    split_dt = pd.to_datetime(split_date)
    train_df = df[df['ds'] < split_dt].copy()
    test_df = df[df['ds'] >= split_dt].copy()

    print(f"[INFO] Training rows: {len(train_df)}, Test rows: {len(test_df)}")
    print(f"[INFO] Training buildings: {train_df['unique_id'].nunique()}, Test buildings: {test_df['unique_id'].nunique()}")

    return train_df, test_df


def scale_target(train_df, test_df, target_col='electricity'):
    mean = train_df[target_col].mean()
    std = train_df[target_col].std()
    std = std if std != 0 else 1.0

    train_df[target_col] = (train_df[target_col] - mean) / std
    test_df[target_col] = (test_df[target_col] - mean) / std

    return train_df, test_df, mean, std


def build_dlinear_input(df, target_col='electricity', include_static_features=True):
    cols = ['unique_id', 'ds', target_col]
    cols += [col for col in DYNAMIC_FEATURES if col in df.columns and col != target_col]
    if include_static_features:
        cols += [col for col in STATIC_CATEGORICAL + STATIC_NUMERIC if col in df.columns]
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()


def build_dlinear_model(
    h=24,
    input_size=168,
    learning_rate=1e-4,
    batch_size=128,
    max_steps=1000,
    early_stop_patience_steps=5,
    logger_dir='nf_logs',
    logger_name='DLinear_Model'
):
    logger = CSVLogger(save_dir=logger_dir, name=logger_name)
    
    # We do NOT pass exog_lists here. DLinear will solely use the target_col ('electricity').
    model = DLinear(
        h=h,
        input_size=input_size,
        max_steps=max_steps,
        early_stop_patience_steps=early_stop_patience_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        logger=logger
    )
    return model, logger


def evaluate_predictions(pred_df, truth_df, target_col='electricity'):
    if 'unique_id' not in pred_df.columns or 'ds' not in pred_df.columns:
        raise ValueError("Predictions must include 'unique_id' and 'ds' columns.")

    pred_cols = [c for c in pred_df.columns if c not in ('unique_id', 'ds')]
    if len(pred_cols) == 0:
        raise ValueError('No prediction column found in prediction dataframe.')

    pred_col = pred_cols[0]
    merged = pred_df.merge(
        truth_df[['unique_id', 'ds', target_col]].rename(columns={target_col: 'y_true'}),
        on=['unique_id', 'ds'],
        how='inner'
    )

    if merged.empty:
        raise ValueError('No matching rows found between predictions and truth data.')

    metrics = compute_metrics(merged['y_true'].to_numpy(), merged[pred_col].to_numpy())
    return merged, metrics


def plot_baseline_first_24h(cv_predictions, output_dir, save_path=None):
    if cv_predictions.empty:
        print('[WARNING] No baseline predictions available for first 24h plot.')
        return

    save_path = save_path or os.path.join(output_dir, 'baseline_predicted_vs_actual_24h.png')
    ordered = cv_predictions.sort_values(['unique_id', 'ds'])
    first_id = ordered['unique_id'].iloc[0]
    subset = ordered[ordered['unique_id'] == first_id].head(24)

    if subset.empty:
        print('[WARNING] Not enough prediction data for first 24h plot.')
        return

    plot_predicted_vs_actual(
        subset['y'].to_numpy(),
        subset['DLinear'].to_numpy(),
        hours=24,
        title=f'Baseline First 24h Actual vs Predicted ({first_id})',
        save_path=save_path
    )


def train_dlinear(
    dataset_path,
    output_dir,
    split_date='2017-06-01',
    target_col='electricity',
    include_static_features=True,
    num_buildings=None,
    context=168,
    pred_window=24,
    batch_size=128,
    learning_rate=1e-4,
    max_steps=1000,
    early_stop_patience_steps=5
):
    os.makedirs(output_dir, exist_ok=True)
    df = load_merged_dataset(
        dataset_path=dataset_path,
        target_col=target_col,
        include_static_features=include_static_features,
        num_buildings=num_buildings
    )

    train_df, test_df = split_train_test(df, split_date=split_date)
    train_df, test_df, mean, std = scale_target(train_df, test_df, target_col=target_col)

    train_input = build_dlinear_input(train_df, target_col=target_col, include_static_features=include_static_features)
    test_input = build_dlinear_input(test_df, target_col=target_col, include_static_features=include_static_features)

    model, _ = build_dlinear_model(
        h=pred_window, input_size=context, learning_rate=learning_rate, batch_size=batch_size,
        max_steps=max_steps, early_stop_patience_steps=early_stop_patience_steps,
        logger_dir=output_dir, logger_name='DLinear_Model'
    )

    nf = NeuralForecast(models=[model], freq='h')
    
    # Combine train and test for the rolling evaluation
    full_df = pd.concat([train_input, test_input])
    
    # Calculate how many rolling windows fit in your test set
    # Note: step_size=24 means it evaluates every day. If DeepAR shifted by 1 hour, set step_size=1
    # (Warning: step_size=1 on a large dataset in NeuralForecast can take a very long time)
    n_windows = len(test_input['ds'].unique()) - pred_window + 1

    print(f"[INFO] Running cross-validation over {n_windows} windows to match DeepAR test phase...")
    cv_predictions = nf.cross_validation(
        df=full_df.rename(columns={target_col: 'y'}),
        n_windows=n_windows,
        step_size=1,
        val_size=pred_window
    )

    # Unscale everything
    cv_predictions['y'] = cv_predictions['y'] * std + mean
    cv_predictions['DLinear'] = cv_predictions['DLinear'] * std + mean

    # Calculate metrics
    metrics = compute_metrics(cv_predictions['y'].to_numpy(), cv_predictions['DLinear'].to_numpy())
    
    metrics_path = os.path.join(output_dir, 'baseline_metrics.csv')
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"[INFO] Saved metrics to {metrics_path}")

    # Use cv_predictions instead of predictions
    pred_path = os.path.join(output_dir, 'baseline_predictions.parquet')
    cv_predictions.to_parquet(pred_path, index=False)
    print(f"[INFO] Saved predictions to {pred_path}")

    # Find the PyTorch Lightning training logs instead of the final test metrics
    pl_log_paths = glob.glob(os.path.join(output_dir, 'DLinear_Model', '**', 'metrics.csv'), recursive=True)
    
    if pl_log_paths:
        # Sort to grab the latest version folder if there are multiple
        latest_log_path = sorted(pl_log_paths)[-1]
        try:
            plot_validation_metrics(latest_log_path, save_path=os.path.join(output_dir, 'validation_metrics.png'))
            print(f"[INFO] Saved validation plot to {os.path.join(output_dir, 'validation_metrics.png')}")
        except Exception as e:
            print(f"[WARNING] Could not plot validation metrics: {e}")
    else:
        print("[WARNING] Could not find PyTorch Lightning metrics.csv to plot training history.")

    # cv_predictions already has 'y' and 'DLinear', so we don't need to rename 'y_true'/'y_pred' anymore
    if not cv_predictions.empty:
        plot_daily_forecast(cv_predictions[['unique_id', 'ds', 'y', 'DLinear']].tail(pred_window))
        plot_baseline_first_24h(cv_predictions, output_dir)

    # Return cv_predictions instead of merged
    return cv_predictions, metrics


def tune_dlinear(
    dataset_path, output_dir, split_date='2017-06-01', target_col='electricity',
    include_static_features=True, num_buildings=None, context=168, pred_window=24,
    n_trials=20, random_seed=42
):
    os.makedirs(output_dir, exist_ok=True)
    df = load_merged_dataset(dataset_path, target_col, include_static_features, num_buildings, random_seed)

    train_df, _ = split_train_test(df, split_date=split_date)
    train_df, _, mean, std = scale_target(train_df, train_df.copy(), target_col=target_col)
    
    def objective(trial):
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        max_steps = trial.suggest_int('max_steps', 500, 5000, step=500)

        # Look here: removed hist_exog_list and stat_exog_list
        # Force early stopping to 0 to disable the callback during CV
        model, _ = build_dlinear_model(
            h=pred_window, 
            input_size=context, 
            learning_rate=learning_rate, 
            batch_size=batch_size,
            max_steps=max_steps, 
            early_stop_patience_steps=0, # <--- EXPLICITLY SET TO 0
            logger_dir=output_dir, 
            logger_name=f'DLinear_Optuna_{trial.number}'
        )

        nf = NeuralForecast(models=[model], freq='h')
        minimal_df = train_df[['unique_id', 'ds', target_col]].rename(columns={target_col: 'y'})
        # Cross validation evaluates over the last few windows properly
        try:
            cv_df = nf.cross_validation(
                df=minimal_df.rename(columns={target_col: 'y'}), 
                n_windows=3, 
                step_size=24
            )
        except Exception as e:
            print(f"\n[OPTUNA CRASH] Trial {trial.number} failed!")
            traceback.print_exc()  # This will print the exact line and cause of the crash
            return float('inf')

        # Unscale for accurate MAE reporting in Optuna
        cv_df['y'] = cv_df['y'] * std + mean
        cv_df['DLinear'] = cv_df['DLinear'] * std + mean

        metrics = compute_metrics(cv_df['y'].to_numpy(), cv_df['DLinear'].to_numpy())
        return float(metrics['mae'])

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=random_seed))
    study.optimize(objective, n_trials=n_trials)

    results_df = study.trials_dataframe()
    results_df.to_csv(os.path.join(output_dir, 'dlinear_optuna_study.csv'), index=False)
    
    print(f"[INFO] Best Optuna MAE: {study.best_value:.6f}")
    return study.best_params, study.best_value

def main():
    parser = argparse.ArgumentParser(description='Train or tune a DLinear baseline model on a merged parquet dataset.')
    parser.add_argument('--mode', type=str, choices=['train', 'tune'], default='train', help='Run mode: train or tune')
    parser.add_argument('--dataset-path', type=str, default='final_lstm_dataset_merged.parquet', help='Path to the merged parquet dataset')
    parser.add_argument('--output-dir', type=str, default='baseline_runs', help='Directory to save metrics, predictions, and logs')
    parser.add_argument('--split-date', type=str, default='2017-06-01', help='Train/test split boundary')
    parser.add_argument('--target-col', type=str, default='electricity', help='Target column name')
    parser.add_argument('--include-static-features', action='store_true', help='Include static metadata features if available')
    parser.add_argument('--num-buildings', type=int, default=None, help='Sample this many buildings from the merged dataset')
    parser.add_argument('--context', type=int, default=168, help='Input context window length')
    parser.add_argument('--pred-window', type=int, default=24, help='Prediction horizon')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for DLinear training')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate for DLinear')
    parser.add_argument('--max-steps', type=int, default=1000, help='Maximum training steps for DLinear')
    parser.add_argument('--early-stop-patience-steps', type=int, default=5, help='Early stopping patience for DLinear')
    parser.add_argument('--n-trials', type=int, default=20, help='Number of Optuna trials when tuning')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == 'train':
        merged, metrics = train_dlinear(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            split_date=args.split_date,
            target_col=args.target_col,
            include_static_features=args.include_static_features,
            num_buildings=args.num_buildings,
            context=args.context,
            pred_window=args.pred_window,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            early_stop_patience_steps=args.early_stop_patience_steps
        )
        print(f"[INFO] Training completed. Metrics: {metrics}")
    else:
        best_params, best_value = tune_dlinear(
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            split_date=args.split_date,
            target_col=args.target_col,
            include_static_features=args.include_static_features,
            num_buildings=args.num_buildings,
            context=args.context,
            pred_window=args.pred_window,
            n_trials=args.n_trials,
            random_seed=args.random_seed
        )
        print(f"[INFO] Optuna tuning completed. Best MAE: {best_value:.6f}")
        print(best_params)


if __name__ == '__main__':
    main()
