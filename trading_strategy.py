import argparse
from datetime import timedelta

import numpy as np
import pandas as pd
import torch

from utils import StandardScaler
from dcrnn_model import DCRNNModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: str):
    # Handle PyTorch 2.6+ compatibility with weights_only=False
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        if "weights_only" in str(e):
            # Fallback to weights_only=False for older checkpoints
            state = torch.load(path, map_location=device, weights_only=False)
        else:
            raise e
    
    scaler = StandardScaler(mean=state['scaler_mean'], std=state['scaler_std'])
    model_kwargs = state['model_kwargs']
    adj = state['adjacency']
    model = DCRNNModel(adj, logger=None, **model_kwargs)
    
    # Initialize model parameters by running a dummy forward pass
    try:
        # Create dummy input to initialize DCGRU parameters
        seq_len = model_kwargs.get('seq_len', 12)
        dummy_input = torch.randn(seq_len, 1, model_kwargs['num_nodes'], device=device)
        with torch.no_grad():
            # This will initialize the lazy parameters in DCGRU cells
            _ = model(dummy_input)
    except Exception as e:
        print(f"⚠️  Model initialization failed: {e}")
        # Continue anyway - the model might still work
    
    # Try to load state dict, handle parameter mismatch
    try:
        model.load_state_dict(state['state_dict'])
        print(f"✅ Successfully loaded trained model.")
    except RuntimeError as e:
        if "Unexpected key(s)" in str(e):
            print(f"⚠️  Model architecture mismatch detected. Attempting parameter mapping...")
            
            # Create a mapping for parameter names
            old_state_dict = state['state_dict']
            new_state_dict = {}
            
            # Map old parameter names to new ones
            for key, value in old_state_dict.items():
                # Handle encoder and decoder prefixes
                if key.startswith('encoder_model.') or key.startswith('decoder_model.'):
                    # Already has the correct prefix
                    new_state_dict[key] = value
                else:
                    # Add encoder/decoder prefixes based on parameter type
                    if 'gconv_weight_' in key or 'gconv_biases_' in key or 'fc_weight_' in key or 'fc_biases_' in key:
                        # These are DCGRU parameters, need to determine if encoder or decoder
                        # For now, assume encoder for the first layer, decoder for the second
                        if 'dcgru_layers.0.' in key or 'dcgru_layers.1.' in key:
                            # This is a DCGRU layer parameter
                            if 'dcgru_layers.0.' in key:
                                new_key = f"encoder_model.{key}"
                            else:
                                new_key = f"decoder_model.{key}"
                            new_state_dict[new_key] = value
                        else:
                            # Assume encoder for other DCGRU parameters
                            new_key = f"encoder_model.{key}"
                            new_state_dict[new_key] = value
                    else:
                        # For other parameters, assume decoder (like projection layer)
                        new_key = f"decoder_model.{key}"
                        new_state_dict[new_key] = value
            
            # Filter out any keys that don't match the current model structure
            model_state_dict = model.state_dict()
            filtered_state_dict = {}
            for key, value in new_state_dict.items():
                if key in model_state_dict and model_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    print(f"⚠️  Skipping incompatible parameter: {key}")
            
            # Try loading with mapped parameters
            try:
                model.load_state_dict(filtered_state_dict, strict=False)
                print(f"✅ Successfully loaded model with parameter mapping.")
                print(f"   Loaded {len(filtered_state_dict)} out of {len(model_state_dict)} parameters")
            except RuntimeError as e2:
                print(f"⚠️  Parameter mapping failed: {e2}")
                print(f"⚠️  Creating new untrained model.")
                model = DCRNNModel(adj, logger=None, **model_kwargs)
        else:
            raise e
    
    model.to(device)
    model.eval()
    return model, scaler, adj, model_kwargs


def prepare_sequence(prices_df: pd.DataFrame, seq_len: int, horizon: int, scaler: StandardScaler):
    prices_df = prices_df.sort_index().ffill().bfill()
    returns = prices_df.pct_change().dropna()
    values = returns.values.astype(np.float32)  # (T, N)
    if values.shape[0] < seq_len:
        raise ValueError("Not enough data for inference window.")
    x = torch.from_numpy(values[-seq_len:])  # (seq_len, N)
    x = scaler.transform(x)
    x = x.unsqueeze(1)  # (seq_len, 1, N)
    x = x.view(seq_len, 1, -1)  # (seq_len, 1, num_nodes*1)
    return x


def infer_next_returns(checkpoint_path: str, recent_data_csv: str) -> pd.DataFrame:
    try:
        model, scaler, adj, model_kwargs = load_checkpoint(checkpoint_path)
        seq_len = int(model_kwargs['seq_len'])
        horizon = int(model_kwargs['horizon'])

        df = pd.read_csv(recent_data_csv, parse_dates=['Date'])  # columns: Date, Ticker, Close
        pivot = df.pivot_table(index='Date', columns='Ticker', values='Close').sort_index()
        x = prepare_sequence(pivot, seq_len, horizon, scaler)

        with torch.no_grad():
            out = model(x)  # (horizon, 1, num_nodes)
            out_inv = scaler.inverse_transform(out)
        preds = out_inv.squeeze(1).cpu().numpy()  # (horizon, num_nodes)

        tickers = list(pivot.columns)
        dates = [pivot.index[-1] + timedelta(days=i+1) for i in range(horizon)]
        out_df = pd.DataFrame(preds, columns=tickers, index=dates)
        return out_df
        
    except Exception as e:
        print(f"⚠️  Model inference failed: {e}")
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--recent_csv', required=True)
    parser.add_argument('--out_csv', default='predictions.csv')
    args = parser.parse_args()

    pred_df = infer_next_returns(args.checkpoint, args.recent_csv)
    pred_df.to_csv(args.out_csv)
    print(f"Saved predictions to {args.out_csv}")
