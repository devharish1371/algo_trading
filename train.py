import argparse
import os

from dcrnn_supervisor import DCRNNSupervisor

# Colab-friendly defaults
DEFAULT_PARAMS = dict(
    data_csv=None,          # must be provided
    seq_len=24,
    horizon=4,
    batch_size=32,
    val_ratio=0.1,
    test_ratio=0.1,
    epochs=50,
    base_lr=1e-3,
    log_dir='logs',
    model_dir='logs/checkpoints',
)


def run_training(
    data_csv: str,
    seq_len: int = DEFAULT_PARAMS['seq_len'],
    horizon: int = DEFAULT_PARAMS['horizon'],
    batch_size: int = DEFAULT_PARAMS['batch_size'],
    val_ratio: float = DEFAULT_PARAMS['val_ratio'],
    test_ratio: float = DEFAULT_PARAMS['test_ratio'],
    epochs: int = DEFAULT_PARAMS['epochs'],
    base_lr: float = DEFAULT_PARAMS['base_lr'],
    log_dir: str = DEFAULT_PARAMS['log_dir'],
    model_dir: str = DEFAULT_PARAMS['model_dir'],
):
    """Run training programmatically (Colab/Notebook friendly).

    Example (in Colab):
        run_training(
            data_csv='/content/bse500_data.csv',
            seq_len=24,
            horizon=4,
            batch_size=32,
            epochs=30,
            log_dir='/content/logs',
            model_dir='/content/logs/checkpoints',
        )
    """
    if not data_csv or not os.path.exists(data_csv):
        raise FileNotFoundError(f"CSV not found: {data_csv}")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    supervisor = DCRNNSupervisor(
        data=dict(
            data_path=data_csv,
            seq_len=seq_len,
            horizon=horizon,
            batch_size=batch_size,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            price_column='Close',
            use_returns=True,
        ),
        model=dict(
            rnn_units=64,
            num_rnn_layers=2,
            filter_type='random_walk',
        ),
        train=dict(
            epochs=epochs,
            base_lr=base_lr,
            log_dir=log_dir,
            model_dir=model_dir,
            lr_steps=[int(epochs * 0.6), int(epochs * 0.8)],
            lr_decay_ratio=0.5,
            patience=10,
        ),
        log_level='INFO',
    )

    supervisor.train()


def train_colab(params: dict | None = None, **kwargs):
    """Convenience wrapper for Google Colab.

    Usage in a notebook cell:
        from train import train_colab
        train_colab({
            'data_csv': '/content/bse500_data.csv',
            'epochs': 30,
            'batch_size': 64,
            'log_dir': '/content/logs',
            'model_dir': '/content/logs/checkpoints',
        })

    You can also pass kwarg overrides: train_colab(epochs=10, ...)
    """
    merged = DEFAULT_PARAMS.copy()
    if params:
        merged.update({k: v for k, v in params.items() if v is not None})
    merged.update({k: v for k, v in kwargs.items() if v is not None})

    return run_training(
        data_csv=merged['data_csv'],
        seq_len=merged['seq_len'],
        horizon=merged['horizon'],
        batch_size=merged['batch_size'],
        val_ratio=merged['val_ratio'],
        test_ratio=merged['test_ratio'],
        epochs=merged['epochs'],
        base_lr=merged['base_lr'],
        log_dir=merged['log_dir'],
        model_dir=merged['model_dir'],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv', required=True, help='CSV with columns Date,Ticker,Close (and others)')
    parser.add_argument('--seq_len', type=int, default=24)
    parser.add_argument('--horizon', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--base_lr', type=float, default=1e-3)
    parser.add_argument('--log_dir', default='logs')
    parser.add_argument('--model_dir', default='logs/checkpoints')
    args = parser.parse_args()

    run_training(
        data_csv=args.data_csv,
        seq_len=args.seq_len,
        horizon=args.horizon,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        epochs=args.epochs,
        base_lr=args.base_lr,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
    )


if __name__ == '__main__':
    main()


