import os
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from utils import get_logger, load_dataset
from dcrnn_model import DCRNNModel


def masked_mae_loss(y_pred, y_true):
    """Masked Mean Absolute Error loss."""
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    loss[torch.isnan(loss)] = 0
    return loss.mean()

def mse_loss(y_pred, y_true):
    """Mean Squared Error loss - better for return prediction."""
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.square(y_pred - y_true)
    loss = loss * mask
    loss[torch.isnan(loss)] = 0
    return loss.mean()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DCRNNSupervisor:
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data', {})
        self._model_kwargs = kwargs.get('model', {})
        self._train_kwargs = kwargs.get('train', {})

        self.max_grad_norm = float(self._train_kwargs.get('max_grad_norm', 1.0))

        # logging
        self._log_dir = self._train_kwargs.get('log_dir', 'logs')
        os.makedirs(self._log_dir, exist_ok=True)
        self._writer = SummaryWriter(self._log_dir)
        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = get_logger(self._log_dir, __name__, 'info.log', level=log_level)

        # dataset
        self._data = load_dataset(**self._data_kwargs)
        self.standard_scaler = self._data['scaler']
        self.num_nodes = int(self._data['num_nodes'])

        # model params
        self._model_kwargs.setdefault('num_nodes', self.num_nodes)
        self._model_kwargs.setdefault('input_dim', 1)
        self._model_kwargs.setdefault('output_dim', 1)
        self._model_kwargs.setdefault('rnn_units', 64)
        self._model_kwargs.setdefault('num_rnn_layers', 2)
        self._model_kwargs.setdefault('seq_len', int(self._data_kwargs.get('seq_len', 12)))
        self._model_kwargs.setdefault('horizon', int(self._data_kwargs.get('horizon', 3)))
        self._model_kwargs.setdefault('filter_type', 'random_walk')

        adj_mx = self._data['adjacency']
        model = DCRNNModel(adj_mx, logger=self._logger, **self._model_kwargs)
        self.model = model.cuda() if torch.cuda.is_available() else model
        self._logger.info("Model created")

        # training params
        self._start_epoch = int(self._train_kwargs.get('start_epoch', 0))
        self._epochs = int(self._train_kwargs.get('epochs', 50))
        self._base_lr = float(self._train_kwargs.get('base_lr', 1e-3))
        self._milestones = list(self._train_kwargs.get('lr_steps', [30, 45]))
        self._lr_gamma = float(self._train_kwargs.get('lr_decay_ratio', 0.1))
        self._patience = int(self._train_kwargs.get('patience', 10))
        self._model_dir = self._train_kwargs.get('model_dir', os.path.join(self._log_dir, 'checkpoints'))
        os.makedirs(self._model_dir, exist_ok=True)

    def save_model(self, epoch: int):
        path = os.path.join(self._model_dir, f'model_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'scaler_mean': self.standard_scaler.mean,
            'scaler_std': self.standard_scaler.std,
            'adjacency': self._data['adjacency'],
            'model_kwargs': self._model_kwargs,
            'num_nodes': self.num_nodes,
        }, path)
        self._logger.info(f"Saved model to {path}")

    def load_latest(self) -> int:
        files = [f for f in os.listdir(self._model_dir) if f.startswith('model_epoch_') and f.endswith('.pt')]
        if not files:
            return 0
        files.sort(key=lambda p: int(p.split('_')[-1].split('.')[0]))
        ckpt = os.path.join(self._model_dir, files[-1])
        state = torch.load(ckpt, map_location=device)
        self.model.load_state_dict(state['state_dict'])
        # restore scaler
        if 'scaler_mean' in state and 'scaler_std' in state:
            from utils import StandardScaler
            self.standard_scaler = StandardScaler(mean=state['scaler_mean'], std=state['scaler_std'])
        # restore adjacency if needed in downstream usage
        self._data['adjacency'] = state.get('adjacency', self._data['adjacency'])
        self.num_nodes = int(state.get('num_nodes', self.num_nodes))
        self._logger.info(f"Loaded checkpoint {ckpt}")
        return int(state.get('epoch', 0))

    def _prepare_batch(self, x: torch.Tensor, y: torch.Tensor) -> tuple:
        return x.to(device), y.to(device)

    def evaluate(self, split: str = 'val', batches_seen: int = 0) -> float:
        self.model.eval()
        loader = self._data[f'{split}_loader']
        losses = []
        with torch.no_grad():
            for x, y in loader.get_iterator():
                x, y = self._prepare_batch(x, y)
                out = self.model(x, labels=y, batches_seen=batches_seen)
                # inverse transform to original scale
                y_true = self.standard_scaler.inverse_transform(y)
                y_pred = self.standard_scaler.inverse_transform(out)
                loss = mse_loss(y_pred, y_true)  # Use MSE for better return prediction
                losses.append(loss.item())
        mean_loss = float(np.mean(losses)) if losses else float('inf')
        self._writer.add_scalar(f'{split}/loss', mean_loss, batches_seen)
        return mean_loss

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self._base_lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self._milestones, gamma=self._lr_gamma)
        self._logger.info('Start training...')

        num_batches = self._data['train_loader'].num_batch
        batches_seen = num_batches * self._start_epoch
        best_val = float('inf')
        wait = 0

        for epoch in range(self._start_epoch, self._epochs):
            self.model.train()
            losses = []
            start = time.time()
            for x, y in self._data['train_loader'].get_iterator():
                optimizer.zero_grad()
                x, y = self._prepare_batch(x, y)
                out = self.model(x, labels=y, batches_seen=batches_seen)
                y_true = self.standard_scaler.inverse_transform(y)
                y_pred = self.standard_scaler.inverse_transform(out)
                loss = mse_loss(y_pred, y_true)  # Use MSE for better return prediction
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                optimizer.step()
                losses.append(loss.item())
                batches_seen += 1

            scheduler.step()
            val_loss = self.evaluate('val', batches_seen)
            dur = time.time() - start
            self._logger.info(f"Epoch {epoch}: train_loss={np.mean(losses):.4f} val_loss={val_loss:.4f} lr={scheduler.get_last_lr()[0]:.6f} time={dur:.1f}s")

            if val_loss < best_val:
                self.save_model(epoch)
                best_val = val_loss
                wait = 0
            else:
                wait += 1
                if wait >= self._patience:
                    self._logger.warning(f"Early stopping at epoch {epoch}")
                    break

