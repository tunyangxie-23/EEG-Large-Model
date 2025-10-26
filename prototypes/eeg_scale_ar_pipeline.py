"""
Prototype training and inference pipeline for autoregressive EEG frequency synthesis.

The pipeline treats each cumulative frequency band (f1, f1+f2, ...) as a progressive
stage analogous to image scales in VAR. A lightweight Transformer learns to map
the previous cumulative signal to the next higher-frequency aggregate.
"""
import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Sampler

from .eeg_frequency_dataset import EEGFrequencyDataset


@dataclass
class TrainConfig:
    data_root: str
    epochs: int = 10
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.0
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    grad_clip: float = 1.0
    device: str = 'cpu'
    num_workers: int = 2
    log_interval: int = 50
    checkpoint_dir: str = 'runs/eeg_ar'
    limit_train_samples: Optional[int] = None
    limit_val_samples: Optional[int] = None
    resume: Optional[str] = None
    seed: int = 42


class EEGScaleAutoregressiveModel(nn.Module):
    """
    Transformer encoder that predicts the next cumulative frequency signal
    from the current aggregate.
    """
    def __init__(
        self,
        channels: int,
        time_steps: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.channels = channels
        self.time_steps = time_steps
        self.d_model = d_model
        self.input_proj = nn.Linear(channels, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional = nn.Parameter(torch.zeros(1, time_steps, d_model))
        self.output_proj = nn.Linear(d_model, channels)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.trunc_normal_(self.input_proj.weight, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.trunc_normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.trunc_normal_(self.positional, std=0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor shaped (batch, channels, time)
        Returns:
            Tensor shaped (batch, channels, time)
        """
        x = x.transpose(1, 2)  # (batch, time, channels)
        if x.size(1) > self.time_steps:
            raise ValueError(f'Input time steps {x.size(1)} exceed configured maximum {self.time_steps}')
        emb = self.input_proj(x)
        emb = emb + self.positional[:, :emb.size(1)]
        emb = self.dropout(emb)
        enc = self.encoder(emb)
        out = self.output_proj(enc)
        return out.transpose(1, 2)


def prepare_inputs_targets(batch: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert batch of cumulative stacks into stage-wise inputs and targets.
    batch: (B, S, C, T)
    """
    batch = batch.to(device)
    prev = torch.zeros_like(batch[:, :1])
    inputs = torch.cat([prev, batch[:, :-1]], dim=1)
    B, S, C, T = inputs.shape
    inputs = inputs.view(B * S, C, T)
    targets = batch.view(B * S, C, T)
    return inputs, targets


def train_epoch(model, optimizer, dataloader, device, grad_clip, scaler=None, log_interval=50):
    model.train()
    mse = nn.MSELoss()
    running_loss = 0.0
    total_loss = 0.0
    batches = 0
    for step, (cumulative, _) in enumerate(dataloader, 1):
        inputs, targets = prepare_inputs_targets(cumulative, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            preds = model(inputs)
            loss = mse(preds, targets)
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        loss_val = loss.detach().item()
        running_loss += loss_val
        total_loss += loss_val
        batches += 1
        if step % log_interval == 0:
            avg = running_loss / log_interval
            print(f'[train] step {step}/{len(dataloader)}  loss={avg:.6f}')
            running_loss = 0.0
    return total_loss / max(batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    mse = nn.MSELoss()
    total_loss = 0.0
    count = 0
    for cumulative, _ in dataloader:
        inputs, targets = prepare_inputs_targets(cumulative, device)
        preds = model(inputs)
        loss = mse(preds, targets)
        total_loss += loss.item()
        count += 1
    return total_loss / max(count, 1)


@torch.no_grad()
def autoregressive_inference(model, num_stages, channels, time_steps, device):
    """
    Sequentially roll out predictions without teacher forcing.
    Returns tensor shaped (num_stages, channels, time_steps).
    """
    model.eval()
    prev = torch.zeros(1, channels, time_steps, device=device)
    outputs = []
    for stage in range(num_stages):
        pred = model(prev)
        outputs.append(pred.squeeze(0).cpu())
        prev = pred
    return torch.stack(outputs, dim=0)


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_limit_dataset(dataset, max_samples: Optional[int]):
    if max_samples is None or max_samples <= 0:
        return dataset
    max_samples = min(len(dataset), max_samples)
    return Subset(dataset, range(max_samples))


class TorchRandomPermutationSampler(Sampler[int]):
    def __init__(self, data_source, seed: int = 0):
        self.data_source = data_source
        self.base_seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.data_source)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.base_seed + self.epoch)
        order = torch.randperm(len(self.data_source), generator=g).tolist()
        for idx in order:
            yield int(idx)


def main(cfg: TrainConfig):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    set_seed(cfg.seed)
    
    train_base = EEGFrequencyDataset(cfg.data_root, split='train')
    try:
        val_base = EEGFrequencyDataset(cfg.data_root, split='val')
    except FileNotFoundError:
        val_base = EEGFrequencyDataset(cfg.data_root, split='', recursive=True)
        print('[warn] validation split not found; using full dataset for evaluation')

    train_dataset = maybe_limit_dataset(train_base, cfg.limit_train_samples)
    val_dataset = maybe_limit_dataset(val_base, cfg.limit_val_samples)
    
    train_sampler = TorchRandomPermutationSampler(train_dataset, seed=cfg.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )
    
    num_stages, channels, time_steps = train_base.input_shape
    print(f'[data] stages={num_stages}, channels={channels}, time_steps={time_steps}, samples={len(train_dataset)}')
    
    model = EEGScaleAutoregressiveModel(
        channels=channels,
        time_steps=time_steps,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(cfg.device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    
    start_epoch = 1
    best_val = float('inf')
    if cfg.resume:
        checkpoint = torch.load(cfg.resume, map_location=cfg.device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        best_val = checkpoint.get('best_val', best_val)
        start_epoch = checkpoint.get('epoch', start_epoch)
        print(f'[resume] loaded checkpoint from {cfg.resume} at epoch {start_epoch-1}')
    use_amp = cfg.device.startswith('cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val = float('inf') if best_val == float('inf') else best_val
    for epoch in range(start_epoch, cfg.epochs + 1):
        train_sampler.set_epoch(epoch)
        print(f'\n=== Epoch {epoch}/{cfg.epochs} ===')
        train_loss = train_epoch(model, optimizer, train_loader, cfg.device, cfg.grad_clip, scaler if use_amp else None, cfg.log_interval)
        val_loss = evaluate(model, val_loader, cfg.device)
        print(f'[val] loss={val_loss:.6f}')
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(cfg.checkpoint_dir, 'best.pt')
            torch.save({
                'model_state': model.state_dict(),
                'config': cfg.__dict__,
                'data_shape': (num_stages, channels, time_steps),
                'best_val': best_val,
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch + 1,
            }, ckpt_path)
            print(f'[ckpt] saved best model to {ckpt_path}')

    # Demo inference using the trained model.
    recon = autoregressive_inference(model, num_stages, channels, time_steps, cfg.device)
    demo_path = os.path.join(cfg.checkpoint_dir, 'demo_prediction.pt')
    torch.save(recon, demo_path)
    print(f'[inference] saved autoregressive rollout to {demo_path}')

    history_path = os.path.join(cfg.checkpoint_dir, 'metrics.json')
    with open(history_path, 'w', encoding='utf-8') as fp:
        json.dump(history, fp, indent=2)
    print(f'[metrics] saved training curves to {history_path}')


def canonicalize_device(device_str: str) -> str:
    if not device_str:
        return 'cpu'
    name = device_str.lower().strip()
    if name in {'gpu', 'cuda'}:
        name = 'cuda:0'
    elif name in {'cuda0', 'gpu0'}:
        name = 'cuda:0'
    if name.startswith('cuda'):
        if not torch.cuda.is_available():
            print('[warn] requested CUDA device but torch.cuda.is_available() is False; falling back to CPU')
            return 'cpu'
        device_name = name if ':' in name else f'{name}:0'
        try:
            torch.zeros(1, device=device_name)
        except Exception as exc:
            print(f"[warn] CUDA device '{device_name}' unavailable ({exc}); falling back to CPU")
            return 'cpu'
        return device_name
    return 'cpu'


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description='Prototype EEG frequency autoregressive training.')
    parser.add_argument('--data-root', type=str, required=True, help='Path containing train/ and optional val/ .npz files.')
    parser.add_argument('--epochs', type=int, default=TrainConfig.epochs)
    parser.add_argument('--batch-size', type=int, default=TrainConfig.batch_size)
    parser.add_argument('--lr', type=float, default=TrainConfig.lr)
    parser.add_argument('--weight-decay', type=float, default=TrainConfig.weight_decay)
    parser.add_argument('--d-model', type=int, default=TrainConfig.d_model)
    parser.add_argument('--nhead', type=int, default=TrainConfig.nhead)
    parser.add_argument('--num-layers', type=int, default=TrainConfig.num_layers)
    parser.add_argument('--dropout', type=float, default=TrainConfig.dropout)
    parser.add_argument('--grad-clip', type=float, default=TrainConfig.grad_clip)
    parser.add_argument('--num-workers', type=int, default=TrainConfig.num_workers)
    parser.add_argument('--log-interval', type=int, default=TrainConfig.log_interval)
    parser.add_argument('--checkpoint-dir', type=str, default=TrainConfig.checkpoint_dir)
    parser.add_argument('--limit-train-samples', type=int, default=None)
    parser.add_argument('--limit-val-samples', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=TrainConfig.seed)
    parser.add_argument('--device', type=str, default=TrainConfig.device)
    args = parser.parse_args()
    return TrainConfig(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
        log_interval=args.log_interval,
        checkpoint_dir=args.checkpoint_dir,
        limit_train_samples=args.limit_train_samples,
        limit_val_samples=args.limit_val_samples,
        resume=args.resume,
        seed=args.seed,
        device=canonicalize_device(args.device),
    )


if __name__ == '__main__':
    main(parse_args())
