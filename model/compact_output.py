"""Rank-zero, latest-only plots/history and non-duplicated periodic checkpoints."""
import os
from pathlib import Path
import time

import numpy as np
import torch
import matplotlib.pyplot as plt


HISTORY_DTYPE = [(name, 'f8') for name in (
    'epoch', 'total', 'data', 'pde', 'raw_data', 'raw_pde',
    'validation_data', 'validation_pde', 'lr', 'train_seconds',
)]


def atomic_save(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_history(path, history):
    def write(temporary):
        with temporary.open('wb') as handle:
            np.save(handle, np.asarray(history, dtype=HISTORY_DTYPE), allow_pickle=False)
    atomic_save(path, write)


def plot_fields(results, path, epoch):
    """One figure contains real/imaginary true/prediction/error for every case."""
    fig, axes = plt.subplots(len(results), 6, figsize=(21, 3.5 * len(results)), squeeze=False)
    try:
        for row, result in enumerate(results):
            for channel in range(2):
                true, pred = result['true'][..., channel], result['pred'][..., channel]
                limit = max(float(np.abs(true).max()), float(np.abs(pred).max()), 1e-12)
                for col, (title, value) in enumerate((('True', true), ('Pred', pred), ('Error', pred - true))):
                    ax = axes[row, channel * 3 + col]
                    scale = max(float(np.abs(value).max()), 1e-12) if col == 2 else limit
                    im = ax.imshow(value, cmap='seismic', vmin=-scale, vmax=scale, aspect='auto')
                    ax.set_title(f'{result.get("name", "case")} {"Re" if channel == 0 else "Im"} {title}')
                    fig.colorbar(im, ax=ax, fraction=.046)
        fig.suptitle(f'Epoch {epoch} — scattered wavefield, full saved domain')
        fig.tight_layout()
        atomic_save(path, lambda temporary: fig.savefig(temporary, format='png', dpi=130))
    finally:
        plt.close(fig)


def plot_history(history, path):
    values = np.asarray(history, dtype=HISTORY_DTYPE)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    try:
        for ax, name in zip(axes, ('total', 'data', 'pde')):
            ax.semilogy(values['epoch'], np.maximum(values[name], 1e-30), label='train')
            if name != 'total':
                valid = values['validation_' + name]
                mask = np.isfinite(valid)
                ax.semilogy(values['epoch'][mask], np.maximum(valid[mask], 1e-30), 'o-', label='validation')
            ax.set(xlabel='Epoch', ylabel='Normalized loss', title=name)
            ax.legend()
        if not history:
            fig.suptitle('Epoch 0: untrained model; training loss not measured yet')
        fig.tight_layout()
        atomic_save(path, lambda temporary: fig.savefig(temporary, format='png', dpi=150))
    finally:
        plt.close(fig)


@torch.no_grad()
def plot_training_fields(model, args, plot_data, device, epoch):
    results = []
    was_training = model.training
    model.eval()
    try:
        for tag, label in (('pred', 'validation'), ('test', 'train')):
            velocity = plot_data['vel_' + tag].to(device)
            background = plot_data['UU0_' + tag].to(device)
            target = plot_data['labels_' + tag]
            freq_key = 'freq_valid' if tag == 'pred' else 'freq_train'
            source_key = 'source_coord_valid' if tag == 'pred' else 'source_coord_train'
            freq, source = plot_data.get(freq_key), plot_data.get(source_key)
            freq = freq[:1].to(device) if freq is not None else None
            source = source.to(device) if source is not None else None
            coordinates = plot_data['y_pred']
            predictions = []
            for part in coordinates.split(args.batch_size):
                predictions.append(model(velocity, part[None].to(device), background,
                    freq_batch=freq, source_coord_batch=source).cpu())
            height, width = target.shape[-2:]
            prediction = torch.cat(predictions, dim=1)[0].reshape(height, width, 2).numpy()
            results.append(dict(name=label, true=target[0].cpu().numpy().transpose(1, 2, 0), pred=prediction))
        plot_fields(results, Path(args.save_doc) / 'wavefields.png', epoch)
    finally:
        model.train(was_training)


def save_interrupted_checkpoint(args, state, error):
    """Best effort, rank-local, with NO collective operations on a failed group.

    Current weights/Adam state can restart an unfinished epoch, but this does
    not claim exact mid-epoch replay or recovery from a partially failed step.
    """
    wrapped = state.get('model')
    if wrapped is None:
        return None
    model = getattr(wrapped, 'module', wrapped)
    optimizer = state.get('optimizer')
    scheduler = state.get('scheduler')
    warmup = state.get('warmup_scheduler')
    completed = int(state.get('last_completed_epoch', int(getattr(args, 'start_epoch', 1)) - 1))
    payload = dict(
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict() if optimizer is not None else None,
        scheduler_state_dict=scheduler.state_dict() if scheduler is not None else None,
        warmup_scheduler_state_dict=warmup.state_dict() if warmup is not None else None,
        epoch=completed,
        interrupted_epoch=int(state.get('i', completed + 1)),
        completed_microsteps=int(state.get('step_counter', 0)),
        last_completed_epoch=completed,
        optimizer_step_in_progress=bool(state.get('optimizer_step_in_progress', False)),
        exact_mid_epoch_resume=False,
        resume_policy='restart unfinished epoch from current parameters; discard pending gradients',
        exception_type=type(error).__name__, exception=str(error),
        loss_history=np.asarray(state.get('compact_history', []), dtype=HISTORY_DTYPE),
        loss_normalizers=(None if state.get('first_flag', True) else dict(
            data=state['data_norm_coe'], pde=state['pde_norm_coe'], env=state['env_norm_coe'])),
        rng_state_cpu=torch.get_rng_state(),
    )
    path = Path(args.save_doc) / 'checkpoint_interrupted.pth'
    atomic_save(path, lambda temporary: torch.save(payload, temporary))
    print(f'[Interrupted] saved {path}; last_completed_epoch={completed}; '
          f'optimizer_step_in_progress={payload["optimizer_step_in_progress"]}; '
          'not an exact mid-epoch resume', flush=True)
    return path


def finish_epoch(args, model, optimizer, scheduler, epoch, history, plot_data,
                 evaluators, normalizers, resume_spec, device, train_seconds, lr,
                 data_loss, pde_loss, validation_data=np.nan, validation_pde=np.nan):
    """Called by rank 0 only. Other ranks wait at the existing epoch barrier."""
    from test import evaluate_training_unseen_frequency
    total = args.a * data_loss + args.b * pde_loss
    if not np.isfinite([total, data_loss, pde_loss]).all():
        raise FloatingPointError(f'Non-finite loss at epoch {epoch}')
    history.append((epoch, total, data_loss, pde_loss,
                    data_loss * normalizers['data'], pde_loss * normalizers['pde'],
                    validation_data, validation_pde, lr, train_seconds))
    print(f'[Epoch {epoch}/{args.NIter - 1}] total={total:.10e} data={data_loss:.10e} '
          f'pde={pde_loss:.10e} val_data={validation_data:.10e} val_pde={validation_pde:.10e} '
          f'raw_data={data_loss * normalizers["data"]:.10e} raw_pde={pde_loss * normalizers["pde"]:.10e} '
          f'lr={lr:.8e} train_seconds={train_seconds:.3f}', flush=True)
    started = time.perf_counter()
    figure_due = epoch == 1 or epoch % args.save_fig_every == 0
    if figure_due:
        plot_history(history, Path(args.save_doc) / 'loss_curve.png')
        plot_training_fields(model, args, plot_data, device, epoch)
        for evaluator in evaluators:
            evaluate_training_unseen_frequency(evaluator, model, epoch)
    if epoch % args.save_model_every == 0 or epoch == args.NIter - 1:
        checkpoint = dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict(), epoch=epoch, loss_normalizers=normalizers,
            resumed_from=resume_spec[0] if resume_spec else None,
            loss_history=np.asarray(history, dtype=HISTORY_DTYPE))
        name = 'checkpoint_final.pth' if epoch == args.NIter - 1 else f'checkpoint_epoch_{epoch:04d}.pth'
        atomic_save(Path(args.save_doc) / name,
                    lambda temporary: torch.save(checkpoint, temporary))
    if figure_due or epoch == args.NIter - 1:
        save_history(Path(args.save_doc) / 'loss_history.npy', history)
    if figure_due or epoch % args.save_model_every == 0 or epoch == args.NIter - 1:
        print(f'[Output epoch={epoch}] seconds={time.perf_counter() - started:.3f}', flush=True)
