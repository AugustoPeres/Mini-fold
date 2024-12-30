"""Use the Folder and model predictions to get the shapes of sequences

python fold_sequence.py --run_id 2f406f1dde874d15bc83f310e750e9ff --sequence abeaefcgeegaccffddeaggdheabbbaggaffd --max_epochs 200
"""
import os
import argparse

import lightning
import torch
import numpy as np
import mlflow
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

from fold import ml
from fold import Folder


def load_lightning_model_from_run(run_id):
    mlflow.set_experiment('SeqTransformer Training')
    artifact_path = 'model/checkpoints/best_model/best_model.ckpt'
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_path)
    model = ml.SeqTransformer.load_from_checkpoint(local_path)
    return model


def create_positions_video(positions_history,
                           output_dir='output',
                           video_name='positions_animation.mp4'):
    """Creates a VIDEO from the history of positions."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots()
    line, = ax.plot([], [], '.', markersize=8)
    trajectory, = ax.plot([], [], '-')

    def init():
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        return line, trajectory

    def update(frame):
        positions = positions_history[frame]
        x = positions[:, 0]
        x = x
        y = positions[:, 1]
        y = y
        line.set_data(x, y)
        trajectory.set_data(x, y)
        return line, trajectory

    ani = FuncAnimation(fig,
                        update,
                        frames=len(positions_history),
                        init_func=init,
                        blit=True)

    video_path = os.path.join(output_dir, video_name)
    video_path = os.path.join(output_dir, video_name)
    writer = FFMpegWriter(fps=10,
                          metadata={'artist': 'Matplotlib'},
                          codec='libx264',
                          bitrate=1800)
    ani.save(video_path, writer=writer)

    print(f'Video saved at {video_path}')
    return video_path


def main(run_id, sequence, iterations_per_epoch, max_epochs):
    model = load_lightning_model_from_run(run_id).to('cpu')
    loader = torch.utils.data.DataLoader(range(iterations_per_epoch),
                                         batch_size=1)
    tokens = [ml.dataset.DEFAULT_TOKENIZER[c] for c in list(sequence)]
    seq_len = len(tokens)
    tokens = torch.tensor([tokens]).to('cpu')

    with torch.no_grad():
        distances = torch.reshape(model(tokens)[0], (seq_len, seq_len))
        distances = distances * (torch.ones_like(distances) -
                                 torch.eye(distances.shape[0]))

    folder = Folder(distances).to('cpu')
    trainer = lightning.Trainer(max_epochs=max_epochs,
                                log_every_n_steps=1,
                                accelerator='cpu')
    trainer.fit(folder, loader)

    positions_history = folder.positions_history
    create_positions_video(positions_history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Lightning model.')
    parser.add_argument('--run_id',
                        type=str,
                        required=True,
                        help='MLflow run ID.')
    parser.add_argument('--sequence',
                        type=str,
                        required=True,
                        help='Input sequence.')
    parser.add_argument('--iterations_per_epoch',
                        type=int,
                        default=1,
                        help='Number of iterations per epoch.')
    parser.add_argument('--max_epochs',
                        type=int,
                        default=1000,
                        help='Maximum number of epochs.')

    args = parser.parse_args()
    main(args.run_id, args.sequence, args.iterations_per_epoch,
         args.max_epochs)
