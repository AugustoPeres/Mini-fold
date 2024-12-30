import os

import numpy as np
import torch

DEFAULT_TOKENIZER = {
    'PAD': 0,
    'a': 1,
    'b': 2,
    'c': 3,
    'd': 4,
    'e': 5,
    'f': 6,
    'g': 7,
    'h': 8
}


class ShapeDataset(torch.utils.data.Dataset):

    def __init__(self, folder):
        self.folder = folder

        self.dirs = [
            os.path.join(folder, d) for d in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, d))
        ]

        self.length = len(self.dirs)

    def _compute_distances(self, positions):
        # TODO: Implement this with numpy operations for efficiency
        seq_len = positions.shape[0]
        distances = np.zeros((seq_len, seq_len))

        for i in range(seq_len):
            for j in range(seq_len):
                distances[i,
                          j] = np.sqrt((positions[i, 0] - positions[j, 0])**2 +
                                       (positions[i, 1] - positions[j, 1])**2)
        return distances

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sub_folder = self.dirs[idx]
        sequence_string = os.path.split(sub_folder)[-1]

        positions = np.load(os.path.join(sub_folder, 'positions.npy'))
        distances = self._compute_distances(positions)
        return {'sequence': sequence_string, 'distances': distances}


def _tokenize_batch(tokenizer, seq_batch):
    """Tokenizes and pads a batch of sequences"""
    tokenized_sequences = []
    max_len = 0
    for seq in seq_batch:
        tokens = [tokenizer[c] for c in seq.split(' ')]
        tokenized_sequences.append(tokens)
        max_len = max(max_len, len(tokens))

    padded_sequences = []
    for t_seq in tokenized_sequences:
        padded_sequences.append(t_seq + [tokenizer['PAD']] *
                                (max_len - len(t_seq)))
    padded_sequences = np.array(padded_sequences)
    padding_mask = padded_sequences == tokenizer['PAD']
    return padded_sequences, padding_mask


def _pad_matrices(matrices):
    """Pads a batch of matrices"""
    max_size = max(map(lambda x: x.shape[0], matrices))
    padded_matrices = []
    padding_masks = []
    for m in matrices:
        i, j = m.shape
        padded_matrices.append(
            np.pad(m, pad_width=((0, max_size - i), (0, max_size - j))))
        padding_mask = np.ones_like(m)
        np.fill_diagonal(padding_mask, 0)
        padding_masks.append(
            np.pad(padding_mask,
                   pad_width=((0, max_size - i), (0, max_size - j))))
    return np.array(padded_matrices), np.array(padding_masks)


def collate_fn(tokenizer):
    """Creates a collate function."""

    def collate(examples):
        """Tokenizes and pads the inputs when extracting a batch from
        the dataset."""
        text = [e['sequence'] for e in examples]
        distances = [e['distances'] for e in examples]
        tokenized_sequences, padding_masks = _tokenize_batch(tokenizer, text)
        distance_matrices, distance_padding = _pad_matrices(distances)
        return {
            'sequences':
            torch.tensor(tokenized_sequences, dtype=torch.int32),
            'seq_padding_mask':
            torch.tensor(padding_masks, dtype=torch.bool),
            'distance_matrices':
            torch.tensor(distance_matrices, dtype=torch.float32),
            'distance_padding':
            torch.tensor(distance_padding, dtype=torch.float32)
        }

    return collate
