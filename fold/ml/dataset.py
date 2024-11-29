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


class ShapeDataset():

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
                                (len(t_seq) - max_len))
    return padded_sequences


def _pad_matrices(matrices):
    """Pads a batch of matrices"""
    pass


def collate_fn(tokenizer):
    """Creates a collate function."""

    def collate(examples):
        """Tokenizes and pads the inputs when extracting a batch from
        the dataset."""
        text = [e['sequence'] for e in examples]
        # TODO: Pad the sequences
        # TODO: Pad the matrixes with the distances

        # tokenizer_output = tokenizer(text, padding='longest')
        # class_target = [e['helpful'] for e in examples]

        # return {
        #     'review_text': torch.tensor(tokenizer_output['input_ids']),
        #     'attention_mask': torch.tensor(tokenizer_output['attention_mask']),
        #     'class_target': torch.tensor(class_target)
        # }

    return collate
