"""Data generation processes"""
import random

import torch


class Generator:

    def __init__(self,
                 alphabet,
                 embedding,
                 foward_rnn,
                 backward_rnn,
                 position_mlp,
                 back_discount,
                 h_0,
                 max_len=40,
                 min_len=20):
        self.alphabet = alphabet
        self.embedding = embedding
        self.foward_rnn = foward_rnn
        self.backward_rnn = backward_rnn
        self.position_mlp = position_mlp
        self.back_discount = back_discount
        self.h_0 = h_0

        self.max_len = max_len
        self.min_len = min_len

        self.tokenizer = {k: v for v, k in enumerate(alphabet)}

    def _random_sequence(self):
        seq_len = random.choice(range(self.min_len, self.max_len))
        return [random.choice(self.alphabet) for _ in range(seq_len)]

    def embed_sequence(self, sequence):
        tokenized_sequence = [self.tokenizer[letter] for letter in sequence]
        return self.embedding(torch.LongTensor([tokenized_sequence
                                                ])).squeeze(0)

    def foward_hidden_states(self, sequence):
        return self.hidden_states(sequence, self.foward_rnn)

    def backward_hidden_states(self, sequence):
        return self.hidden_states(reversed(sequence), self.backward_rnn)

    def make_positions_for_random_sequence(self):
        seq = self._random_sequence()
        return seq, self.make_positions(seq)

    @torch.no_grad
    def make_positions(self, sequence):
        f_hidden_states = self.foward_hidden_states(sequence)
        b_hidden_states = self.backward_hidden_states(sequence)

        hidden_states = [
            h_f + self.back_discount * h_b
            for h_f, h_b in zip(f_hidden_states, b_hidden_states)
        ]

        positions = self.position_mlp(torch.stack(hidden_states))

        first_point = positions[0]
        final_positions = [(first_point - first_point).tolist()]
        point = first_point
        for p in positions[1:]:
            point = point + p
            final_positions.append(point.tolist())
        return final_positions

    @torch.no_grad
    def hidden_states(self, sequence, rnn):
        emb_sequence = self.embed_sequence(sequence)

        h = None
        h_list = []
        for embedding in emb_sequence:
            if h is None:
                h = self.h_0
            _, h = rnn(torch.reshape(embedding, (1, 1, embedding.shape[0])), h)
            h_list.append(torch.squeeze(h))
        return h_list


_DEFAULT_ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']


def default_generator():
    torch.manual_seed(0)
    embedding = torch.nn.Embedding(len(_DEFAULT_ALPHABET), 8)
    torch.nn.init.uniform_(embedding.weight)
    foward_rnn = torch.nn.RNN(8, 4, batch_first=True)
    backward_rnn = torch.nn.RNN(8, 4, batch_first=True)
    linear = torch.nn.Linear(4, 2)
    position_mlp = torch.nn.Sequential(linear, torch.nn.Tanh())
    return Generator(alphabet=_DEFAULT_ALPHABET,
                     embedding=embedding,
                     foward_rnn=foward_rnn,
                     backward_rnn=backward_rnn,
                     position_mlp=position_mlp,
                     back_discount=.1,
                     h_0=torch.zeros(1, 1, 4))
