"""Script to generate data with"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot

from fold.data_generation import generator


def vis_sequence(sequence, positions, save_path):
    x = [p[0] for p in positions]
    y = [p[1] for p in positions]
    plt.plot(x, y, '.')
    plt.plot(x, y)
    plt.savefig(save_path)
    plt.close()


def main(num_sequences, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    g = generator.default_generator()
    for _ in range(num_sequences):
        seq, pos = g.make_positions_for_random_sequence()
        folder = os.path.join(output_folder, ' '.join(seq))
        pos_save_path = os.path.join(folder, 'positions.npy')
        img_save_path = os.path.join(folder, 'image.png')
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.save(pos_save_path, np.array(pos))
        vis_sequence(seq, pos, img_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate dummy data for mini alpha fold.')
    parser.add_argument('num_sequences',
                        help='The number of sequences to generate.',
                        default=10,
                        type=int)
    parser.add_argument('output_folder',
                        help='Where to store the data',
                        default='out')

    args = parser.parse_args()
    main(args.num_sequences, args.output_folder)
