"""Training script"""
import argparse
import os

import torch
import lightning
from lightning.pytorch.loggers import MLFlowLogger
import mlflow

from fold import ml


def _make_dataloaders(data_path, batch_size, tokenizer, shuffle):
    dataset = ml.ShapeDataset(data_path)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=ml.dataset.collate_fn(tokenizer),
        shuffle=shuffle)
    return data_loader


def main(train_data_path, validation_data_path, batch_size, d_model, nhead,
         d_hid, nlayers, learning_rate, max_sequence_len,
         distance_mlp_layer_sizes, dropout, max_epochs):
    train_dataloader = _make_dataloaders(train_data_path,
                                         batch_size,
                                         ml.dataset.DEFAULT_TOKENIZER,
                                         shuffle=True)
    validation_dataloader = _make_dataloaders(validation_data_path,
                                              batch_size,
                                              ml.dataset.DEFAULT_TOKENIZER,
                                              shuffle=False)

    ntoken = len(ml.dataset.DEFAULT_TOKENIZER.keys())
    model = ml.model.SeqTransformer(
        ntoken=ntoken,
        d_model=d_model,
        nhead=nhead,
        d_hid=d_hid,
        nlayers=nlayers,
        learning_rate=learning_rate,
        max_sequence_len=max_sequence_len,
        distance_mlp_layer_sizes=distance_mlp_layer_sizes,
        dropout=dropout)

    mlf_logger = MLFlowLogger(experiment_name='SeqTransformer Training',
                              log_model=True)
    mlf_logger.log_hyperparams({
        'd_model': d_model,
        'nhead': nhead,
        'd_hid': d_hid,
        'nlayers': nlayers,
        'learning_rate': learning_rate,
        'max_sequence_len': max_sequence_len,
        'distance_mlp_layer_sizes': distance_mlp_layer_sizes,
        'dropout': dropout,
        'batch_size': batch_size,
        'max_epochs': max_epochs
    })
    checkpoint_callback = lightning.pytorch.callbacks.ModelCheckpoint(
        monitor='val_loss', mode='min', save_top_k=1, filename='best_model')

    trainer = lightning.Trainer(max_epochs=max_epochs,
                                logger=mlf_logger,
                                callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloader, validation_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train a SeqTransformer model.")
    parser.add_argument('--train_dataset_path',
                        type=str,
                        required=True,
                        help="Path to training data.")
    parser.add_argument('--validation_dataset_path',
                        type=str,
                        required=True,
                        help="Path to validation data.")
    parser.add_argument('--d_model',
                        type=int,
                        default=256,
                        help="Dimension of the model.")
    parser.add_argument('--nhead',
                        type=int,
                        default=4,
                        help="Number of attention heads.")
    parser.add_argument('--d_hid',
                        type=int,
                        default=256,
                        help="Dimension of the feedforward network.")
    parser.add_argument('--nlayers',
                        type=int,
                        default=4,
                        help="Number of layers in the transformer.")
    parser.add_argument('--learning_rate',
                        type=float,
                        default=1e-3,
                        help="Learning rate for the optimizer.")
    parser.add_argument('--max_sequence_len',
                        type=int,
                        default=100,
                        help="Maximum sequence length.")
    parser.add_argument('--distance_mlp_layer_sizes',
                        nargs='+',
                        type=int,
                        default=[256, 64, 1],
                        help="Sizes of layers in the distance MLP.")
    parser.add_argument('--dropout',
                        type=float,
                        default=.1,
                        help="Dropout rate.")
    parser.add_argument('--max_epochs',
                        type=int,
                        default=100,
                        help="Maximum number of training epochs.")
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help="Batch size for dataloaders.")

    args = parser.parse_args()

    pairwise_layer_sizes = [
        (args.distance_mlp_layer_sizes[i],
         args.distance_mlp_layer_sizes[i + 1])
        for i in range(len(args.distance_mlp_layer_sizes) - 1)
    ]

    main(train_data_path=args.train_dataset_path,
         validation_data_path=args.validation_dataset_path,
         batch_size=args.batch_size,
         d_model=args.d_model,
         nhead=args.nhead,
         d_hid=args.d_hid,
         nlayers=args.nlayers,
         learning_rate=args.learning_rate,
         max_sequence_len=args.max_sequence_len,
         distance_mlp_layer_sizes=pairwise_layer_sizes,
         dropout=args.dropout,
         max_epochs=args.max_epochs)
