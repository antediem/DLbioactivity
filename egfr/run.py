
import torch
import torch.nn as nn
import tensorboard_logger
from nets import CnnNet, DenseNet, CombinedNet
from torch.utils.data import dataloader
from dataset import EGFRDataset, train_validation_split
import torch.optim as optim
from metrics import auc
import collections


def get_max_length(x):
    return len(max(x, key=len))


def pad_sequence(seq):
    def _pad(_it, _max_len):
        return [0] * (_max_len - len(_it)) + _it
    padded = [_pad(it, get_max_length(seq)) for it in seq]
    return padded


def custom_collate(batch):
    """
        Custom collate function for our batch, a batch in dataloader looks like
            [(0, [24104, 27359], 6684),
            (0, [24104], 27359),
            (1, [16742, 31529], 31485),
            (1, [16742], 31529),
            (2, [6579, 19316, 13091, 7181, 6579, 19316], 13091)]
    """
    transposed = zip(*batch)
    lst = []
    for samples in transposed:
        if isinstance(samples[0], int):
            lst.append(torch.LongTensor(samples))
        elif isinstance(samples[0], float):
            lst.append(torch.DoubleTensor(samples))
        elif isinstance(samples[0], collections.Sequence):
            lst.append(torch.LongTensor(pad_sequence(samples)))
    return lst


def train_validate(train_dataset,
                   val_dataset,
                   train_device,
                   val_device,
                   n_epoch,
                   batch_size,
                   metrics,
                   hash_code):
    train_loader = dataloader.DataLoader(dataset=train_dataset,
                                         batch_size=batch_size,
                                         collate_fn=custom_collate,
                                         shuffle=True)

    val_loader = dataloader.DataLoader(dataset=val_dataset,
                                       batch_size=batch_size,
                                       collate_fn=custom_collate,
                                       shuffle=True)

    tensorboard_logger.configure('logs/' + hash_code)

    criterion = nn.BCELoss()
    cnn_net = CnnNet().to(train_device)
    dense_net = DenseNet(input_dim=train_dataset.get_dim('mord')).to(train_device)
    combined_net = CombinedNet().to(train_device)

    opt = optim.SGD(list(cnn_net.parameters()) + list(dense_net.parameters()) + list(combined_net.parameters()),
                    lr=1e-5,
                    momentum=0.99)
    for e in range(n_epoch):
        train_losses = []
        val_losses = []
        train_outputs = []
        val_outputs = []
        train_labels = []
        val_labels = []
        print('TRAINING ==============>')
        for i, (mord_ft, non_mord_ft, label) in enumerate(train_loader):
            mord_ft = mord_ft.float().to(train_device)
            non_mord_ft = non_mord_ft.view((-1, 1, 42, 150)).float().to(train_device)
            label = label.float().to(train_device)

            # Forward
            opt.zero_grad()
            non_mord_output = cnn_net(non_mord_ft)
            mord_output = dense_net(mord_ft)

            combined_output = combined_net(mord_output, non_mord_output)
            loss = criterion(combined_output, label)
            train_losses.append(float(loss.item()))
            train_outputs.extend(combined_output)
            train_labels.extend(label)

            # Parameters update
            loss.backward()
            opt.step()

        # Validate after each epoch
        print('VALIDATION ==============>')
        for i, (mord_ft, non_mord_ft, label) in enumerate(val_loader):
            mord_ft = mord_ft.float().to(val_device)
            non_mord_ft = non_mord_ft.view((-1, 1, 42, 150)).float().to(val_device)
            label = label.float().to(val_device)

            with torch.no_grad():
                opt.zero_grad()
                mord_output = cnn_net(non_mord_ft)
                non_mord_output = dense_net(mord_ft)
                combined_output = combined_net(mord_output, non_mord_output)
                loss = criterion(combined_output, label)
                val_losses.append(float(loss.item()))
                val_outputs.extend(combined_output)
                val_labels.extend(label)

        train_outputs = torch.stack(train_outputs)
        val_outputs = torch.stack(val_outputs)
        train_labels = torch.stack(train_labels)
        val_labels = torch.stack(val_labels)
        tensorboard_logger.log_value('train_loss', sum(train_losses) / len(train_losses), e + 1)
        tensorboard_logger.log_value('val_loss', sum(val_losses) / len(val_losses), e + 1)

        for key in metrics.keys():
            train_metric = metrics[key](train_labels, train_outputs)
            val_metric = metrics[key](val_labels, val_outputs)
            tensorboard_logger.log_value('train_{}'.format(key),
                                         train_metric, e + 1)
            tensorboard_logger.log_value('val_{}'.format(key),
                                         train_metric, e + 1)


def main():
    train_data, val_data = train_validation_split('data/egfr_10_full_ft_pd_lines.json')
    train_dataset = EGFRDataset(train_data)
    val_dataset = EGFRDataset(val_data)
    train_device = 'cpu'
    val_device = 'cpu'
    train_validate(train_dataset,
                   val_dataset,
                   train_device,
                   val_device,
                   500,
                   128,
                   {'auc': auc},
                   'TEST')


if __name__ == '__main__':
    main()




