import torch
import torch.nn as nn
import math


# a simple but versatile d1 convolutional neural net
class ConvNet1d(nn.Module):

    def __init__(self, in_channels: int, hidden_channels: list,
                 kernel_lengths: list, dropout=None, stride=1, dilation=1, batch_norm=False):
        super().__init__()
        assert len(hidden_channels) == len(kernel_lengths)

        self.in_channels = in_channels

        layers = []
        num_of_layers = len(hidden_channels)
        layer_in_channels = in_channels

        for i in range(num_of_layers):

            layer_out_channels = hidden_channels[i]
            layers.append(nn.Conv1d(layer_in_channels, layer_out_channels, kernel_size=kernel_lengths[i],
                                    stride=stride, dilation=dilation))
            if batch_norm:
                layers.append(nn.BatchNorm1d(layer_out_channels))
            if dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())

            layer_in_channels = layer_out_channels

        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn(x)


class Ecg12LeadNet(nn.Module):
    def forward(self, x):
        x1, x2 = x
        out1 = self.short_cnn(x1).reshape((x1.shape[0], -1))
        out2 = self.long_cnn(x2).reshape((x2.shape[0], -1))
        out = torch.cat((out1, out2), 1)
        return self.fc(out)

    def __init__(self,
                 short_hidden_channels: list, long_hidden_channels: list,
                 short_kernel_lengths: list, long_kernel_lengths: list,
                 fc_hidden_dims: list,
                 short_dropout=None, long_dropout=None,
                 short_stride=1, long_stride=1,
                 short_dilation=1, long_dilation=1,
                 short_batch_norm=False, long_batch_norm=False,
                 short_input_length=1250, long_input_length=5000,
                 num_of_classes=2
                 ):

        super().__init__()
        assert len(short_hidden_channels) == len(short_kernel_lengths)
        assert len(long_hidden_channels) == len(long_kernel_lengths)

        self.short_cnn = ConvNet1d(12, short_hidden_channels, short_kernel_lengths, short_dropout,
                                   short_stride, short_dilation, short_batch_norm)
        self.long_cnn = ConvNet1d(1, long_hidden_channels, long_kernel_lengths, long_dropout,
                                  long_stride, long_dilation, long_batch_norm)

        short_out_channels = short_hidden_channels[-1]
        short_out_dim = short_out_channels * self.calc_out_length(short_input_length, short_kernel_lengths,
                                                                  short_stride, short_dilation)
        long_out_channels = long_hidden_channels[-1]
        long_out_dim = long_out_channels * self.calc_out_length(long_input_length, long_kernel_lengths,
                                                                long_stride, long_dilation)

        in_dim = short_out_dim + long_out_dim
        layers = []
        for out_dim in fc_hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim

        # single score for binary classification, class score for multi-class
        if num_of_classes == 2:
            layers.append(nn.Linear(in_dim, 1))
        else:
            layers.append(nn.Linear(in_dim, num_of_classes))

        self.fc = nn.Sequential(*layers)

    @staticmethod
    def calc_out_length(l_in, kernel_lengths, stride, dilation):
        l_out = l_in
        for kernel in kernel_lengths:
            l_out = math.floor((l_out - dilation * (kernel - 1) - 1) / stride + 1)
        return l_out


class Ecg12ImageNet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: list, kernel_sizes: list, in_h: int, in_w: int,
                 fc_hidden_dims: list, dropout=None, stride=1, dilation=1, batch_norm=False, num_of_classes=2):
        super().__init__()
        assert len(hidden_channels) == len(kernel_sizes)

        self.in_channels = in_channels

        layers = []
        num_of_layers = len(hidden_channels)
        layer_in_channels = in_channels

        for i in range(num_of_layers):

            layer_out_channels = hidden_channels[i]
            layers.append(nn.Conv2d(layer_in_channels, layer_out_channels, kernel_size=kernel_sizes[i],
                                    stride=stride, dilation=dilation))
            if batch_norm:
                layers.append(nn.BatchNorm2d(layer_out_channels))
            if dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())

            layer_in_channels = layer_out_channels

        self.cnn = nn.Sequential(*layers)

        out_channels = hidden_channels[-1]
        out_h = self.calc_out_length(in_h, kernel_sizes, stride, dilation)
        out_w = self.calc_out_length(in_w, kernel_sizes, stride, dilation)
        in_dim = out_channels * out_h * out_w
        print('Input dim to the fc layer:', in_dim)

        layers = []
        for out_dim in fc_hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim

        # single score for binary classification, class score for multi-class
        if num_of_classes == 2:
            layers.append(nn.Linear(in_dim, 1))
        else:
            layers.append(nn.Linear(in_dim, num_of_classes))

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        out = self.cnn(x)
        out = out.reshape((x.shape[0], -1))
        return self.fc(out)

    @staticmethod
    def calc_out_length(l_in, kernel_lengths, stride, dilation):
        l_out = l_in
        for kernel in kernel_lengths:
            l_out = math.floor((l_out - dilation * (kernel - 1) - 1) / stride + 1)
        return l_out
