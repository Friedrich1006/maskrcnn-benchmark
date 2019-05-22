"""
Implements Conv-LSTM
Modified from https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py
"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: int
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        """
        Forward pass for ConvLSTM Cell.

        Parameters
        ----------
        input_tensor:
            4-D tensor of shape (b, c, h, w).
        cur_state:
            List of two 4-D tensors of shape (b, c, h, w).

        Returns
        -------
        h_next, c_next:
            4-D tensors of shape (b, c, h, w).
        """

        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, height, width, device):
        return [torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device)]


class ConvLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size,
                 num_layers=1, bias=True, pretrain=None,
                 return_all_layers=True):
        """
        Initialize ConvLSTM.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: list[int] or int
            Number of channels of hidden state of each layer.
        kernel_size: list[(int, int)] or (int, int)
            Size of the convolutional kernel of each layer.
        num_layers: int
            Number of layers.
        bias: bool
            Whether or not to add the bias.
        pretrain: str
            Path to pretrained model.
        return_all_layers: bool
            Whether or not to return all layer states.
        """

        super(ConvLSTM, self).__init__()

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

        self.freezed = False
        if pretrain is not None:
            self.load_state_dict(torch.load(pretrain))
            self.freeze()

    def forward(self, input_tensor, hidden_state=None):
        """
        Forward pass for Conv LSTM.
        
        Parameters
        ----------
        input_tensor:
            4-D tensor of shape (b, c, h, w).
        hidden_state:
            List of num_layers lists of two 4-D tensors of shape (b, c, h, w).
            
        Returns
        -------
        last_state_list:
            List of num_layers lists of two 4-D tensors of shape (b, c, h, w).
        """

        if hidden_state is None:
            hidden_state = self._init_hidden(input_tensor.size(0),
                                             input_tensor.size(2),
                                             input_tensor.size(3),
                                             input_tensor.device)

        last_state_list = []

        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            h, c = self.cell_list[layer_idx](cur_layer_input, [h, c])
            output_inner.append(h)

            layer_output = torch.cat(output_inner, dim=0)
            cur_layer_input = layer_output

            last_state_list.append([h, c])

        if not self.return_all_layers:
            last_state_list = last_state_list[-1:]

        return last_state_list

    def freeze(self):
        for p in self.cell_list.parameters():
            p.requires_grad = False
        self.freezed = True

    def _init_hidden(self, batch_size, height, width, device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, height,
                                                             width, device))
        return init_states

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
