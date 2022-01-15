import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import spectral_norm

class GCN_Layer(nn.Module):

    def __init__(self, in_channel: int, out_channel: int, init_mode: str = 'glorot'):
        super(GCN_Layer, self).__init__()

        #self.in_channel = in_channel
        #self.out_channel = out_channel
        self.init_mode = init_mode

        self.weights = nn.parameter.Parameter(torch.FloatTensor(in_channel, out_channel))
        self.bias = nn.parameter.Parameter(torch.FloatTensor(out_channel))

        self._init_params()

    def _init_params(self):

        if self.init_mode == 'kaiming':
            nn.init.kaiming_uniform_(self.weights)
        elif self.init_mode == 'glorot':
            nn.init.xavier_uniform_(self.weights)
        else:
            raise ValueError("Unrecognized initalization mode!")

        self.bias.data.fill_(0.0)

    def forward(self, x: torch.tensor, A: torch.tensor):

        out = torch.mm(x, self.weights)
        out = torch.spmm(A, out)
        out = out + self.bias
        return out

    def lipschitz_norm(self):

        w = self.weights.data
        normed_weights = spectral_norm(w)
        self.weights = nn.parameter.Parameter(normed_weights)

        # b = self.bias.data
        # normed_bias = spectral_norm(b)
        # self.bias = nn.parameter.Parameter(normed_bias)


class MLP(nn.Module):

    def __init__(self, in_channel: int, out_channel: int, init_mode: str = 'glorot'):
        super(MLP, self).__init__()

        #self.in_channel = in_channel
        #self.out_channel = out_channel
        self.init_mode = init_mode

        self.weights = nn.parameter.Parameter(torch.FloatTensor(in_channel, out_channel))
        self.bias = nn.parameter.Parameter(torch.FloatTensor(out_channel))

        self._init_params()

    def _init_params(self):

        if self.init_mode == 'kaiming':
            nn.init.kaiming_uniform_(self.weights)
        elif self.init_mode == 'glorot':
            nn.init.xavier_uniform_(self.weights)
        else:
            raise ValueError("Unrecognized inital mode!")

        self.bias.data.fill_(0.0)

    def forward(self, x: torch.tensor):

        out = torch.mm(x, self.weights) + self.bias
        return out

    def lipschitz_norm(self):

        w = self.weights.data
        normed_weights = spectral_norm(w)
        self.weights = nn.parameter.Parameter(normed_weights)

        # b = self.bias.data
        # normed_bias = spectral_norm(b)
        # self.bias = nn.parameter.Parameter(normed_bias)


class GCN(nn.Module):

    def __init__(self, in_channel, num_hidden, out_channel, dropout):
        super(GCN, self).__init__()

        self.conv = GCN_Layer(in_channel, num_hidden)
        #self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(num_hidden, out_channel)

    def forward(self, x, A):
        out = self.conv(x, A)
        #out = self.dropout(out)
        out = self.fc(out)
        return out


class Encoder(nn.Module):

    def __init__(self, in_channel, num_hidden, out_channel, dropout):
        super(Encoder, self).__init__()

        self.gcn = GCN_Layer(in_channel, num_hidden)
        #self.dropout = nn.Dropout(p=dropout)
        self.linear1 = MLP(num_hidden, num_hidden)
        self.bn1 = nn.BatchNorm1d(num_hidden)
        self.linear2 = MLP(num_hidden, out_channel)
        self.bn2 = nn.BatchNorm1d(num_hidden)

    def forward(self, x: torch.tensor, A: torch.tensor):

        out = self.gcn(x, A)
        out = F.relu(self.bn1(self.linear1(out)))
        #out = F.relu(self.linear1(out))
        #out = self.dropout(out)
        out = self.bn2(self.linear2(out))
        return out

    def lipschitz_norm(self):
        self.gcn.lipschitz_norm()
        self.linear1.lipschitz_norm()
        self.linear2.lipschitz_norm()

class Predictor(nn.Module):

    def __init__(self, in_channel: int, num_hidden: int, out_channel: int):
        super(Predictor, self).__init__()

        self.linear1 = MLP(in_channel, num_hidden)
        self.linear2 = MLP(num_hidden, out_channel)
        self.bn = nn.BatchNorm1d(num_hidden)

    def forward(self, x: torch.tensor):
        out = F.relu(self.bn(self.linear1(x)))
        out = self.linear2(out)
        return out

    def lipschitz_norm(self):
        self.linear1.lipschitz_norm()
        self.linear2.lipschitz_norm()

class Classifier(nn.Module):

    def __init__(self, in_channel, num_class):
        super(Classifier, self).__init__()
        self.linear = MLP(in_channel, num_class)

    def forward(self, x: torch.tensor):
        return self.linear(x)

