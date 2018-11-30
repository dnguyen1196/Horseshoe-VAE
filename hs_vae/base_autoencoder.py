import torch.nn as nn

class VAE(nn.Module):
    def __init__(self):
        pass

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def fit(self, X, Y):
        raise NotImplementedError

    def elbos(self):
        raise NotImplementedError

    def train_accuracy(self):
        raise NotImplementedError

    def test_accuracy(self):
        raise NotImplementedError