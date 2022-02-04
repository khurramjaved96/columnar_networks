import torch
from torch import nn

class LSTMNet(torch.nn.Module):
    def __init__(self, inputs, hidden_units):
        """
        In the constructor we instantiate five parameters and assign them as members.
        """
        super().__init__( )
        self.recurrent_net = nn.LSTM(inputs, hidden_units, 1)
        self.predictor = nn.Linear(hidden_units, 1)
        self.predictor.weight.data *= 0
        self.predictor.bias.data *= 0

    def forward(self, x, state):

        features, state = self.recurrent_net(x, state)
        features = features.view(1, -1)
        output = self.predictor(features)
        return output.squeeze(), state

    def decay_gradients(self, decay_rate):
        for name, param in self.named_parameters():
            if param.grad is not None:
                # print(name)
                # print(param.grad)
                param.grad = param.grad*decay_rate
                # print(param.grad)
                # print("")