import torch


class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn1 = torch.nn.RNN(784, 100, 3, nonlinearity='relu')
        self.rnn2 = torch.nn.RNN(100, 10, 1, nonlinearity='relu')

    def forward(self, x):
        x = self.rnn1(x)
        x = torch.Tensor(x[1])
        x = self.rnn2(x)
        return x


model = RNN()
print(model)
