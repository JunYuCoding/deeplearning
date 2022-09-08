import torch
from torch import nn
from torch.nn import ReLU

input = torch.Tensor([[1, -0.5],
                      [-1, 3]])
print(input.shape)
input = torch.reshape(input, (-1, 1, 2, 2,))
print(input.shape)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.relu1 = ReLU()

    def forward(self, input):
        output = self.relu1(input)
        return output
cnn = CNN()
output = cnn(input)
print(output)
print(output.shape)