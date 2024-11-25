import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, device=None):
        super().__init__()
        self.conv = nn.Conv(in_channels, out_channels, kernel_size, stride=stride, device=device)
        self.bn = nn.BatchNorm2d(out_channels, device=device)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.relu(x)
        return out

class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.network = nn.Sequential(
            ConvBN(3, 16, 7, 4, device=device),
            ConvBN(16, 32, 3, 2, device=device),
            
            nn.Residual(nn.Sequential(
                ConvBN(32, 32, 3, 1, device=device),
                ConvBN(32, 32, 3, 1, device=device)
            )),
            
            ConvBN(32, 64, 3, 2, device=device),
            ConvBN(64, 128, 3, 2, device=device),
            
            nn.Residual(nn.Sequential(
                ConvBN(128, 128, 3, 1, device=device),
                ConvBN(128, 128, 3, 1, device=device)
            )),
            
            nn.Flatten(),
            nn.Linear(128, 128, device=device),
            nn.ReLU(),
            nn.Linear(128, 10, device=device)
        )

        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        out = self.network(x)
        return out
        ### END YOUR SOLUTION

# class ResNet9(nn.Module):
#     def __init__(self, device=None, dtype="float32"):
#         super().__init__()
        
#         # Initial convolutional layers
#         self.convbn1 = ConvBN(3, 16, 7, 4, device=device)
#         self.convbn2 = ConvBN(16, 32, 3, 2, device=device)
        
#         # Residual block 1
#         self.convbn3 = ConvBN(32, 32, 3, 1, device=device)
#         self.convbn4 = ConvBN(32, 32, 3, 1, device=device)
        
#         # Residual block 2
#         self.convbn5 = ConvBN(32, 64, 3, 2, device=device)
        
#         # Residual block 3
#         self.convbn6 = ConvBN(64, 128, 3, 2, device=device)
#         self.convbn7 = ConvBN(128, 128, 3, 1, device=device)
#         self.convbn8 = ConvBN(128, 128, 3, 1, device=device)
        
#         # Fully connected layers
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(128, 128, device=device)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(128, 10, device=device)

#     def forward(self, x):
#         # Initial convolutional layers
#         x = self.convbn1(x)
#         x = self.convbn2(x)
        
#         # First residual block
#         residual = x
#         x = self.convbn3(x)
#         x = self.convbn4(x)
#         x += residual  # Residual connection
        
#         # Second residual block
#         x = self.convbn5(x)
#         x = self.convbn6(x)
        
#         # Third residual block
#         residual = x
#         # x = self.convbn6(x)
#         x = self.convbn7(x)
#         x = self.convbn8(x)
#         x += residual  # Residual connection
        
#         # Fully connected layers
#         x = self.flatten(x)
#         x = self.fc1(x)
#         x = self.relu(x)
#         out = self.fc2(x)
        
#         return out

class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)

        if seq_model == "rnn":
            self.seq_model = nn.RNN(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        else:
            self.seq_model = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)

        self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape

        x = self.embed(x)

        x, h = self.seq_model(x, h)

        x = x.reshape((seq_len * bs, self.hidden_size))
        x = self.linear(x)

        return x, h
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)