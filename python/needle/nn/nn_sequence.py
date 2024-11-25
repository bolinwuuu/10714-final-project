"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return (ops.exp(-x) + 1)**(-1)
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity

        k_root = 1 / hidden_size**0.5
        
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-k_root, high=k_root, device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-k_root, high=k_root, device=device, dtype=dtype, requires_grad=True))

        self.bias_ih = Parameter(init.rand(hidden_size, low=-k_root, high=k_root, device=device, dtype=dtype, requires_grad=True)) if bias else None
        self.bias_hh = Parameter(init.rand(hidden_size, low=-k_root, high=k_root, device=device, dtype=dtype, requires_grad=True)) if bias else None

        self.device = device
        self.dtype = dtype
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ## BEGIN YOUR SOLUTION
        bs, _ = X.shape
        if h is None:
            h = init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
            # h = Tensor(self.device.full((bs, self.hidden_size), 0, dtype=self.dtype), device=self.device)

        h_next = ops.matmul(X, self.W_ih)
        if self.bias_ih is not None:
            h_next += self.bias_ih.reshape((1, self.hidden_size)).broadcast_to(h_next.shape)
        h_next += ops.matmul(h, self.W_hh)
        if self.bias_hh is not None:
            h_next += self.bias_hh.reshape((1, self.hidden_size)).broadcast_to(h_next.shape)

        if self.nonlinearity == 'tanh':
            h_next = ops.tanh(h_next)
        else:
            h_next = ops.relu(h_next)
        
        return h_next
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn_cells = []
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.rnn_cells.append(RNNCell(layer_input_size, hidden_size, bias=bias, \
              nonlinearity=nonlinearity, device=device, dtype=dtype))
        
        self.device = device
        self.dtype = dtype
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, input_size = X.shape

        if h0 is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype)
            
        h0_split = ops.split(h0, axis=0)

        h_n = []
        X_split = list(ops.split(X, axis=0))
        for i in range(self.num_layers):
            h = h0_split[i]
            for j in range(len(X_split)):
                h = self.rnn_cells[i](X_split[j], h)
                X_split[j] = h
            h_n.append(h)
        
        X_out = ops.stack(X_split, axis=0)
        h_n_out = ops.stack(h_n, axis=0)
        return X_out, h_n_out
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype

        k_root = 1 / hidden_size**0.5

        self.W_ih = Parameter(init.rand(input_size, 4 * hidden_size, low=-k_root, high=k_root, device=device, dtype=dtype, requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, 4 * hidden_size, low=-k_root, high=k_root, device=device, dtype=dtype, requires_grad=True))

        self.bias_ih = Parameter(init.rand(4 * hidden_size, low=-k_root, high=k_root, device=device, dtype=dtype, requires_grad=True)) if bias else None
        self.bias_hh = Parameter(init.rand(4 * hidden_size, low=-k_root, high=k_root, device=device, dtype=dtype, requires_grad=True)) if bias else None

        self.sigmoid = Sigmoid()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs = X.shape[0]
        
        if h is None:
            h_t = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype)
            c_t = init.zeros(bs, self.hidden_size, device=X.device, dtype=X.dtype)
        else:
            h_t, c_t = h

        temp = X @ self.W_ih
        temp += h_t @ self.W_hh
        if self.bias_ih is not None:
            temp += self.bias_ih.reshape((1, 4 * self.hidden_size)).broadcast_to(temp.shape)
        if self.bias_hh is not None:
            temp += self.bias_hh.reshape((1, 4 * self.hidden_size)).broadcast_to(temp.shape)

        gates = temp # (bs, 4 * hidden_size)

        i_gate, f_gate, g_gate, o_gate = ops.split(gates.reshape((bs, 4, self.hidden_size)), axis=1)

        i = self.sigmoid(i_gate).reshape(c_t.shape)
        f = self.sigmoid(f_gate).reshape(c_t.shape)
        g = ops.tanh(g_gate).reshape(c_t.shape)
        o = self.sigmoid(o_gate).reshape(c_t.shape)

        c_next = f * c_t + i * g
        h_next = o * ops.tanh(c_next)

        return h_next, c_next
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype

        self.lstm_cells = []
        for layer in range(num_layers):
            cell_input_size = input_size if layer == 0 else hidden_size
            self.lstm_cells.append(
                LSTMCell(cell_input_size, hidden_size, bias=bias, device=device, dtype=dtype)
            )
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape

        if h is None:
            h = (
                init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype),
                init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype),
            )
        h_states, c_states = h

        outputs = []
        layer_h_states = []
        layer_c_states = []

        X_split = list(ops.split(X, axis=0))
        h_split = list(ops.split(h_states, axis=0))
        c_split = list(ops.split(c_states, axis=0))
        for t in range(seq_len):
            x_t = X_split[t]
            layer_input = x_t

            for layer in range(self.num_layers):
                h_t, c_t = self.lstm_cells[layer](layer_input, (h_split[layer], c_split[layer]))
                layer_input = h_t
                h_split[layer] = h_t
                c_split[layer] = c_t

            outputs.append(h_t)

        output = ops.stack(outputs, axis=0)

        h_n = ops.stack(h_split, axis=0)
        c_n = ops.stack(c_split, axis=0)

        return output, (h_n, c_n)
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, mean=0, std=1, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        X = init.one_hot(self.num_embeddings, x, device=x.device, dtype=x.dtype).reshape((seq_len * bs, self.num_embeddings))

        res = (X @ self.weight).reshape((seq_len, bs, self.embedding_dim))

        return res
        ### END YOUR SOLUTION