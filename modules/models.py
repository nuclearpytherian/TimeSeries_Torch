import torch
from torch import nn
from torch.nn import functional as F


class TimeClassifierLSTMNet(nn.Module):
    def __init__(self, INPUT_DIM, HIDDEN_DIM, N_LAYER, OUTPUT_DIM):
        super().__init__()
        self.INPUT_DIM = INPUT_DIM
        self.HIDDEN_DIM = HIDDEN_DIM
        self.N_LAYER = N_LAYER
        self.OUTPUT_DIM = OUTPUT_DIM
        self.lstm = nn.LSTM(INPUT_DIM, HIDDEN_DIM, num_layers=N_LAYER, batch_first=True, dropout=0.0, bidirectional=False)
        self.fc = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, x):
        assert x.dtype == torch.float32, "x.dtype is not float32"
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x, _ = self.lstm(x, self.init_hidden(x.size(0)))
        x = x[:,-1,:]
        output = self.fc(x)
        return F.softmax(output, dim=1)

    def init_hidden(self, size):
        h0 = torch.zeros(self.N_LAYER, size, self.HIDDEN_DIM).requires_grad_()
        return h0.detach(), h0.detach()


class TimeClassifierConv1dLSTMNet(nn.Module):
    def __init__(self, INPUT_DIM, TIME_STEP, HIDDEN_DIM, N_LAYER, OUTPUT_DIM):
        super().__init__()
        self.INPUT_DIM = INPUT_DIM
        self.TIME_STEP = TIME_STEP
        self.HIDDEN_DIM = HIDDEN_DIM
        self.N_LAYER = N_LAYER
        self.OUTPUT_DIM = OUTPUT_DIM
        FILTER_SIZE = 2
        STRIDE = 1
        PADDING = 0
        CNN_OUT_DIM = self.conv1d_output_dim(self.TIME_STEP, FILTER_SIZE, STRIDE, PADDING)
        self.conv1d = nn.Conv1d(in_channels=INPUT_DIM, out_channels=64, kernel_size=FILTER_SIZE, stride=STRIDE, padding=PADDING)
        self.lstm = nn.LSTM(CNN_OUT_DIM, HIDDEN_DIM, num_layers=N_LAYER, batch_first=True, dropout=0.0, bidirectional=False)
        self.fc = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, x):
        assert x.dtype == torch.float32, "x.dtype is not float32"
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = x.transpose(1, 2).contiguous()
        x = self.conv1d(x)
        x, _ = self.lstm(x, self.init_hidden(x.size(0)))
        x = x[:, -1, :]
        output = self.fc(x)
        return F.softmax(output, dim=1)

    def init_hidden(self, size):
        h0 = torch.zeros(self.N_LAYER, size, self.HIDDEN_DIM).requires_grad_()
        return h0.detach(), h0.detach()

    def conv1d_output_dim(self, length_in, kernel_size, stride=1, padding=0, dilation=1):
        return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
