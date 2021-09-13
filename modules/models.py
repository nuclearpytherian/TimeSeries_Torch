import torch
from torch import nn
from utils.utils import Conv1d_output_dim

# LSTM for Time-Series
class LSTMNet(nn.Module):
    def __init__(self, INPUT_DIM, TIME_STEP, HIDDEN_DIM, N_LAYER, OUTPUT_DIM):
        super().__init__()
        self.INPUT_DIM = INPUT_DIM
        self.TIME_STEP = TIME_STEP # not used in LSTMNet
        self.HIDDEN_DIM = HIDDEN_DIM
        self.N_LAYER = N_LAYER
        self.OUTPUT_DIM = OUTPUT_DIM
        self.lstm = nn.LSTM(INPUT_DIM, HIDDEN_DIM, num_layers=N_LAYER, batch_first=True, dropout=0.0, bidirectional=False)
        self.fc = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, x):
        assert x.dtype == torch.float32, "x.dtype is not float32"
        h0 = torch.zeros(self.N_LAYER, x.shape[0], self.HIDDEN_DIM).requires_grad_()
        c0 = torch.zeros(self.N_LAYER, x.shape[0], self.HIDDEN_DIM).requires_grad_()
        x, _ = self.lstm(x, (h0.detach(), c0.detach()))
        x = x[:,-1,:]
        output = self.fc(x)
        return output


class Conv1dLSTMNet(nn.Module):
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
        CNN_OUT_DIM = Conv1d_output_dim(self.TIME_STEP, FILTER_SIZE, STRIDE, PADDING)
        self.conv1d = nn.Conv1d(in_channels=INPUT_DIM, out_channels=64, kernel_size=FILTER_SIZE, stride=STRIDE, padding=PADDING)
        self.lstm = nn.LSTM(CNN_OUT_DIM, HIDDEN_DIM, num_layers=N_LAYER, batch_first=True, dropout=0.0, bidirectional=False)
        self.fc = nn.Linear(HIDDEN_DIM, OUTPUT_DIM)

    def forward(self, x):
        assert x.dtype == torch.float32, "x.dtype is not float32"
        x1 = x.transpose(1, 2).contiguous()
        x2 = self.conv1d(x1)
        h0 = torch.zeros(self.N_LAYER, x2.shape[0], self.HIDDEN_DIM).requires_grad_()
        c0 = torch.zeros(self.N_LAYER, x2.shape[0], self.HIDDEN_DIM).requires_grad_()
        x3, _ = self.lstm(x2, (h0.detach(), c0.detach()))
        x4 = x3[:, -1, :]
        output = self.fc(x4)
        return output

