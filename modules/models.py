import torch
from torch import nn
from torch.nn import functional as F


class TimeLSTMNetClassifier(nn.Module):
    def __init__(self, INPUT_DIM, TIME_STEP, HIDDEN_DIM, N_LAYER, OUTPUT_DIM):
        super().__init__()
        self.INPUT_DIM = INPUT_DIM
        self.TIME_STEP = TIME_STEP
        self.HIDDEN_DIM = HIDDEN_DIM
        self.N_LAYER = N_LAYER
        self.OUTPUT_DIM = OUTPUT_DIM
        self.lstm = nn.LSTM(self.INPUT_DIM,self.HIDDEN_DIM,num_layers=self.N_LAYER,batch_first=True,dropout=0.0,bidirectional=False)
        self.fc = nn.Linear(self.HIDDEN_DIM,self.OUTPUT_DIM)
        assert self.TIME_STEP > 1, "TIME_STEP should be > 1"

    def forward(self, x):
        assert x.dtype == torch.float32, "x.dtype is not float32"
        if x.ndim == 1:
            x = x.unsqueeze(1)
        if x.ndim == 2:
            x = x.unsqueeze(2)
        x, _ = self.lstm(x, self.init_hidden(x.size(0)))
        x = x[:,-1,:]
        output = self.fc(x)
        return F.softmax(output, dim=1)

    def init_hidden(self, size):
        h0 = torch.zeros(self.N_LAYER, size, self.HIDDEN_DIM).requires_grad_()
        return h0.detach(), h0.detach()


class TimeConv1dLSTMNetClassifier(nn.Module):
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
        self.conv1d = nn.Conv1d(in_channels=self.INPUT_DIM,out_channels=64,kernel_size=FILTER_SIZE, stride=STRIDE, padding=PADDING)
        self.lstm = nn.LSTM(CNN_OUT_DIM, self.HIDDEN_DIM, num_layers=self.N_LAYER, batch_first=True, dropout=0.0, bidirectional=False)
        self.fc = nn.Linear(self.HIDDEN_DIM, self.OUTPUT_DIM)
        assert self.TIME_STEP > 1, "TIME_STEP should be > 1"

    def forward(self, x):
        # x.shape = batch_size / TIME_STEP / N_FEATURES
        assert x.dtype == torch.float32, "x.dtype is not float32"
        if x.ndim == 1:
            x = x.unsqueeze(1)
        if x.ndim == 2:
            x = x.unsqueeze(2)
        x = x.transpose(1, 2).contiguous() # batch_size/ N_FEATURES / TIME_STEP
        x = self.conv1d(x) # batch_size / 64 / TIME_STEP-1
        x, _ = self.lstm(x, self.init_hidden(x.size(0))) # batch_size / 64 / hidden
        x = x[:, -1, :] # batch_size / hidden
        output = self.fc(x) # batch_size / output_dim
        return F.softmax(output, dim=1)

    def init_hidden(self, size):
        h0 = torch.zeros(self.N_LAYER, size, self.HIDDEN_DIM).requires_grad_()
        return h0.detach(), h0.detach()

    def conv1d_output_dim(self, length_in, kernel_size, stride=1, padding=0, dilation=1):
        return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1



class TimeLSTMNetRegressor(nn.Module):
    def __init__(self, INPUT_DIM, TIME_STEP, HIDDEN_DIM, N_LAYER):
        super().__init__()
        self.INPUT_DIM = INPUT_DIM
        self.TIME_STEP = TIME_STEP
        self.HIDDEN_DIM = HIDDEN_DIM
        self.N_LAYER = N_LAYER
        self.OUTPUT_DIM = 1
        self.lstm = nn.LSTM(self.INPUT_DIM, self.HIDDEN_DIM, num_layers=self.N_LAYER, batch_first=True, dropout=0.0, bidirectional=False)
        self.fc = nn.Linear(self.HIDDEN_DIM, self.OUTPUT_DIM)
        assert self.TIME_STEP > 1, "TIME_STEP should be > 1"

    def forward(self, x):
        assert x.dtype == torch.float32, "x.dtype is not float32"
        if x.ndim == 1:
            x = x.unsqueeze(1)
        if x.ndim == 2:
            x = x.unsqueeze(2)
        x, _ = self.lstm(x, self.init_hidden(x.size(0)))
        x = x[:,-1,:]
        output = self.fc(x)
        return output

    def init_hidden(self, size):
        h0 = torch.zeros(self.N_LAYER, size, self.HIDDEN_DIM).requires_grad_()
        return h0.detach(), h0.detach()