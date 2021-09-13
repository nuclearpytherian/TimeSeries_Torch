
import numpy as np
import pandas as pd

def data_to_series(X, y, TIME_STEP):
    N = len(X)
    output_X = []
    output_y = []
    for i in range(TIME_STEP, N):
        t = []
        for j in range(TIME_STEP):
            t.append(X[[j], :])
        output_X.append(t)
        output_y.append(y[i])
    return np.squeeze(np.array(output_X)), np.array(output_y)

def Conv1d_output_dim(length_in, kernel_size, stride=1, padding=0, dilation=1):
    return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

def random_pandasDF(N, N_FEATURES):
    data = pd.DataFrame(np.random.normal(20, 5, (N, N_FEATURES)))
    data['y'] = [x for x in range(N)]
    return data